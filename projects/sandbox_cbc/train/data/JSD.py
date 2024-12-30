import os
import torch
import torch.nn.functional as F  # 添加这行导入
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows.iaf import InverseAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging


# 蒸馏损失计算函数
def distillation_loss(student_outputs, teacher_outputs, ground_truth, alpha, temperature):
    """计算蒸馏损失"""
    student_loss = torch.nn.functional.nll_loss(student_outputs, ground_truth)

    teacher_probs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)
    student_probs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=-1)

    kl_loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    total_loss = alpha * student_loss + (1 - alpha) * kl_loss
    return total_loss

# JSD计算函数
def jsd(P, Q, eps=1e-10):
    """计算两个分布 P 和 Q 的 Jensen-Shannon Divergence"""
    P = P + eps
    Q = Q + eps

    M = 0.5 * (P + Q)

    kl_pm = F.kl_div(torch.log(P), M, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(Q), M, reduction='batchmean')

    jsd_value = 0.5 * (kl_pm + kl_qm)
    return jsd_value

# 比较教师模型和学生模型权重分布的JSD
def compare_model_weights_jsd(teacher_model, student_model):
    """比较教师模型和学生模型的所有权重，计算它们的Jensen-Shannon Divergence (JSD)"""
    jsd_values = []
    
    for (teacher_param, student_param) in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_weights = teacher_param.data
        student_weights = student_param.data

        teacher_weights = teacher_weights.view(-1)
        student_weights = student_weights.view(-1)

        teacher_probs = F.softmax(teacher_weights, dim=0)
        student_probs = F.softmax(student_weights, dim=0)

        jsd_value = jsd(teacher_probs, student_probs)
        jsd_values.append(jsd_value.item())
        
    return jsd_values

def main():
    # 参数定义
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 800
    batches_per_epoch = 200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    learning_rate = 0.005929470937010296
    resnet_context_dim = 100
    resnet_layers = [5, 5]
    resnet_norm_groups = 8
    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]
    num_transforms_teacher = 50
    num_blocks_teacher = 4
    hidden_features_teacher = 100
    num_transforms_student = 30
    num_blocks_student = 4
    hidden_features_student = 80

    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    # 初始化嵌入网络
    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    # 教师模型
    teacher_model = InverseAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=num_transforms_teacher,
        num_blocks=num_blocks_teacher,
        hidden_features=hidden_features_teacher
    )
    checkpoint_callback_teacher = ModelCheckpoint(
        dirpath=os.getenv("BASE_DIR") + "/checkpoints2/",
        filename="teacher_model_checkpoint",
        save_top_k=1,
        monitor="valid_loss",
        mode="min"
    )
    # 教师模型检查点
    ckpt_dir = os.path.join(os.getenv("BASE_DIR"), "teacher_model_ckpt")
    teacher_checkpoint_path = os.path.join(ckpt_dir, "teacher_model_checkpoint.ckpt")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=1000, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-60-transforms-4-4-resnet-wider-dl")

    if os.path.exists(teacher_checkpoint_path):
        print(f"加载教师模型的权重: {teacher_checkpoint_path}")
        checkpoint = torch.load(teacher_checkpoint_path)
        teacher_model.load_state_dict(checkpoint['state_dict'])
        teacher_model.eval()
    else:
        print("未找到教师模型的权重，开始训练教师模型...")
        os.makedirs(ckpt_dir, exist_ok=True)
        teacher_trainer = Trainer(
            max_epochs=200,
            log_every_n_steps=100,
            callbacks=[early_stop_cb, lr_monitor, checkpoint_callback_teacher],
            logger=logger,
            gradient_clip_val=10.0,
            default_root_dir=ckpt_dir,
        )
        teacher_trainer.fit(model=teacher_model, datamodule=SignalDataSet(
            background_path, ifos, valid_frac, batch_size, batches_per_epoch,
            sample_rate, time_duration, f_min, f_max, f_ref,
            prior_func=prior_func, approximant=IMRPhenomD
        ))
        teacher_trainer.save_checkpoint(teacher_checkpoint_path)
        print(f"教师模型权重已保存至: {teacher_checkpoint_path}")

    # 冻结教师模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 学生模型
    student_model = InverseAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=num_transforms_student,
        num_blocks=num_blocks_student,
        hidden_features=hidden_features_student
    )
    checkpoint_callback_student = ModelCheckpoint(
        dirpath=os.getenv("BASE_DIR") + "/checkpoints2/",
        filename="student_model_checkpoint",
        save_top_k=1,
        monitor="valid_loss",
        mode="min"
    )
    sig_dat = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    sig_dat.setup(None)

    # 蒸馏训练
    def distillation_step(batch, batch_idx):
        strain, parameters = batch
        teacher_outputs = teacher_model(strain, parameters)
        student_outputs = student_model(strain, parameters)
        loss = distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            ground_truth=parameters,
            alpha=0.767224603337332,
            temperature=4.0
        )
        return loss

    student_trainer = Trainer(
        max_epochs=200,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor, checkpoint_callback_student],
        logger=logger,
        gradient_clip_val=10.0,
        default_root_dir=ckpt_dir,
    )
    student_trainer.fit(model=student_model, datamodule=sig_dat)

    # 比较教师模型和学生模型的权重分布差异
    jsd_results = compare_model_weights_jsd(teacher_model, student_model)

    # 输出每一层的 JSD 值
    for i, jsd_value in enumerate(jsd_results):
        print(f"第 {i + 1} 层的 JSD: {jsd_value:.4f}")

if __name__ == '__main__':
    main()