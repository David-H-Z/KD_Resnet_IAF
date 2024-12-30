import os
import torch
import random
import numpy as np
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows.iaf import InverseAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging

def distillation_loss(student_outputs, teacher_outputs, ground_truth, alpha, temperature):
    """计算蒸馏损失"""
    # 监督损失：学生模型对真实数据的损失
    student_loss = torch.nn.functional.nll_loss(student_outputs, ground_truth)

    # 蒸馏损失：教师模型和学生模型的输出概率分布的KL散度
    teacher_probs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)
    student_probs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=-1)

    # KL散度作为蒸馏损失
    kl_loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # 总损失为监督损失和蒸馏损失的加权和
    total_loss = alpha * student_loss + (1 - alpha) * kl_loss
    return total_loss

def set_random_seed(seed):
    random.seed(seed)                 # Python的随机种子
    np.random.seed(seed)              # NumPy的随机种子
    torch.manual_seed(seed)           # PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)      # PyTorch的GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    # 确保CuDNN使用确定性算法（会降低训练速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 环境变量和参数定义
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
    learning_rate = 0.001
    resnet_context_dim = 100
    resnet_layers = [3, 5]
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
    num_transforms_teacher = 70
    num_blocks_teacher = 4
    hidden_features_teacher = 120
    num_transforms_student = 20
    num_blocks_student = 4
    hidden_features_student = 60

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

    # 创建教师模型
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
    # 检查是否已有保存的教师模型权重
    ckpt_dir = os.path.join(os.getenv("BASE_DIR"), "teacher_model_ckpt")
    teacher_checkpoint_path = os.path.join(ckpt_dir, "teacher_model_checkpoint_0.001_70_120.ckpt")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=1000, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-60-transforms-4-4-resnet-wider-dl")

    if os.path.exists(teacher_checkpoint_path):
        # 如果权重文件存在，加载教师模型的 state_dict
        print(f"加载教师模型的权重: {teacher_checkpoint_path}")
        checkpoint = torch.load(teacher_checkpoint_path)  # 加载 checkpoint 文件
        teacher_model.load_state_dict(checkpoint['state_dict'])  # 只加载 state_dict 部分
        teacher_model.eval()  # 确保模型在评估模式下
    else:
        # 如果权重文件不存在，训练教师模型并保存权重
        print("未找到教师模型的权重，开始训练教师模型...")
        os.makedirs(ckpt_dir, exist_ok=True)
        teacher_trainer = Trainer(
            max_epochs=150,  # 适当减少教师模型的训练轮数
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
    
        # 保存教师模型的完整 checkpoint
        teacher_trainer.save_checkpoint(teacher_checkpoint_path)
        print(f"教师模型权重已保存至: {teacher_checkpoint_path}")


    # 冻结教师模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 创建学生模型
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
        filename="student_model_checkpoint_20_4_60",
        save_top_k=1,
        monitor="valid_loss",
        mode="min"
    )
    # 数据集
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

    # 学生模型训练（蒸馏）
    def distillation_step(batch, batch_idx):
        strain, parameters = batch
        # 获取教师模型的输出
        teacher_outputs = teacher_model(strain, parameters)

        # 获取学生模型的输出
        student_outputs = student_model(strain, parameters)

        # 计算蒸馏损失
        loss = distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            ground_truth=parameters,
            alpha=0.75,
            temperature=4.0
        )
        return loss

    # 配置训练器
    student_trainer = Trainer(
        max_epochs=150,  # 适当减少学生模型的训练轮数
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor, checkpoint_callback_student],
        logger=logger,
        gradient_clip_val=10.0,
        default_root_dir=ckpt_dir,
    )
    student_trainer.fit(model=student_model, datamodule=sig_dat)

if __name__ == '__main__':
    seed = 42  # 您可以选择任意整数值作为随机种子
    set_random_seed(seed)
    main()
