import os
import torch
from data import SignalDataSet
from lightning.pytorch import Trainer
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows.iaf import InverseAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler

os.environ["SCRATCH_DIR"] = "/scratch"

# Distillation loss function
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

# Training function
def train_func(config):
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 800
    batches_per_epoch = 50
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
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
    num_transforms_student = config['num_transforms_student']
    num_blocks_student = 4
    hidden_features_student = config['hidden_features_student']
    learning_rate = config['learning_rate']
    alpha = config['alpha']
    temperature = config['temperature']
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
    teacher_checkpoint_path = os.getenv("BASE_DIR") + "/checkpoints2/teacher_model_checkpoint.ckpt"
    teacher_model = InverseAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=50,  # Use pre-trained teacher
        num_blocks=4,
        hidden_features=100
    )
    teacher_model.load_state_dict(torch.load(teacher_checkpoint_path)['state_dict'])
    teacher_model.eval()

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

    # 蒸馏步骤
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
            alpha=alpha,
            temperature=temperature
        )
        return loss

    # Trainer 设置
    trainer = Trainer(
        max_epochs=200,
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model=student_model, datamodule=sig_dat)


# 超参数调优
def tune_with_asha(ray_trainer, num_samples=10, num_epochs=15):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=3, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

if __name__ == '__main__':
    import torch
    print("CUDA available", torch.cuda.is_available())
    ray.init(configure_logging=False)

    # 定义超参数搜索空间
    search_space = {
        "num_transforms_student": tune.choice([30, 40]),
        "hidden_features_student": tune.choice([60, 80]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "alpha": tune.uniform(0.1, 0.9),  # 蒸馏损失中的 alpha
        "temperature": tune.choice([2.0, 3.0, 4.0])  # 蒸馏温度
    }

    # Scaling and run config
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path=os.getenv("SCRATCH_DIR") + "/ray_results",
        name="distill_tune_expt",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="valid_loss",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    results = tune_with_asha(ray_trainer, num_samples=30, num_epochs=30)
    print("Best hyperparameters found were: ", results.get_best_result().config)