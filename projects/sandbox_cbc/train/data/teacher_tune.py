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

# 教师模型的训练函数
def train_func(config):
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = config['batch_size']
    batches_per_epoch = 50
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    resnet_context_dim = 100
    resnet_layers = [4, 4]
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

    # 从 config 获取教师模型的超参数
    num_transforms_teacher = config['num_transforms_teacher']
    num_blocks_teacher = config['num_blocks_teacher']
    hidden_features_teacher = config['hidden_features_teacher']
    
    learning_rate = config['learning_rate']

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
        num_transforms=num_transforms_teacher,  # 进行调优
        num_blocks=num_blocks_teacher,          # 进行调优
        hidden_features=hidden_features_teacher # 进行调优
    )

    # 数据集 (SignalDataSet)
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
    trainer.fit(model=teacher_model, datamodule=sig_dat)

# 使用 ASHA 进行超参数调优
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

    # 教师模型的超参数搜索空间
    search_space = {
        "num_transforms_teacher": tune.choice([50, 60, 70]),
        "num_blocks_teacher": tune.choice([4, 5, 6]),
        "hidden_features_teacher": tune.choice([100, 120, 140]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([500, 800, 1000])
    }

    # Scaling 和运行配置
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path=os.getenv("SCRATCH_DIR") + "/ray_results",
        name="teacher_model_tune_expt",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="valid_loss",
            checkpoint_score_order="min",
        ),
    )

    # 定义 TorchTrainer 进行教师模型超参数调优
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    results = tune_with_asha(ray_trainer, num_samples=20, num_epochs=30)
    print("Best hyperparameters found were: ", results.get_best_result().config)
