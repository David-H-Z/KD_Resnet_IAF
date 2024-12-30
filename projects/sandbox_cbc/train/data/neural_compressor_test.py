import os

import torch
import torch.nn as nn
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI
import bilby
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet, QuantizedResNet
from ml4gw.nn.resnet import ResNet1D
from mlpe.architectures.embeddings import DenseNet1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging


from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor import quantization
from neural_compressor.utils.pytorch import load

def main():
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
    valid_frac = 0.2
    learning_rate = 1e-3
    resnet_context_dim = 100
    resnet_layers = [4, 4]
    resnet_norm_groups = 8

    inference_params = [
        #"mass_1",
        #"mass_2",
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
       # "ra",
    ]
    num_transforms = 60
    num_blocks = 5
    hidden_features = 120

    optimizer = torch.optim.AdamW
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )


    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )
    
    embedding = QuantizedResNet(
        (n_ifos, strain_dim),
         context_dim=resnet_context_dim,
         layers=resnet_layers,
         norm_groups=resnet_norm_groups,
    )

   
    #prior_func = nonspin_bbh_component_mass_parameter_sampler
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features
    ) 


    ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_51/checkpoints/epoch=323-step=64800.ckpt"
    ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_815/checkpoints/epoch=232-step=46600.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    flow_obj.load_state_dict(checkpoint["state_dict"])

    # def wrap_to_quantized_resnet(resnet_instance):
    #     quantized_resnet = QuantizedResNet(
    #          (n_ifos, strain_dim),
    #          context_dim=resnet_context_dim,
    #          layers=resnet_layers,
    #          norm_groups=resnet_norm_groups,
    #     )
        
    #     quantized_resnet.__dict__.update(resnet_instance.__dict__)
    #     quantized_resnet.quant = torch.ao.quantization.QuantStub()
    #     quantized_resnet.dequant = torch.ao.quantization.DeQuantStub()
    #     return quantized_resnet
    
    # flow_obj.embedding_net = wrap_to_quantized_resnet(flow_obj.embedding_net)
    
    # data
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
    
    print("##### Initialized data loader, calling setup ####")
    # sig_dat.device="cpu"
    sig_dat.setup(None)

    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=50, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR") 
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-60-transforms-4-4-resnet-wider-dl")
    #logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-50-transforms-2-2-resnet")
    print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=1,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
        # precision='16-mixed',
        accelerator = "cpu"
    )
    
    # trainer.fit(model=flow_obj, datamodule=sig_dat)
    flow_obj.embedding_net.fuse_model()
    flow_obj.embedding_net.to("cpu")
    
    # accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    # tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default"
    )

    q_model = quantization.fit(
        model=flow_obj.embedding_net,
        conf=conf,
        calib_dataloader=sig_dat.val_dataloader(),
    )


    
    # # # Define the quantizer
    # quantizer = Quantization("./config.yml")
    # quantizer.model = flow_obj.embedding_net
    
    # # # Calibration dataloader
    # dataloader = sig_dat.val_dataloader()
    # quantizer.calib_dataloader = dataloader
    
    # # # Custom calibration function
    # def calib_func(model):
    #     model.eval()
    #     i = 0
    #     with torch.no_grad():
    #         for inputs, _ in dataloader:
    #             inputs = inputs.to("cpu")  # Ensure inputs are on the same device as the model
    #             model(inputs)
    #             i+=1
    #             if i ==10:
    #                 break
    # quantizer.calib_func = calib_func  # Set the custom calibration function
    
    # # # Run the quantization process
    # q_model = quantizer()
    print(q_model)
    q_model.save('./output_PTDQ')

    flow_obj.embedding_net = q_model
    flow_obj.embedding_net.to('cpu')

    # int8_model = load('./output', flow_obj.embedding_net)
    # flow_obj.embedding_net = int8_model
    # flow_obj.embedding_net.to('cpu')

    
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=None)

if __name__ == '__main__':
    main()

