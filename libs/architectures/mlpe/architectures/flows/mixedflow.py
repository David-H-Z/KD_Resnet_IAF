import torch
import torch.nn as nn
import torch.distributions as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch import pl
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineAutoregressive
from pyro.nn import ConditionalAutoRegressiveNN
from mlpe.architectures.flows.flow import NormalizingFlow
from ml4gw.nn.resnet import ResNet1D  # Assuming ResNet1D is implemented or available from a library

class AttentionFlowModel(pl.LightningModule, NormalizingFlow):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        resnet_layers: List[int],
        resnet_context_dim: int,
        hidden_features: int,
        num_blocks: int,
        num_transforms: int,
        inference_params: list,
        num_samples_draw: int = 3000,
        num_plot_corner: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.param_dim, self.n_ifos, self.strain_dim = shape
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.inference_params = inference_params
        self.num_samples_draw = num_samples_draw
        self.num_plot_corner = num_plot_corner
        self.lr = lr
        self.weight_decay = weight_decay

        # ResNet for feature extraction
        self.embedding_net = ResNet1D(
            n_ifos=self.n_ifos,
            classes=resnet_context_dim,
            layers=resnet_layers
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=resnet_context_dim, num_heads=4)

        # Build the flow with alternating IAF, MAF, and INN transforms
        self.build_flow()

    def transform_block(self, use_iaf=True):
        arn = ConditionalAutoRegressiveNN(
            self.param_dim,
            self.context_dim,
            self.num_blocks * [self.hidden_features],
            nonlinearity=torch.tanh,
        )
        if use_iaf:
            return ConditionalAffineAutoregressive(arn).inv  # IAF
        else:
            return ConditionalAffineAutoregressive(arn)  # MAF

    def build_flow(self):
        """Build the transform with alternating IAF, MAF, and INN"""
        self.transforms = []
        for idx in range(self.num_transforms):
            if idx % 3 == 0:
                _transform = self.transform_block(use_iaf=True)  # IAF
            elif idx % 3 == 1:
                _transform = self.transform_block(use_iaf=False)  # MAF
            else:
                _transform = self.transform_block(use_iaf=False)  # Placeholder for INN
            self.transforms.extend([_transform])
        self.transforms = ConditionalComposeTransformModule(self.transforms)

    def forward(self, x, context):
        embedded_context = self.embedding_net(context)
        attended_context, _ = self.attention(embedded_context, embedded_context, embedded_context)
        return self.flow(x, attended_context)

    @property
    def context_dim(self):
        dummy_tensor = torch.zeros(
            (1, self.n_ifos, self.strain_dim), device=self.device
        )
        _context_dim = self.embedding_net(dummy_tensor).shape[-1]
        return _context_dim

    def log_prob(self, x, context):
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")
        embedded_context = self.embedding_net(context)
        attended_context, _ = self.attention(embedded_context, embedded_context, embedded_context)
        return self.flow.condition(attended_context).log_prob(x)

    def sample(self, n, context):
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")
        embedded_context = self.embedding_net(context)
        attended_context, _ = self.attention(embedded_context, embedded_context, embedded_context)
        n = [n] if isinstance(n, int) else n
        return self.flow.condition(attended_context).sample(n)

    def training_step(self, batch, batch_idx):
        strain, parameters = batch
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        strain, parameters = batch
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"}}

# Example usage
if __name__ == '__main__':
    shape = (8, 2, 2048 * 4)  # Example shape: param_dim, n_ifos, strain_dim
    resnet_layers = [2, 2, 2]
    resnet_context_dim = 128
    hidden_features = 64
    num_blocks = 3
    num_transforms = 6
    inference_params = ['chirp_mass', 'mass_ratio', 'luminosity_distance']

    model = AttentionFlowModel(
        shape=shape,
        resnet_layers=resnet_layers,
        resnet_context_dim=resnet_context_dim,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        num_transforms=num_transforms,
        inference_params=inference_params,
        lr=1e-3,
        weight_decay=0.001,
    )

    # Define your dataset and dataloader here
    # trainer = pl.Trainer(max_epochs=1000)
    # trainer.fit(model, train_dataloader, val_dataloader)
