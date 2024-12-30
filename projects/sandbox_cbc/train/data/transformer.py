import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Optional, Tuple

class TransformerEmbedding(nn.Module):
    """Transformer Embedding Layer for 1D sequences.

    Args:
        num_ifos: Number of interferometers used for BBH detection. Sets the channel dimension of the input tensor.
        context_dim: Dimension of the output context.
        d_model: Dimension of the embedding vector.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        dim_feedforward: Dimension of the feedforward network model.
        dropout: Dropout value.
        use_mixed_precision: Whether to use mixed precision for training.
    """
    def __init__(
        self,
        num_ifos: int,
        context_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_mixed_precision: bool = True,  # Enable mixed precision
    ):
        super(TransformerEmbedding, self).__init__()
        self.model_type = 'Transformer'
        self.use_mixed_precision = use_mixed_precision  # Mixed precision flag
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Linear(num_ifos, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, context_dim)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src)
        src = self.positional_encoder(src)
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            output = self.transformer_encoder(src)
        output = torch.mean(output, dim=1)  # Global average pooling
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    """Positional Encoding for the Transformer.

    Args:
        d_model: Dimension of the embedding vector.
        dropout: Dropout value.
        max_len: Maximum length of the input sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
