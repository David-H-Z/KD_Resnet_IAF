from typing import Optional

import torch

from ml4gw.transforms.transform import FittableTransform


class ChannelWiseScaler(FittableTransform):
    """Scale timeseries channels to be zero mean unit variance

    Scales timeseries channels by the mean and standard
    deviation of the channels of the timeseries used to
    fit the module. To reverse the scaling, provide the
    `reverse=True` keyword argument at call time.
    By default, the scaling parameters are set to zero mean
    and unit variance, amounting to an identity transform.

    Args:
        num_channels:
            The number of channels of the target timeseries.
            If left as `None`, the timeseries will be assumed
            to be 1D (single channel).
    """

    def __init__(self, num_channels: Optional[int] = None) -> None:
        super().__init__()

        shape = (num_channels or 1,)
        if num_channels is not None:
            shape += (1,)

        mean = torch.zeros(shape)
        std = torch.ones(shape)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def fit(self, X: torch.Tensor) -> None:
        """Fit the scaling parameters to a timeseries

        Computes the channel-wise mean and standard deviation
        of the timeseries `X` and sets these values to the
        `mean` and `std` parameters of the scaler.
        """

        if X.ndim == 1:
            assert self.mean.ndim == self.std.ndim == 1
            mean = X.mean(dim=0, keepdim=True)
            # default for torch is to include bessel correction
            std = X.std(dim=0, correction=0, keepdim=True)
        elif X.ndim == 2:
            assert self.mean.ndim == self.std.ndim == 2
            assert len(X) == self.mean.size(0) == self.std.size(0)
            mean = X.mean(dim=-1, keepdim=True)
            std = X.std(dim=-1, correction=0, keepdim=True)
        else:
            raise ValueError(
                "Can't fit channel wise mean and standard deviation "
                "from tensor of shape {}".format(X.shape)
            )

        super().build(mean=mean, std=std)

    def forward(self, X: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            return (X - self.mean) / self.std
        else:
            return self.std * X.to("cuda") + self.mean
