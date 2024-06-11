from __future__ import annotations

import sys

import torch
from torch import nn

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class NormalizeMinMax(nn.Module):
    def __init__(self,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()
        self.min = nn.Parameter(torch.tensor(float('+inf'), device=device, dtype=dtype), requires_grad=False)
        self.max = nn.Parameter(torch.tensor(float('-inf'), device=device, dtype=dtype), requires_grad=False)

    def update_min_max(self, x: torch.Tensor) -> None:
        _ = torch.minimum(self.min, x.min(), out=self.min)
        _ = torch.maximum(self.max, x.max(), out=self.max)

    def normalize_min_max(self, x: torch.Tensor, min_val: torch.Tensor, reciprocal_divisor: torch.Tensor) -> torch.Tensor:
        x -= min_val
        x *= reciprocal_divisor
        return x

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.training:
            self.update_min_max(input)
        return self.normalize_min_max(input, self.min, self.reciprocal_divisor)

    @property
    def reciprocal_divisor(self) -> torch.Tensor:
        return torch.reciprocal(self.max - self.min)
