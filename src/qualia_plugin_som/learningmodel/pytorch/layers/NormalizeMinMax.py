import torch
import torch.nn as nn

class NormalizeMinMax(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.min = nn.Parameter(torch.tensor(float('+inf'), device=device, dtype=dtype), requires_grad=False)
        self.max = nn.Parameter(torch.tensor(float('-inf'), device=device, dtype=dtype), requires_grad=False)

    def update_min_max(self, x):
        torch.minimum(self.min, x.min(), out=self.min)
        torch.maximum(self.max, x.max(), out=self.max)

    def normalize_min_max(self, x, min, reciprocal_divisor):
        x -= min
        x *= reciprocal_divisor
        return x

    def forward(self, x):
        if self.training:
            self.update_min_max(x)
        return self.normalize_min_max(x, self.min, self.reciprocal_divisor)

    @property
    def reciprocal_divisor(self):
        return torch.reciprocal(self.max - self.min)
