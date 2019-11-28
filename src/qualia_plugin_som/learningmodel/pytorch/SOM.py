import math

from torch import nn

from .layers import SOMLabelling
from .layers import som as som_layers


class SOM(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, som_layer: str, neurons: tuple, label_sigma: float, *args, **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
    
        if len(input_shape) > 1:
            self.flatten = nn.Flatten()
        self.som = getattr(som_layers, som_layer)(in_features=(math.prod(input_shape), ), out_features=tuple(int(n) for n in neurons), *args, **kwargs)
        self.som_labelling = SOMLabelling(out_features=output_shape, som=self.som, sigma=label_sigma)

    def forward(self, x):
        if len(self.input_shape) > 1:
            x = self.flatten(x)
        x = self.som(x, return_position=True, return_value=False)
        x = self.som_labelling(x)
        return x
