from __future__ import annotations

import math
import sys

import torch
from qualia_core.typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from qualia_plugin_som.typing import SOMLayerConfigDict  # noqa: TCH001

from .layers import SOMLabelling
from .layers import som as som_layers

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SOM(nn.Module):
    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int,...],
                 output_shape: tuple[int, ...],
                 som_layer: SOMLayerConfigDict | None,
                 neurons: list[int],
                 label_sigma: float) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        if len(input_shape) > 1:
            self.flatten = nn.Flatten()

        if som_layer is not None:
            som_layer_cls: type[nn.Module] = getattr(som_layers, som_layer['kind'])

            self.som = som_layer_cls(in_features=(math.prod(input_shape), ),
                                     out_features=tuple(int(n) for n in neurons),
                                     **som_layer['params'])

            self.som_labelling = SOMLabelling(out_features=output_shape, som=self.som, sigma=label_sigma)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) > 1:
            x = self.flatten(x)
        x = self.som(x, return_position=True, return_value=False)
        return self.som_labelling(x)
