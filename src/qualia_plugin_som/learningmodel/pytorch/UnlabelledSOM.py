from __future__ import annotations

import math
import sys

import torch
from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch
from qualia_core.typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from qualia_plugin_som.learningmodel.pytorch.layers.som.SOM import SOM  # noqa: TCH001
    from qualia_plugin_som.typing import SOMLayerConfigDict  # noqa: TCH001

from .layers import som as som_layers

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class UnlabelledSOM(LearningModelPyTorch):
    def __init__(self,
                 input_shape: tuple[int,...],
                 output_shape: tuple[int, ...],
                 som_layer: SOMLayerConfigDict | None,
                 neurons: list[int]) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        if len(input_shape) > 1:
            self.flatten = nn.Flatten()

        if som_layer is not None:
            som_layer_cls: type[SOM] = getattr(som_layers, som_layer['kind'])

            self.som = som_layer_cls(in_features=(math.prod(input_shape), ),
                                     out_features=tuple(int(n) for n in neurons),
                                     **som_layer['params'])

    @override
    def forward(self,
                x: torch.Tensor,
                current_epoch: int | None = None,
                max_epochs: int | None = None) -> torch.Tensor:
        if len(self.input_shape) > 1:
            x = self.flatten(x)
        return self.som(x, current_epoch=current_epoch, max_epochs=max_epochs, return_position=True, return_value=True)
