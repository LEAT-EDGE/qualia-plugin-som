from __future__ import annotations

import logging
import sys

from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import (
    QuantizedLayer,
    QuantizerActProtocol,
    QuantizerInputProtocol,
    QuantizerWProtocol,
)
from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig, Quantizer, update_params
from qualia_core.typing import TYPE_CHECKING

from .KSOM import KSOM

if TYPE_CHECKING:
    import torch  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedKSOM(KSOM, QuantizerInputProtocol, QuantizerActProtocol, QuantizerWProtocol, QuantizedLayer):
    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 in_features: tuple[int, ...],
                 out_features: tuple[int, ...],
                 learning_rate: list[float],
                 neighbourhood_width: list[float],
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 quantize_learning_rate_neighbourhood_width: bool=False) -> None:  # noqa: FBT001, FBT002
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         learning_rate=learning_rate,
                         neighbourhood_width=neighbourhood_width,
                         device=device,
                         dtype=dtype)
        if quantize_learning_rate_neighbourhood_width:
            logger.error('Quantization not implemented for KSOM learning rate and neighbourhood width')
            raise NotImplementedError
        self.disable_som_training = False

        # Create the quantizer instance
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_w = update_params(tensor_type='w', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)
        self.quantizer_w = Quantizer(**quant_params_w)

    @override
    def forward(self,
                input: torch.Tensor,  # noqa: A002
                current_epoch: int | None = None,
                max_epochs: int | None = None,
                return_position: bool = True,
                return_value: bool = True) -> torch.Tensor:

        q_x = self.quantizer_input(input)
        q_w = self.quantizer_w(self.neurons)


        y = super().ksom(q_x,
                         q_w,
                         self.learning_rate,
                         self.neighbourhood_width,
                         current_epoch=current_epoch,
                         max_epochs=max_epochs,
                         return_position=return_position, return_value=return_value,
                         training=self.training and not self.disable_som_training)

        if return_position and return_value:
            return y[0], self.quantizer_act(y[1])
        if return_position:
            return y
        if return_value:
            return self.quantizer_act(y)

        logger.error('One or both of return_position and return_value must be True')
        raise ValueError
