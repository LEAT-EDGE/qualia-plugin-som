from __future__ import annotations

import logging
import sys

import torch
from qualia_core.learningmodel.pytorch.layers.QuantizedLayer import (
    QuantizedLayer,
    QuantizerActProtocol,
    QuantizerInputProtocol,
    QuantizerWProtocol,
)
from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig, Quantizer, update_params

from .NormalizeMinMax import NormalizeMinMax

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedNormalizeMinMax(NormalizeMinMax, QuantizerInputProtocol, QuantizerActProtocol, QuantizerWProtocol, QuantizedLayer):
    def __init__(self,
                 quant_params: QuantizationConfig,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__(device=device, dtype=dtype)

        # Create the quantizer instance
        quant_params_input = update_params(tensor_type='input', quant_params=quant_params)
        quant_params_act = update_params(tensor_type='act', quant_params=quant_params)
        quant_params_w = update_params(tensor_type='w', quant_params=quant_params)
        self.quantizer_input = Quantizer(**quant_params_input)
        self.quantizer_act = Quantizer(**quant_params_act)
        self.quantizer_w = Quantizer(**quant_params_w)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        q_x = self.quantizer_input(input)

        # Make sure to update params when training
        if self.training:
            self.update_min_max(q_x)

        q_hyperparams = self.quantizer_w(self.get_hyperparams_tensor(device=self.min.device, dtype=self.min.dtype))
        q_min = q_hyperparams[0]
        q_reciprocal_divisor = q_hyperparams[1]


        # Cannot call super().forward() here since we need to apply quantization when updating parameters
        y = super().normalize_min_max(q_x,
                                      q_min,
                                      q_reciprocal_divisor)

        return self.quantizer_w(y)

    def get_hyperparams_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pack min and reciprocal_divisor into the same Tensor.

        :param device: Device to create the tensor on
        :param dtype: Data type for the created tensor
        :return: New tensor with hyperparemeters concatenated
        """
        return torch.tensor([self.min, self.reciprocal_divisor], device=device, dtype=dtype)
