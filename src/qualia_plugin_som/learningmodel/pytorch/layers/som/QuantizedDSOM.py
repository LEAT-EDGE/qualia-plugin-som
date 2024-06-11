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

from .DSOM import DSOM

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedDSOM(DSOM, QuantizerInputProtocol, QuantizerActProtocol, QuantizerWProtocol, QuantizedLayer):
    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 in_features: tuple[int, ...],
                 out_features: tuple[int, ...],
                 learning_rate: float = 0.01,
                 elasticity: float = 0.01,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 quantize_learning_rate_elasticity: bool = False) -> None:  # noqa: FBT001, FBT002
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         learning_rate=learning_rate,
                         elasticity=elasticity,
                         device=device,
                         dtype=dtype)
        self.quantize_learning_rate_elasticity = quantize_learning_rate_elasticity
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
                current_epoch: int | None = None, # We always receive current_epoch even though we do not use it in DSOM
                max_epochs: int | None = None, # We always receive max_epochs even though we do not use it in DSOM
                return_position: bool = True,
                return_value: bool = True) -> torch.Tensor:
        q_x = self.quantizer_input(input)
        if self.quantize_learning_rate_elasticity:
            q_w, q_hyperparams = self.quantizer_w(self.neurons,
                                   bias_tensor=self.get_hyperparams_tensor(device=self.neurons.device, dtype=self.neurons.dtype))
            q_learning_rate = q_hyperparams[0]
            q_elasticity_squared = q_hyperparams[1]
        else:
            q_w = self.quantizer_w(self.neurons)


        y = super().dsom(q_x,
                         q_w,
                         q_learning_rate if self.quantize_learning_rate_elasticity else self.learning_rate,
                         q_elasticity_squared if self.quantize_learning_rate_elasticity else self.elasticity_squared,
                         return_position=return_position,
                         return_value=return_value,
                         training=self.training and not self.disable_som_training)

        if return_position and return_value:
            return y[0], self.quantizer_act(y[1])
        if return_position:
            return y
        if return_value:
            return self.quantizer_act(y)

        logger.error('One or both of return_position and return_value must be True')
        raise ValueError

    def get_hyperparams_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pack learning_rate and elasticity_squared into the same Tensor.

        :param device: Device to create the tensor on
        :param dtype: Data type for the created tensor
        :return: New tensor with hyperparemeters concatenated
        """
        return torch.tensor([self.learning_rate, self.elasticity_squared], device=device, dtype=dtype)
