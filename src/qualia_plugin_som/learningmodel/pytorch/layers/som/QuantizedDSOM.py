from .DSOM import DSOM

import torch

from qualia_core.learningmodel.pytorch.quantizer import maxWeight, quantifier

class QuantizedDSOM(DSOM):
    def __init__(self, bits=-1, force_q=None, quantize_learning_rate_elasticity: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_activation = -float('inf')
        self.bits=bits
        self.max_activation2 = -float('inf')
        self.force_q = force_q
        self.quantize_learning_rate_elasticity = quantize_learning_rate_elasticity
        self.disable_som_training = False

    def forward(self, input, return_position: bool=True, return_value: bool=True, *args, **kwargs):
        if self.quantize_learning_rate_elasticity:
            max_weight = torch.maximum(torch.max(torch.abs(self.neurons)), torch.maximum(self.learning_rate, self.elasticity_squared))
        else:
            max_weight = torch.max(torch.abs(self.neurons))

        if self.training:
            maxi = torch.max(torch.abs(input)).item()
            if self.max_activation < maxi:
                self.max_activation = maxi
        y = super().dsom(
                quantifier(input, self.bits, self.training, self.max_activation, force_q=self.force_q),
                quantifier(self.neurons, self.bits, self.training, max_weight, force_q=self.force_q),
                quantifier(self.learning_rate, self.bits, self.training, max_weight, force_q=self.force_q)
                    if self.quantize_learning_rate_elasticity else self.learning_rate,
                quantifier(self.elasticity_squared, self.bits, self.training, max_weight, force_q=self.force_q)
                    if self.quantize_learning_rate_elasticity else self.elasticity_squared,
                return_position=return_position, return_value=return_value,
                training=self.training and not self.disable_som_training)

        if return_position and return_value:
            if self.training:
                maxi = torch.max(torch.abs(y[1])).item()
                if self.max_activation2 < maxi:
                    self.max_activation2 = maxi
            return y[0], quantifier(y[1], self.bits, self.training, self.max_activation2, force_q=self.force_q)
        elif return_position:
            return y
        elif return_value:
            if self.training:
                maxi = torch.max(torch.abs(y)).item()
                if self.max_activation2 < maxi:
                    self.max_activation2 = maxi
            return quantifier(y, self.bits, self.training, self.max_activation2, force_q=self.force_q)

    @property
    def input_q(self):
        if self.force_q is not None:
            return self.force_q
        else:
            return self.bits - 1 + maxWeight(torch.zeros(1), self.max_activation, training=False)

    @property
    def activation_q(self):
        if self.force_q is not None:
            return self.force_q
        else:
            return self.bits - 1 + maxWeight(torch.zeros(1), self.max_activation2, training=False)

    @property
    def weights_q(self):
        if self.force_q is not None:
            return self.force_q
        else:
            if self.quantize_learning_rate_elasticity:
                maxi = torch.maximum(torch.max(torch.abs(self.neurons)), torch.maximum(self.learning_rate, self.elasticity_squared))
            else:
                maxi = torch.max(torch.abs(self.neurons))
            return self.bits - 1 + maxWeight(torch.zeros(1), maxi, training=False)

