from .KSOM import KSOM

import torch

from qualia_core.learningmodel.pytorch.quantizer import maxWeight, quantifier

class QuantizedKSOM(KSOM):
    def __init__(self, bits=-1, force_q=None, quantize_learning_rate_neighbourhood_width: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_activation = -float('inf')
        self.bits=bits
        self.max_activation2 = -float('inf')
        self.force_q = force_q
        if quantize_learning_rate_neighbourhood_width:
            raise NotImplemented()
        self.disable_som_training = False

    def forward(self, input, current_epoch: int=None, max_epochs: int=None, return_position: bool=True, return_value: bool=True, *args, **kwargs):
        max_weight = torch.max(torch.abs(self.neurons))

        if self.training:
            maxi = torch.max(torch.abs(input)).item()
            if self.max_activation < maxi:
                self.max_activation = maxi
        y = super().ksom(
                quantifier(input, self.bits, self.training, self.max_activation, force_q=self.force_q),
                quantifier(self.neurons, self.bits, self.training, max_weight, force_q=self.force_q),
                self.learning_rate,
                self.neighbourhood_width,
                current_epoch=current_epoch,
                max_epochs=max_epochs,
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
            maxi = torch.max(torch.abs(self.neurons))
            return self.bits - 1 + maxWeight(torch.zeros(1), maxi, training=False)

