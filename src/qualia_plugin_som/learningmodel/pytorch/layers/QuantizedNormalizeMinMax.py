import torch

from .NormalizeMinMax import NormalizeMinMax

from qualia_core.learningmodel.pytorch.quantizer import maxWeight, quantifier

class QuantizedNormalizeMinMax(NormalizeMinMax):
    def __init__(self, bits=-1, force_q=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_activation = -float('inf')
        self.bits = bits
        self.max_activation2 = -float('inf')
        self.force_q = force_q

    def forward(self, input):
        if self.training:
            maxi = torch.max(torch.abs(input)).item()
            if self.max_activation < maxi:
                self.max_activation = maxi

        x = quantifier(input, self.bits, self.training, self.max_activation, force_q=self.force_q)

        # Make sure to update params when training
        if self.training:
            self.update_min_max(x)

        max_weight = torch.maximum(torch.abs(self.min), torch.abs(self.reciprocal_divisor))

        # Cannot call super().forward() here since we need to apply quantization when updating parameters
        y = super().normalize_min_max(
                        x,
                        quantifier(self.min, self.bits, self.training, max_weight, force_q=self.force_q),
                        quantifier(self.reciprocal_divisor, self.bits, self.training, max_weight, force_q=self.force_q)
                    )

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
            maxi = torch.maximum(torch.abs(self.min), torch.abs(self.reciprocal_divisor))
            return self.bits - 1 + maxWeight(torch.zeros(1), maxi, training=False)

