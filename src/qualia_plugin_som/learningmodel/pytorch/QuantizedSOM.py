import math

from .layers import som as som_layers
from .SOM import SOM


class QuantizedSOM(SOM):
    def __init__(self,
            input_shape: tuple,
            output_shape: tuple,
            som_layer: str,
            neurons: tuple,
            label_sigma: float,
            bits=-1,
            force_q=None,
            quantize_learning_rate_elasticity: bool=False,
            *args, **kwargs):
        super().__init__(input_shape=input_shape, output_shape=output_shape, som_layer=som_layer, neurons=neurons, label_sigma=label_sigma, *args, **kwargs)
    
        extra_args = {}
        if 'DSOM' in som_layer:
            extra_args['quantize_learning_rate_elasticity'] = quantize_learning_rate_elasticity

        self.som = getattr(som_layers, 'Quantized' + som_layer)(in_features=(math.prod(input_shape), ),
                                      out_features=tuple(int(n) for n in neurons),
                                      bits=bits,
                                      force_q=force_q,
                                      *args, **extra_args, **kwargs)
