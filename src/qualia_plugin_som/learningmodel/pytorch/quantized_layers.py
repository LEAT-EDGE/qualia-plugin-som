from .layers.QuantizedNormalizeMinMax import QuantizedNormalizeMinMax
from .layers.som import QuantizedDSOM

quantized_layers = (QuantizedDSOM, QuantizedNormalizeMinMax)
