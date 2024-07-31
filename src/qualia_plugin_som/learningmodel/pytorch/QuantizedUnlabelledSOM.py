from __future__ import annotations

import math

from qualia_core.typing import TYPE_CHECKING

from .layers import som as som_layers
from .UnlabelledSOM import UnlabelledSOM

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH002

    from qualia_plugin_som.learningmodel.pytorch.layers.som.SOM import SOM  # noqa: TCH001
    from qualia_plugin_som.typing import SOMLayerConfigDict  # noqa: TCH001

class QuantizedUnlabelledSOM(UnlabelledSOM):
    def __init__(self,
                 quant_params: QuantizationConfig,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 som_layer: SOMLayerConfigDict,
                 neurons: list[int]) -> None:
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         som_layer=None, # We instantiate the SOM layer ourself
                         neurons=neurons)

        som_layer_cls: type[SOM] = getattr(som_layers, 'Quantized' + som_layer['kind'])

        self.som = som_layer_cls(quant_params=quant_params,
                                 in_features=(math.prod(input_shape), ),
                                 out_features=tuple(int(n) for n in neurons),
                                 **som_layer['params'])
