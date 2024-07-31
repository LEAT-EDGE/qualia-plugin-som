from __future__ import annotations

import math

from qualia_core.typing import TYPE_CHECKING

from .LabelledSOM import LabelledSOM
from .layers import SOMLabelling
from .layers import som as som_layers

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH002

    from qualia_plugin_som.learningmodel.pytorch.layers.som.SOM import SOM  # noqa: TCH001
    from qualia_plugin_som.typing import SOMLayerConfigDict  # noqa: TCH001

class QuantizedLabelledSOM(LabelledSOM):
    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 som_layer: SOMLayerConfigDict,
                 neurons: list[int],
                 label_sigma: float) -> None:
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         som_layer=None, # We instantiate the SOM layer ourself
                         neurons=neurons,
                         label_sigma=label_sigma)

        som_layer_cls: type[SOM] = getattr(som_layers, 'Quantized' + som_layer['kind'])

        self.som = som_layer_cls(quant_params=quant_params,
                                 in_features=(math.prod(input_shape), ),
                                 out_features=tuple(int(n) for n in neurons),
                                 **som_layer['params'])


        self.som_labelling = SOMLabelling(out_features=output_shape, som=self.som, sigma=label_sigma)
