from __future__ import annotations

import copy
import logging
import math
import sys

import qualia_core.learningmodel.pytorch as learningmodels
import torch
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from qualia_plugin_som.learningmodel.pytorch.LabelledSOM import LabelledSOM
from qualia_plugin_som.learningmodel.pytorch.QuantizedLabelledSOM import QuantizedLabelledSOM

from .DLSOM import DLSOM
from .layers import QuantizedNormalizeMinMax

if TYPE_CHECKING:
    from qualia_core.learningmodel.pytorch.LearningModelPyTorch import LearningModelPyTorch  # noqa: TCH002
    from qualia_core.learningmodel.pytorch.Quantizer import QuantizationConfig  # noqa: TCH002
    from qualia_core.typing import ModelConfigDict, ModelParamsConfigDict

    from qualia_plugin_som.typing import SOMModelConfigDict  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class QuantizedDLSOM(DLSOM):
    def __init__(self,  # noqa: PLR0913
                 quant_params: QuantizationConfig,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 dl: ModelConfigDict,
                 som: ModelConfigDict,
                 fm_output: str,
                 iteration: int=1,
                 quantize_dl: bool=True,  # noqa: FBT001, FBT002
                 quantize_som: bool=True) -> None:  # noqa: FBT001, FBT002
        self.__quantize_dl = quantize_dl
        self.__quantize_som = quantize_som
        self.__quant_params = quant_params
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         dl=dl,
                         som=som,
                         fm_output=fm_output,
                         iteration=iteration)

    @override
    def _build_model(self,
                     input_shape: tuple[int, ...],
                     output_shape: tuple[int, ...],
                     iteration: int,
                     dl: ModelConfigDict,
                     som: SOMModelConfigDict,
                     fm_output: str) -> None:
        from qualia_core.learningframework import PyTorch

        framework = PyTorch()

        # Complete deep learning model
        dl_model_params: ModelParamsConfigDict = copy.deepcopy(dl.get('params', {}))
        if 'input_shape' not in dl_model_params:
            dl_model_params['input_shape'] = input_shape
        if 'output_shape' not in dl_model_params:
            dl_model_params['output_shape'] = output_shape

        if self.__quantize_dl:
            # Complete quantized deep learning model
            dl_class: type[LearningModelPyTorch] = getattr(learningmodels,  'Quantized' + dl['kind'])
            dl_model_params['quant_params'] = self.__quant_params
            self.dl = dl_class(**dl_model_params)
        else:
            dl_class: type[LearningModelPyTorch] = getattr(learningmodels,  dl['kind'])
            self.dl = dl_class(**dl_model_params)

        if dl.get('load', False):
            dl_iteration = dl.get('iteration', iteration)
            logger.info("Loading pre-trained DL model '%s_r%s'", dl['name'], dl_iteration)
            self.dl = framework.load(f'{dl["name"]}_r{dl_iteration}', self.dl)

        self.dl_epochs = dl.get('epochs', 0)
        self.dl_batch_size = dl.get('batch_size', 1)

        # Feature extractor model
        self.fm = self.create_feature_extractor(self.dl, fm_output)

        self.fm_shape: tuple[int, ...] = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

        # Flatten features
        self.flatten = nn.Flatten()

        # Quantized min-max normalization layer
        self.normalizeminmax = QuantizedNormalizeMinMax(quant_params=self.__quant_params)

        # Quantized self-organizing map model
        som_model_params = som.get('params', {})
        som_model_params['input_shape'] = (math.prod(self.fm_shape[1:]), ) # Flattened features
        som_model_params['output_shape'] = output_shape
        if self.__quantize_som:
            som_model_params['quant_params'] = self.__quant_params
            self.som = QuantizedLabelledSOM(**som_model_params)
        else:
            self.som = LabelledSOM(**som_model_params)

        self.som_epochs = som.get('epochs', 0)
        self.som_batch_size = som.get('batch_size', 1)
