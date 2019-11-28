import copy
import math

import qualia_core.learningmodel.pytorch as learningmodels
import torch
import torch.nn as nn

from .DLSOM import DLSOM
from .layers import QuantizedNormalizeMinMax


class QuantizedDLSOM(DLSOM):
    def __init__(self, 
            input_shape: tuple,
            output_shape: tuple,

            bits: int=0,
            quantize_bias: bool=True,
            quantize_linear: bool=True,
            quantize_add: bool=True,
            quantize_dl: bool=True,
            quantize_som: bool=True,
            force_q: int=None,
            fused_relu: bool=True,
            *args, **kwargs):
        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         bits=bits,
                         quantize_bias=quantize_bias,
                         quantize_linear=quantize_linear,
                         quantize_add=quantize_add,
                         quantize_dl=quantize_dl,
                         quantize_som=quantize_som,
                         force_q=force_q,
                         fused_relu=fused_relu,
                         *args, **kwargs)

    def _build_model(self,
            input_shape: tuple,
            output_shape: tuple,
            iteration: int,
            dl: dict,
            som: dict,
            fm_output: str,
            bits: int=0,
            quantize_bias: bool=True,
            quantize_linear: bool=True,
            quantize_add: bool=True,
            quantize_dl: bool=True,
            quantize_som: bool=True,
            force_q: int=None,
            fused_relu: bool=True):
        from qualia_core.learningframework import PyTorch

        framework = PyTorch()

        self.quantize_som = quantize_som
        self.quantize_dl = quantize_dl

        # Complete deep learning model
        dl_model_params = copy.deepcopy(dl.get('params', {}))
        if 'input_shape' not in dl_model_params:
            dl_model_params['input_shape'] = input_shape
        if 'output_shape' not in dl_model_params:
            dl_model_params['output_shape'] = output_shape

        if quantize_dl:
            # Complete quantized deep learning model
            self.dl = getattr(learningmodels,  'Quantized' + dl['kind'])(
                                bits=bits,
                                quantize_bias=quantize_bias,
                                quantize_linear=quantize_linear,
                                quantize_add=quantize_add,
                                force_q=force_q,
                                fused_relu=fused_relu,
                                **dl_model_params)
        else:
            self.dl = getattr(learningmodels, dl['kind'])(**dl_model_params)

        if dl['load']:
            if 'iteration' in dl:
                dl_iteration = dl['iteration']
            else:
                dl_iteration = iteration
            print(f"Loading pre-trained DL model '{dl['name']}_r{dl_iteration}'")
            self.dl = framework.load(f'{dl["name"]}_r{dl_iteration}', self.dl)

        self.dl_epochs = dl['epochs']
        self.dl_batch_size = dl['batch_size']

        # Feature extractor model
        self.fm = self.create_feature_extractor(self.dl, fm_output)

        self.fm_shape = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

        # Flatten features
        self.flatten = nn.Flatten()

        # Quantized min-max normalization layer
        self.normalizeminmax = QuantizedNormalizeMinMax(bits=bits, force_q=force_q)

        # Quantized self-organizing map model

        if quantize_som:
            som['params']['som_layer'] = 'Quantized' + som['params']['som_layer']
            som['params']['bits'] = bits
            som['params']['force_q'] = force_q

        # Self-organizing map model
        self.som = learningmodels.SOM(
                            input_shape=(math.prod(self.fm_shape[1:]), ), # Flattened features
                            output_shape=output_shape,
                            **som.get('params', {}))

        self.som_epochs = som['epochs']
        self.som_batch_size = som['batch_size']

