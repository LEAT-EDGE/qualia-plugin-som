from __future__ import annotations

import copy
import logging
import math
import sys

import torch
from qualia_core.learningmodel.pytorch.layers import layers as custom_layers
from qualia_core.typing import TYPE_CHECKING
from torch import nn
from torch.fx import GraphModule, Tracer

from qualia_plugin_som.learningmodel.pytorch.LabelledSOM import LabelledSOM

if TYPE_CHECKING:
    from qualia_core.typing import ModelConfigDict, ModelParamsConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DLSOM(nn.Module):
    # Custom tracer that generates call_module for our custom Qualia layers instead of attempting to trace their forward()
    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple[type[nn.Module], ...]) -> None:
            super().__init__()
            self.custom_layers = custom_layers

        @override
        def is_leaf_module(self, m: nn.Module, module_qualified_name : str) -> bool:
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, custom_layers)

    def __init__(self,  # noqa: PLR0913
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 dl: ModelConfigDict,
                 som: ModelConfigDict,
                 fm_output: str,
                 iteration: int=1) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)

        self._build_model(self.input_shape,
                          self.output_shape,
                          dl=dl,
                          som=som,
                          fm_output=fm_output,
                          iteration=iteration)

    def _build_model(self,  # noqa: PLR0913
                     input_shape: tuple[int, ...],
                     output_shape: tuple[int, ...],
                     iteration: int,
                     dl: ModelConfigDict,
                     som: ModelConfigDict,
                     fm_output: str) -> None:
        import qualia_core.learningmodel.pytorch as learningmodels
        from qualia_core.learningframework import PyTorch

        from .layers import NormalizeMinMax

        framework = PyTorch()

        # Complete deep learning model
        dl_model_params: ModelParamsConfigDict = copy.deepcopy(dl.get('params', {}))
        if 'input_shape' not in dl_model_params:
            dl_model_params['input_shape'] = input_shape
        if 'output_shape' not in dl_model_params:
            dl_model_params['output_shape'] = output_shape
        self.dl = getattr(learningmodels, dl['kind'])(**dl_model_params)

        if dl.get('fuse_batch_norm', False):
            from qualia_core.postprocessing import FuseBatchNorm
            self.dl.eval()
            fused_dl = FuseBatchNorm().fuse(self.dl, graphmodule_cls=GraphModule, inplace=True)
            fused_dl.input_shape = self.dl.input_shape
            fused_dl.output_shape = self.dl.output_shape
            self.dl = fused_dl
        if dl.get('load', False):
            dl_iteration = dl.get('iteration', iteration)
            logger.info("Loading pre-trained DL model '%s_r%s'", dl['name'], dl_iteration)
            self.dl = framework.load(f'{dl["name"]}_r{dl_iteration}', self.dl)

        self.dl_epochs = dl.get('epochs', 0)
        self.dl_batch_size = dl.get('batch_size', 1)

        # Feature extractor model
        self.fm = self.create_feature_extractor(self.dl, fm_output)

        self.fm_shape = self.fm(torch.rand((1, *self._shape_channels_last_to_first(input_shape)))).shape

        # Flatten features
        self.flatten = nn.Flatten()

        # Min-max normalization layer
        self.normalizeminmax = NormalizeMinMax()

        # Self-organizing map model
        self.som = LabelledSOM(
                input_shape=(math.prod(self.fm_shape[1:]), ), # Flattened features
                output_shape=output_shape,
                **som.get('params', {}))
        self.som_epochs = som['epochs']
        self.som_batch_size = som['batch_size']

    def _shape_channels_last_to_first(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (shape[-1], ) + shape[0:-1]

    # Similar to torchvision's but simplified for our specific use case
    def create_feature_extractor(self, model: nn.Module, return_node: str) -> nn.Module:
        # Feature extractor only used in eval mode
        _ = model.eval()

        tracer = DLSOM.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()
        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        # Remove existing output node
        old_output = [n for n in graphmodule.graph.nodes if n.op == 'output']
        if not old_output:
            logger.error('No output in DL model')
            raise ValueError
        if len(old_output) > 1:
            logger.error('Multiple outputs in DL model')
            raise ValueError
        graphmodule.graph.erase_node(old_output[0])

        # Find desired output layer
        new_output = [n for n in graphmodule.graph.nodes if n.name == return_node]
        if not new_output:
            logger.error("fm_output='%s' not found in DL model", return_node)
            raise ValueError
        if len(new_output) > 1:
            logger.error("Multiple matches for fm_output='%s' in DL model", return_node)
            raise ValueError

        # Add new output for desired layer
        with graphmodule.graph.inserting_after(list(graphmodule.graph.nodes)[-1]):
            graphmodule.graph.output(new_output[0])

        # Remove unused layers
        _ = graphmodule.graph.eliminate_dead_code()

        _ = graphmodule.recompile()

        return graphmodule

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        x = self.fm(input)
        x = self.flatten(x)
        x = self.normalizeminmax(x)
        return self.som(x)
