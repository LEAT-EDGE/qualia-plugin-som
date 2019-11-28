import copy
import math

import torch
import torch.nn as nn
from torch.fx import Tracer, GraphModule

from qualia_core.learningmodel.pytorch.layers import layers as custom_layers

class DLSOM(nn.Module):
    # Custom tracer that generates call_module for our custom Qualia layers instead of attempting to trace their forward()
    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_layers = custom_layers

        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, custom_layers)

    def __init__(self, input_shape: tuple, output_shape: tuple, iteration: int=1, *args, **kwargs):

        super().__init__()

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)

        self._build_model(self.input_shape, self.output_shape, iteration=iteration, *args, **kwargs)

    def _build_model(self, input_shape: tuple, output_shape: tuple, iteration: int, dl: dict, som: dict, fm_output: str, *args, **kwargs):
        import qualia_core.learningmodel.pytorch as learningmodels
        from .layers import NormalizeMinMax
        from qualia_core.learningframework import PyTorch

        framework = PyTorch()

        # Complete deep learning model
        dl_model_params = copy.deepcopy(dl.get('params', {}))
        if 'input_shape' not in dl_model_params:
            dl_model_params['input_shape'] = input_shape
        if 'output_shape' not in dl_model_params:
            dl_model_params['output_shape'] = output_shape
        self.dl = getattr(learningmodels, dl['kind'])(**dl_model_params)

        if dl.get('fuse_batch_norm', False):
            from qualia_core.postprocessing import FuseBatchNorm
            self.dl.eval()
            fused_dl = FuseBatchNorm().fuse(self.dl, inplace=True)
            fused_dl.input_shape = self.dl.input_shape
            fused_dl.output_shape = self.dl.output_shape
            self.dl = fused_dl
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

        # Min-max normalization layer
        self.normalizeminmax = NormalizeMinMax()

        # Self-organizing map model
        self.som = learningmodels.SOM(
                            input_shape=(math.prod(self.fm_shape[1:]), ), # Flattened features
                            output_shape=output_shape,
                            **som.get('params', {}))
        self.som_epochs = som['epochs']
        self.som_batch_size = som['batch_size']

    def _shape_channels_last_to_first(self, shape):
        return (shape[-1], ) + shape[0:-1]

    # Similar to torchvision's but simplified for our specific use case
    def create_feature_extractor(self, model: nn.Module, return_node: str):
        # Feature extractor only used in eval mode
        model.eval()

        tracer = DLSOM.TracerCustomLayers(custom_layers=custom_layers)
        graph = tracer.trace(model)
        graph.print_tabular()
        graphmodule = GraphModule(tracer.root, graph, tracer.root.__class__.__name__)

        # Remove existing output node
        old_output = [n for n in graphmodule.graph.nodes if n.op == 'output']
        if not old_output:
            raise ValueError(f'No output in dl model')
        if len(old_output) > 1:
            raise ValueError(f'Multiple outputs in dl model')
        graphmodule.graph.erase_node(old_output[0])

        # Find desired output layer
        new_output = [n for n in graphmodule.graph.nodes if n.name == return_node]
        if not new_output:
            raise ValueError(f'fm_output = \'{return_node}\' not found in dl model')
        if len(new_output) > 1:
            raise ValueError(f'Multiple matches for fm_output = \'{return_node}\' in dl model')

        # Add new output for desired layer
        with graphmodule.graph.inserting_after(list(graphmodule.graph.nodes)[-1]):
            graphmodule.graph.output(new_output[0])

        # Remove unused layers
        graphmodule.graph.eliminate_dead_code()

        graphmodule.recompile()

        return graphmodule

    def forward(self, x):
        x = self.fm(x)
        x = self.flatten(x)
        x = self.normalizeminmax(x)
        x = self.som(x)
        return x
