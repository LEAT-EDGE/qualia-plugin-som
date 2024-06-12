from __future__ import annotations

import logging
import math
import sys

import torch
from qualia_core.typing import TYPE_CHECKING
from torch import nn

from .SOM import SOM

if TYPE_CHECKING:
    from qualia_plugin_som.learningmodel.pytorch.layers.SOMLabelling import SOMLabelling  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class DSOM(SOM):
    def __init__(self,  # noqa: PLR0913
                 in_features: tuple[int, ...],
                 out_features: tuple[int, ...],
                 learning_rate: float = 0.01,
                 elasticity: float = 0.01,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.out_features = out_features
        self.in_features = in_features

        self.elasticity_squared = nn.Parameter(torch.tensor(elasticity, device=device, dtype=dtype).square(), requires_grad=False)
        self.learning_rate = nn.Parameter(torch.tensor(learning_rate, device=device, dtype=dtype), requires_grad=False)

        self.neurons = nn.Parameter(torch.empty((math.prod(out_features), math.prod(in_features)), device=device, dtype=dtype),
                                    requires_grad=False)

        with torch.no_grad():
            _ = torch.nn.init.uniform_(self.neurons, a=0.0, b=1.0)

    @override
    def extra_repr(self) -> str:
        return f'learning_rate={self.learning_rate.item()}, elasticity={self.elasticity_squared.sqrt().item()}'

    # From https://stackoverflow.com/a/65168284/1447751
    def unravel_index(self,
        indices: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Convert flat indices into unraveled coordinates in a target shape.

        This is a PyTorch implementation of :meth:`numpy.unravel_index`.

        :param indices: A tensor of (flat) indices, ``(*, N)``
        :param shape: The targeted shape, ``(D,)``
        :return: The unraveled coordinates, ``(*, N, D)``
        """
        coord: list[torch.Tensor] = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode='trunc')

        return torch.stack(coord[::-1], dim=-1)

    def dsom(self,  # noqa: PLR0913
             x_batch: torch.Tensor,
             neurons: torch.Tensor,
             learning_rate: torch.Tensor,
             elasticity_squared: torch.Tensor,
             return_position: bool = True,  # noqa: FBT001, FBT002
             return_value: bool = True,  # noqa: FBT001, FBT002
             training: bool = True) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # noqa: FBT002, FBT001
        with torch.no_grad():
            x_batch = x_batch.reshape((x_batch.shape[0], 1, -1)) # Flatten input dimensions

            # Preallocate BMU indices array for entire batch
            bmu = torch.empty((x_batch.shape[0]), device=x_batch.device, dtype=torch.long)
            # Preallocate BMU grid position for entire batch
            bmu_location = torch.empty((x_batch.shape[0], len(self.out_features)), device=x_batch.device, dtype=torch.long)

            # Handle batch sequentially
            # Still provides a performance boost with decent batch size (32-128) instead of feeding input one by one
            # Without having to change learning algorithm
            for i, x in enumerate(x_batch):
                input_neuron_differences = (x - neurons)

                input_neuron_differences_squared = input_neuron_differences.pow(2)

                neurons_distances_to_input_squared = input_neuron_differences_squared.sum(-1)

                bmu_distance_to_input_squared, bmu[i] = neurons_distances_to_input_squared.min(dim=-1)

                bmu_location[i] = self.unravel_index(bmu[i], self.out_features)

                if training:
                    grid_coordinates = torch.cartesian_prod(*[torch.arange(0, l, device=neurons.device, dtype=neurons.dtype)
                                                              for l in self.out_features])
                    differences_to_bmu_on_grid = (bmu_location[i] - grid_coordinates)
                    distances_to_bmu_on_grid = differences_to_bmu_on_grid.abs().sum(-1)

                    neighbourhood = torch.where(
                        bmu_distance_to_input_squared == bmu_distance_to_input_squared.new_tensor(0),
                        bmu_distance_to_input_squared.new_tensor(0),
                        torch.exp(-distances_to_bmu_on_grid.square() / (elasticity_squared * bmu_distance_to_input_squared)))

                    learning = (learning_rate
                                * (neurons_distances_to_input_squared.sqrt() * neighbourhood).unsqueeze(-1)
                                * input_neuron_differences)

                    neurons += learning

        if return_position and return_value:
            return bmu_location, neurons[bmu].reshape((-1, *self.in_features))
        if return_position:
            return bmu_location
        if return_value:
            return neurons[bmu].reshape((-1, *self.in_features))

        logger.error('One or both of return_position and return_value must be True')
        raise ValueError

    @override
    def forward(self,
                input: torch.Tensor,  # noqa: A002
                current_epoch: int | None = None, # We always receive current_epoch even though we do not use it in DSOM
                max_epochs: int | None = None, # We always receive max_epochs even though we do not use it in DSOM
                targets: torch.Tensor | None = None, # Unused for unsupervised learning
                som_labelling: SOMLabelling | None = None, # Unused for unsupervised learning,
                return_position: bool = True,
                return_value: bool = True) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.dsom(input,
                         self.neurons,
                         self.learning_rate,
                         self.elasticity_squared,
                         return_position=return_position,
                         return_value=return_value,
                         training=self.training)
