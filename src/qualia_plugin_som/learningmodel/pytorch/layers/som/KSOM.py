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

class KSOM(SOM):
    def __init__(self,  # noqa: PLR0913
                 in_features: tuple[int, ...],
                 out_features: tuple[int, ...],
                 learning_rate: list[float],
                 neighbourhood_width: list[float],
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

        # [learning_rate_i, learning_rate_f]
        self.learning_rate = nn.Parameter(torch.tensor(learning_rate, device=device, dtype=dtype), requires_grad=False)
        # [neighbourhood_i, neighbourhood_f]
        self.neighbourhood_width = nn.Parameter(torch.tensor(neighbourhood_width, device=device, dtype=dtype), requires_grad=False)

        self.neurons = nn.Parameter(torch.empty((math.prod(out_features), math.prod(in_features)), device=device, dtype=dtype),
                                    requires_grad=False)

        with torch.no_grad():
            _ = torch.nn.init.uniform_(self.neurons, a=0.0, b=1.0)

    # From https://stackoverflow.com/a/65168284/1447751
    def unravel_index(self,
        indices: torch.LongTensor,
        shape: tuple[int, ...],
    ) -> torch.LongTensor:
        r"""Converts flat indices into unraveled coordinates in a target shape.

        This is a `torch` implementation of `numpy.unravel_index`.

        Args:
            indices: A tensor of (flat) indices, (*, N).
            shape: The targeted shape, (D,).

        Returns:
            The unraveled coordinates, (*, N, D).
        """

        coord = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode='trunc')

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    def ksom(self,  # noqa: PLR0913
             x_batch: torch.Tensor,
             neurons: torch.Tensor,
             learning_rate: list[float],
             neighbourhood_width: list[float],
             current_epoch: int,
             max_epochs: int,
             return_position: bool = True,  # noqa: FBT001, FBT002
             return_value: bool = True,  # noqa: FBT001, FBT002
             training: bool = True) -> torch.Tensor:  # noqa: FBT001, FBT002
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

                bmu_distance_to_input_squared, bmu[i] = neurons_distances_to_input_squared.min(axis=-1)

                bmu_location[i] = self.unravel_index(bmu[i], self.out_features)

                if training:
                    current_learning_rate = learning_rate[0] * (learning_rate[1] / learning_rate[0]).pow(current_epoch / max_epochs)
                    current_neighbourhood_width = neighbourhood_width[0] * (neighbourhood_width[1] / neighbourhood_width[0]).pow(current_epoch / max_epochs)

                    grid_coordinates = torch.cartesian_prod(*[torch.arange(0, l, device=neurons.device, dtype=neurons.dtype) for l in self.out_features])
                    differences_to_bmu_on_grid = (bmu_location[i] - grid_coordinates)
                    distances_to_bmu_on_grid = differences_to_bmu_on_grid.abs().sum(-1) # Manhattan distance

                    neighbourhood = torch.exp(-distances_to_bmu_on_grid.square() / 2 * current_neighbourhood_width.square())

                    learning = current_learning_rate * neighbourhood.unsqueeze(-1) * input_neuron_differences

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
    def forward(self,  # noqa: PLR0913
                input: torch.Tensor,  # noqa: A002
                current_epoch: int | None = None,
                max_epochs: int | None = None,
                targets: torch.Tensor | None = None, # Unused for unsupervised learning
                som_labelling: SOMLabelling | None = None, # Unused for unsupervised learning,
                return_position: bool = True,  # noqa: FBT001, FBT002
                return_value: bool = True) -> torch.Tensor:  # noqa: FBT001, FBT002
        return self.ksom(input,
                         self.neurons,
                         self.learning_rate,
                         self.neighbourhood_width,
                         current_epoch=current_epoch,
                         max_epochs=max_epochs,
                         return_position=return_position,
                         return_value=return_value,
                         training=self.training)
