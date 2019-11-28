import torch
import torch.nn as nn
import math

class DSOM(nn.Module):
    def __init__(self, in_features: tuple, out_features: tuple, learning_rate: float=0.01, elasticity: float=0.01, device=None, dtype=None):
        super().__init__()

        self.out_features = out_features
        self.in_features = in_features

        self.elasticity_squared = nn.Parameter(torch.tensor(elasticity, device=device, dtype=dtype).square(), requires_grad=False)
        self.learning_rate = nn.Parameter(torch.tensor(learning_rate, device=device, dtype=dtype), requires_grad=False)

        self.neurons = nn.Parameter(torch.empty((math.prod(out_features), math.prod(in_features)), device=device, dtype=dtype), requires_grad=False)
        
        with torch.no_grad():
            torch.nn.init.uniform_(self.neurons, a=0.0, b=1.0),

    # From https://stackoverflow.com/a/65168284/1447751
    def unravel_index(self,
        indices: torch.LongTensor,
        shape: tuple,
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

    
    def dsom(self, x_batch, neurons, learning_rate, elasticity_squared, return_position: bool=True, return_value: bool=True, training: bool=True):
        with torch.no_grad():
            x_batch = x_batch.reshape((x_batch.shape[0], 1, -1)) # Flatten input dimensions

            bmu = torch.empty((x_batch.shape[0]), device=x_batch.device, dtype=torch.long) # Preallocate BMU indices array for entire batch
            bmu_location = torch.empty((x_batch.shape[0], len(self.out_features)), device=x_batch.device, dtype=torch.long) # Preallocate BMU grid position for entire batch

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

                    grid_coordinates = torch.cartesian_prod(*[torch.arange(0, l, device=neurons.device, dtype=neurons.dtype) for l in self.out_features])
                    differences_to_bmu_on_grid = (bmu_location[i] - grid_coordinates)
                    distances_to_bmu_on_grid = differences_to_bmu_on_grid.abs().sum(-1)

                    neighbourhood = torch.where(
                        bmu_distance_to_input_squared == bmu_distance_to_input_squared.new_tensor(0),
                        bmu_distance_to_input_squared.new_tensor(0),
                        torch.exp(-distances_to_bmu_on_grid.square() / (elasticity_squared * bmu_distance_to_input_squared))
                    )

                    learning = learning_rate * (neurons_distances_to_input_squared.sqrt() * neighbourhood).unsqueeze(-1) * input_neuron_differences

                    neurons += learning

        if return_position and return_value:
            return bmu_location, neurons[bmu].reshape((-1, *self.in_features))
        elif return_position:
            return bmu_location
        elif return_value:
            return neurons[bmu].reshape((-1, *self.in_features))

    def forward(self, input, return_position: bool=True, return_value: bool=True, *args, **kwargs):
        return self.dsom(input, self.neurons, self.learning_rate, self.elasticity_squared, return_position=return_position, return_value=return_value, training=self.training)
