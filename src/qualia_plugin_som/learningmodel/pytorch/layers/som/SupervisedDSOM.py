import torch
import torch.nn as nn
import math

from .DSOM import DSOM

class SupervisedDSOM(DSOM):
    def dsom(self, x_batch, y_batch, neurons, learning_rate, elasticity_squared, return_position: bool=True, return_value:
    bool=True, training: bool=True, som_labelling=None):
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

                _, bmu[i] = neurons_distances_to_input_squared.min(axis=-1)

                bmu_location[i] = self.unravel_index(bmu[i], self.out_features)

                if training:

                    y = y_batch[i]
                    matching_neurons = som_labelling.labels.argmax(dim=-1).flatten() == y

                    ### Supervised: select BMU from matching class
                    neurons_distances_to_input_squared_filtered =  torch.where(
                        matching_neurons, # Should have same order as neurons
                        neurons_distances_to_input_squared,
                        torch.full_like(neurons_distances_to_input_squared, float('inf')) # Infinity should never be the min distance
                    )

                    if not matching_neurons.any(): # No matching neuron, do not filter BMU
                        neurons_distances_to_input_squared_filtered = neurons_distances_to_input_squared

                    bmu_distance_to_input_squared, supervised_bmu = neurons_distances_to_input_squared_filtered.min(axis=-1)

                    supervised_bmu_location = self.unravel_index(supervised_bmu, self.out_features)
                    ###

                    grid_coordinates = torch.cartesian_prod(*[torch.arange(0, l, device=neurons.device, dtype=neurons.dtype) for l in self.out_features])
                    differences_to_bmu_on_grid = (supervised_bmu_location - grid_coordinates)
                    distances_to_bmu_on_grid = differences_to_bmu_on_grid.abs().sum(-1)

                    neighbourhood = torch.where(
                        bmu_distance_to_input_squared == bmu_distance_to_input_squared.new_tensor(0),
                        bmu_distance_to_input_squared.new_tensor(0),
                        torch.exp(-distances_to_bmu_on_grid.square() / (elasticity_squared * bmu_distance_to_input_squared))
                    )

                    learning = learning_rate * (neurons_distances_to_input_squared.sqrt() * neighbourhood).unsqueeze(-1) * input_neuron_differences

                    neurons += learning

            # Update labels at each batch
            if training:
                som_labelling(x_batch, y_batch)
                som_labelling.update_labels()

        if return_position and return_value:
            return bmu_location, neurons[bmu].reshape((-1, *self.in_features))
        elif return_position:
            return bmu_location
        elif return_value:
            return neurons[bmu].reshape((-1, *self.in_features))

    def forward(self, input, return_position: bool=True, return_value: bool=True, y=None, som_labelling=None, *args, **kwargs):
        return self.dsom(input, y, self.neurons, self.learning_rate, self.elasticity_squared, return_position=return_position,
        return_value=return_value, training=self.training, som_labelling=som_labelling)
