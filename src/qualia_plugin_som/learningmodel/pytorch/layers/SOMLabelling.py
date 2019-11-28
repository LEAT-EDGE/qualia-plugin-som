import sys
import math
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

class SOMLabelling(nn.Module, Callback):
    def __init__(self, out_features: tuple, som, sigma: float, device=None, dtype=None):
        super().__init__()

        if len(out_features) != 1:
            raise ValueError('out_features must be 1D')

        self.in_features = som.in_features
        self.out_features = out_features

        self.som = som

        self.sigma = nn.Parameter(torch.tensor(sigma, device=device, dtype=dtype), requires_grad=False)

        self.labels = nn.Parameter(nn.functional.one_hot(
                                    torch.randint(0, out_features[-1], som.out_features + tuple(out_features[:-1]), device=device),
                                    num_classes=out_features[-1]).double().to(dtype), requires_grad=False)

        self.activities = nn.Parameter(torch.zeros((math.prod(som.out_features), out_features[0]), device=device, dtype=dtype), requires_grad=False)

        self.labels_count = nn.Parameter(torch.zeros(tuple(out_features), device=device, dtype=torch.long), requires_grad=False)

    def forward(self, x, y=None):
        debug = False

        with torch.no_grad():
            if self.training:
                # Training mode with x being the input data to the SOM and y the associated labels

                if y is None:
                    print(f'{self.__class__.__name__} requires labels as second input to forward()', file=sys.stderr)
                    return

                x = x.reshape((x.shape[0], 1, -1))

                #### Compute neurons distance to input
                input_neuron_differences = (x - self.som.neurons)

                input_neuron_differences_squared = input_neuron_differences.pow(2)

                neurons_distances_to_input_squared = input_neuron_differences_squared.sum(-1)

                neurons_distances_to_input = neurons_distances_to_input_squared.sqrt()
                ####

                
                #### Find distance of BMU to input
                # Perform before Gaussian method because limited float32 precision could cause many values to fall to 0 after exp()
                bmu_distance_to_input_squared, bmu = neurons_distances_to_input_squared.min(axis=-1)

                bmu_distance_to_input_gaussian = (-bmu_distance_to_input_squared.sqrt() / self.sigma).exp()
                ####


                #### Compute Gaussian method
                neurons_distances_to_input_gaussian = (-neurons_distances_to_input / self.sigma).exp()
                ####

                #### Normalize distances
                bmu_distance_to_input_gaussian_reshaped_over_neurons = bmu_distance_to_input_gaussian.unsqueeze(1)

                neurons_distances_to_input_gaussian_normalized = torch.where(
                    bmu_distance_to_input_gaussian_reshaped_over_neurons == bmu_distance_to_input_gaussian_reshaped_over_neurons.new_tensor(0),
                    bmu_distance_to_input_gaussian_reshaped_over_neurons.new_tensor(0),
                    neurons_distances_to_input_gaussian / bmu_distance_to_input_gaussian_reshaped_over_neurons
                )
                ####


                #### Put batch dimension last for activities indexing
                neurons_distances_to_input_gaussian_normalized_batch_last = torch.movedim(neurons_distances_to_input_gaussian_normalized, 0, -1)
                ####

                
                #### Accumulate actitivites
                self.activities[...,y] += neurons_distances_to_input_gaussian_normalized_batch_last
                ####


                ### Count labels
                self.labels_count[y] += 1
                ###

                ### Compute BMU location on 2D grid from 1D index
                bmu_location = self.som.unravel_index(bmu, self.som.out_features)
                ###

                return self.labels[bmu_location.split(1, dim=-1)].squeeze(1)
            else:
                if y is None: # Inference mode with input x being SOM outputs (position of BMUs)
                    return self.labels[x.split(1, dim=-1)].squeeze(1)
                else: # For validation step of PyTorch Lightning during labelling
                    return torch.zeros_like(y)

    def update_labels(self):
        self.activities /= self.labels_count

        label_from_activities = self.activities.argmax(dim=-1).reshape(self.som.out_features)
        label_one_hot = torch.eye(self.out_features[0])[label_from_activities]

        self.labels.copy_(label_one_hot)

    def reset(self):
        self.activities.fill_(0)
        self.labels_count.fill_(0)

    def on_train_end(self, trainer, pl_module):
        self.update_labels()
        self.reset()
