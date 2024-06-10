from __future__ import annotations

import logging
import os
import sys

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from qualia_core.learningframework import PyTorch
from qualia_core.typing import TYPE_CHECKING
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pytorch_lightning import Callback  # noqa: TCH002
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT  # noqa: TCH002
    from qualia_core.dataaugmentation.DataAugmentation import DataAugmentation  # noqa: TCH002
    from qualia_core.datamodel.RawDataModel import RawData  # noqa: TCH002
    from qualia_core.experimenttracking.ExperimentTracking import ExperimentTracking  # noqa: TCH002
    from qualia_core.typing import OptimizerConfigDict
    from torch import nn  # noqa: TCH002
    import torch

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class PyTorchSOM(PyTorch):
    import qualia_plugin_som.learningmodel.pytorch as learningmodels
    learningmodels.__dict__.update(PyTorch.learningmodels.__dict__) # Merge core models back. Warning: module name changes too!

    class TrainerModuleSOM(PyTorch.TrainerModule):
        @override
        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
            x, y = batch
            if len(x.shape) > 2:
                x = self.model.flatten(x)
            bmu_indices, bmu_vectors = self.model.som(x, current_epoch=self.current_epoch, max_epochs=self.trainer.max_epochs)
            self.train_metrics(bmu_vectors, x)
            self.log_dict(self.train_metrics, prog_bar=True)

        @override
        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            if len(x.shape) > 2:
                x = self.model.flatten(x)
            bmu_indices, bmu_vectors = self.model.som(x)
            self.val_metrics(bmu_vectors, x)
            self.log_dict(self.val_metrics, prog_bar=True)

    class TrainerModuleSOMLabelling(PyTorch.TrainerModule):
        @override
        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
            x, y = batch
            logits = self.model(x, y)
            self.train_metrics(logits, y)
            self.log_dict(self.train_metrics, prog_bar=True)

        @override
        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            logits = self.model(x, y)
            self.val_metrics(logits, y)
            self.log_dict(self.val_metrics, prog_bar=True)

        def configure_callbacks(self):
            return [self.model]

    def plot_neurons(self, model: nn.Module, trainset: RawData):
        import matplotlib.pyplot as plt

        map_wth = model.som.out_features[0]
        map_hgt = model.som.out_features[1]


        # display neurons weights as mnist digits
        som_grid = plt.figure(figsize=(22, 22)) # width, height in inches
        for x in range(map_wth):
            for y in range(map_hgt):
                sub = som_grid.add_subplot(map_wth, map_hgt, x * map_hgt + y + 1)
                #sub.set_axis_off()
                #sub.set_title(model.som_labelling.labels[x][y].argmax())
                #data = model.som.neurons[x][y]
                #data = data.reshape(data.shape[0] * data.shape[1])
                _ = sub.set_title(model.som_labelling.labels[x][y].argmax())
                data = model.som.neurons[x * map_hgt + y]
                data = data.reshape((model.input_shape[1], model.input_shape[0]))
                #data = data.swapaxes(0, 1)
                #data = data.reshape(data.shape[0] * data.shape[1])
                for channel in data:
                    clr = sub.plot(channel)#, linestyle='None', marker=',')

        som_grid.tight_layout(pad=0, w_pad=0, h_pad=0.1)

        plt.show()

    @override
    def train(self,
              model: nn.Module,
              trainset: RawData | None,
              validationset: RawData | None,
              epochs: int,
              batch_size: int,
              optimizer: OptimizerConfigDict | None,
              dataaugmentations: list[DataAugmentation] | None = None,
              experimenttracking: ExperimentTracking | None = None,
              name: str | None = None,
              precision: _PRECISION_INPUT | None = None) -> nn.Module:
        logger.info('Training SOM')

        ## Copied from qualia_core.learningframework.PyTorch, need to refactor one day…
        from pytorch_lightning import Trainer, seed_everything
        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

        # PyTorch-Lightning >= 1.3.0 resets seed before training
        # increment seed between trainings to get different values between experiments
        seed = os.environ.get('PL_GLOBAL_SEED', None)
        if seed is None:
            logger.warning('PyTorch not seeded')
        else:
            _ = seed_everything((int(seed) * 100) % 4294967295)

        checkpoint_callback = ModelCheckpoint(dirpath=f'out/checkpoints/{name}',
                                              save_top_k=2,
                                              monitor=self._checkpoint_metric['name'],
                                              mode=self._checkpoint_metric['mode'])
        callbacks: list[Callback] = [checkpoint_callback]
        if self._enable_progress_bar:
            callbacks.append(TQDMProgressBar(refresh_rate=self._progress_bar_refresh_rate))

        experimenttracking_init = experimenttracking.initializer if experimenttracking is not None else None

        ### End of copy

        # Stop training when not converging or diverging
        early_stopping_callback = EarlyStopping(monitor='trainmse',
                                                mode='min',
                                                min_delta=0.0,
                                                patience=9999,
                                                check_finite=True,
                                                strict=True)
        # Train SOM
        trainer_som = Trainer(max_epochs=epochs,
                              accelerator=self.accelerator,
                              devices=self.devices,
                              precision=self.precision if precision is None else precision,
                              deterministic=True,
                              logger=self.logger(experimenttracking, name=name),
                              enable_progress_bar=self._enable_progress_bar,
                              callbacks=[*callbacks, early_stopping_callback])
        trainer_module_som = self.TrainerModuleSOM(model=model,
                                                   max_epochs=epochs,
                                                   optimizer=None,
                                                   dataaugmentations=dataaugmentations,
                                                   num_outputs=trainset.y.shape[-1],
                                                   experimenttracking_init=experimenttracking_init,
                                                   loss=None,
                                                   metrics=['mse'])
        trainer_som.fit(trainer_module_som,
                        DataLoader(self.DatasetFromArray(trainset),
                                   batch_size=batch_size,
                                   shuffle=True),
                        DataLoader(self.DatasetFromArray(validationset),
                                   batch_size=batch_size) if validationset is not None else None,
                        )


        logger.info('Labelling SOM')
        # Label SOM
        trainer_som_labelling = Trainer(max_epochs=1,
                                        accelerator=self.accelerator,
                                        devices=self.devices,
                                        precision=self.precision if precision is None else precision,
                                        deterministic=True,
                                        logger=self.logger(experimenttracking, name=name),
                                        enable_progress_bar=self._enable_progress_bar)
        trainer_module_som_labelling = self.TrainerModuleSOMLabelling(model=model.som_labelling,
                                                                      max_epochs=1,
                                                                      optimizer=None,
                                                                      dataaugmentations=dataaugmentations,
                                                                      num_outputs=trainset.y.shape[-1],
                                                                      experimenttracking_init=experimenttracking_init,
                                                                      loss=None,
                                                                      metrics=['prec', 'rec', 'f1', 'acc', 'avgclsacc'])
        trainer_som_labelling.fit(trainer_module_som_labelling,
                                  DataLoader(self.DatasetFromArray(trainset),
                                             batch_size=32,
                                             shuffle=True),
                                  DataLoader(self.DatasetFromArray(validationset),
                                             batch_size=32) if validationset is not None else None,
                                  )
        #self.plot_neurons(model, trainset)
        logger.info('SOM training and labelling finished')

        return model
