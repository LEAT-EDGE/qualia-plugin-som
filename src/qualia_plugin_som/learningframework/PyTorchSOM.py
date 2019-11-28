import torchmetrics
from qualia_core.learningframework import PyTorch
from qualia_plugin_som.learningmodel.pytorch.layers import SOMLabelling
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader


class PyTorchSOM(PyTorch):
    import qualia_plugin_som.learningmodel.pytorch as learningmodels
    learningmodels.__dict__.update(PyTorch.learningmodels.__dict__) # Merge core models back. Warning: module name changes too!

    class TrainerModuleSOM(PyTorch.TrainerModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            metrics = torchmetrics.MetricCollection({
                'mse': torchmetrics.MeanSquaredError(),
            })
            self.train_metrics = metrics.clone(prefix='train')
            self.val_metrics = metrics.clone(prefix='val')
            self.test_metrics = metrics.clone(prefix='test')

        def training_step(self, batch, batch_nb):
            x, y = batch
            if len(x.shape) > 2:
                x = self.model.flatten(x)
            bmu_indices, bmu_vectors = self.model.som(x, current_epoch=self.current_epoch, max_epochs=self.trainer.max_epochs)
            self.train_metrics(bmu_vectors, x)
            self.log_dict(self.train_metrics, prog_bar=True)

        def validation_step(self, batch, batch_nb):
            x, y = batch
            if len(x.shape) > 2:
                x = self.model.flatten(x)
            bmu_indices, bmu_vectors = self.model.som(x)
            self.val_metrics(bmu_vectors, x)
            self.log_dict(self.val_metrics, prog_bar=True)
        
        def on_train_epoch_end(self):
            pass

    class TrainerModuleSOMLabelling(PyTorch.TrainerModule):
        def training_step(self, batch, batch_nb):
            x, y = batch
            logits = self.model(x, y)
            self.train_metrics(logits, y)
            self.log_dict(self.train_metrics, prog_bar=True)

        def validation_step(self, batch, batch_nb):
            x, y = batch
            logits = self.model(x, y)
            self.val_metrics(logits, y)
            self.log_dict(self.val_metrics, prog_bar=True)

        def configure_callbacks(self):
            return [self.model]

    def plot_neurons(self, model, trainset):
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
                sub.set_title(model.som_labelling.labels[x][y].argmax())
                data = model.som.neurons[x * map_hgt + y]
                data = data.reshape((model.input_shape[1], model.input_shape[0]))
                #data = data.swapaxes(0, 1)
                #data = data.reshape(data.shape[0] * data.shape[1])
                for channel in data:
                    clr = sub.plot(channel)#, linestyle='None', marker=',')

        som_grid.tight_layout(pad=0, w_pad=0, h_pad=0.1)

        plt.show()

    def train(self, model, trainset, validationset, epochs, batch_size, optimizer, dataaugmentations=None, experimenttracking=None, name=None):
        print(f'Training SOM, {epochs=}')
        # Train SOM
        early_stop_callback = EarlyStopping(monitor="trainmse", mode='min', min_delta=0.0, patience=9999, check_finite=True, strict=True)
        trainer_som = Trainer(max_epochs=epochs,
                              accelerator=self.accelerator,
                              devices=self.devices,
                              deterministic=True,
                              logger=self.logger(experimenttracking, name=name),
                              enable_progress_bar=self._enable_progress_bar,
                              callbacks=[early_stop_callback])
        trainer_module_som = self.TrainerModuleSOM(model=model, max_epochs=epochs, optimizer=None, dataaugmentations=dataaugmentations, num_classes=trainset.y.shape[-1])
        trainer_som.fit(trainer_module_som,
                            DataLoader(self.DatasetFromArray(trainset), batch_size=batch_size, shuffle=True),
                            DataLoader(self.DatasetFromArray(validationset), batch_size=batch_size) if len(validationset.x) > 0 else None
                        )


        print('Labelling SOM')
        # Label SOM
        trainer_som_labelling = Trainer(max_epochs=1,
                                        accelerator=self.accelerator,
                                        devices=self.devices,
                                        deterministic=True,
                                        logger=self.logger(experimenttracking, name=name),
                                        enable_progress_bar=self._enable_progress_bar)
        trainer_module_som_labelling = self.TrainerModuleSOMLabelling(model=model.som_labelling, max_epochs=1, optimizer=None, dataaugmentations=dataaugmentations, num_classes=trainset.y.shape[-1])
        trainer_som_labelling.fit(trainer_module_som_labelling,
                            DataLoader(self.DatasetFromArray(trainset), batch_size=32, shuffle=True),
                            DataLoader(self.DatasetFromArray(validationset), batch_size=32) if len(validationset.x) > 0 else None
                        )
        #self.plot_neurons(model, trainset)
        print('SOM training and labelling finished')
