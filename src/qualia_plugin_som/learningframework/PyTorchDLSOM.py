import colorful as cf
import numpy as np
import torchmetrics
from qualia_core.datamodel import RawDataModel
from qualia_core.learningframework import PyTorch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader


class PyTorchDLSOM(PyTorch):
    import qualia_plugin_som.learningmodel.pytorch as learningmodels
    learningmodels.__dict__.update(PyTorch.learningmodels.__dict__) # Merge core models back. Warning: module name changes too!

    class TrainerModuleDLSOM(PyTorch.TrainerModule):
        def configure_metrics(self, *args, **kwargs):
            metrics = torchmetrics.MetricCollection({
                'mse': torchmetrics.MeanSquaredError(),
            })
            self.train_metrics = metrics.clone(prefix='train')
            self.val_metrics = metrics.clone(prefix='val')
            self.test_metrics = metrics.clone(prefix='test')

            #self.train_mse = torchmetrics.MeanSquaredError()
            #self.valid_mse = torchmetrics.MeanSquaredError(compute_on_step=False)

        def training_step(self, batch, batch_nb):
            x, y = batch
            fm = self.model.fm(x)
            ffm = self.model.flatten(fm)
            nfm = self.model.normalizeminmax(ffm)
            bmu_indices, bmu_vectors = self.model.som.som(nfm,
                                                          current_epoch=self.current_epoch,
                                                          max_epochs=self.trainer.max_epochs,
                                                          y=y,
                                                          som_labelling=self.model.som.som_labelling)

            #self.train_mse(bmu_vectors, nfm)
            #self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
            self.train_metrics(bmu_vectors, nfm)
            self.log_dict(self.train_metrics, prog_bar=True)

        def validation_step(self, batch, batch_nb):
            x, y = batch
            fm = self.model.fm(x)
            ffm = self.model.flatten(fm)
            nfm = self.model.normalizeminmax(ffm)
            bmu_indices, bmu_vectors = self.model.som.som(nfm)

            #self.valid_mse(bmu_vectors, nfm)
            #self.log('valid_mse', self.valid_mse, on_step=False, on_epoch=True, prog_bar=True)
            self.val_metrics(bmu_vectors, nfm)
            self.log_dict(self.val_metrics, prog_bar=True)

        def on_train_epoch_end(self):
            pass

    class TrainerModuleDLSOMLabelling(PyTorch.TrainerModule):
        def training_step(self, batch, batch_nb):
            x, y = batch
            fm = self.model.fm(x)
            ffm = self.model.flatten(fm)
            nfm = self.model.normalizeminmax(ffm)
            logits = self.model.som.som_labelling(nfm, y)
            self.train_metrics(logits, y)
            self.log_dict(self.train_metrics, prog_bar=True)

        def validation_step(self, batch, batch_nb):
            x, y = batch
            fm = self.model.fm(x)
            ffm = self.model.flatten(fm)
            nfm = self.model.normalizeminmax(ffm)
            logits = self.model.som.som_labelling(nfm, y)
            self.val_metrics(logits, y)
            self.log_dict(self.val_metrics, prog_bar=True)

        def configure_callbacks(self):
            return [self.model.som.som_labelling]

    def __init__(self, label_ratio: float=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f'{label_ratio=}')
        self.label_ratio = label_ratio

    def plot_neurons(self, model):
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
                #data = data.reshape((trainset.x.shape[2], trainset.x.shape[1]))
                #data = data.swapaxes(0, 1)
                #data = data.reshape(data.shape[0] * data.shape[1])
                sub.plot(data, linestyle='None', marker=',')

        som_grid.tight_layout(pad=0, w_pad=0, h_pad=0.1)

        plt.show()


    def train_dl(self, model, trainset, validationset, epochs, batch_size, *args, **kwargs):
        # Train dl model with parent trainer, also increments random state seed 
        print(f'{cf.bold}Training DL{cf.reset}')
        super().train(model.dl,
                        trainset=trainset,
                        validationset=validationset,
                        epochs=model.dl_epochs,
                        batch_size=model.dl_batch_size,
                        *args, **kwargs)

    def evaluate_dl(self, model, trainset, validationset, epochs, batch_size, optimizer, *args, **kwargs):
        if 'Quantized' in model.dl.__class__.__name__:
            print(f"{cf.bold}{self.__class__.__name__}: Updating activation ranges{cf.reset}")
            # Must train to update activation range
            super().train(model.dl,
                            trainset=trainset,
                            validationset=validationset,
                            epochs=1,
                            batch_size=model.dl_batch_size,
                            optimizer=None, # Disable optimizer
                            *args, **kwargs)

        print(f'{cf.bold}Evaluating DL on train dataset{cf.reset}')
        super().evaluate(model.dl,
                        testset=trainset,
                        batch_size=model.dl_batch_size,
                        *args, **kwargs)

        print(f'{cf.bold}Evaluating DL on test dataset{cf.reset}')
        super().evaluate(model.dl,
                        testset=validationset,
                        batch_size=model.dl_batch_size,
                        *args, **kwargs)


    def freeze_fm(self, model):
        #Freeze weights of feature model
        for param in model.fm.parameters():
            param.requires_grad = False
        model.fm.eval()

    def train_som(self,
                  model,
                  trainset,
                  validationset,
                  epochs,
                  batch_size,
                  optimizer,
                  dataaugmentations=None,
                  experimenttracking=None,
                  name: str=None):
        # Train unlabelled SOM model from feature model of dl model
        print(f'{cf.bold}Training SOM{cf.reset}')

        early_stop_callback = EarlyStopping(monitor="trainmse", mode='min', min_delta=0.0, patience=9999, check_finite=True, strict=True, )
        # Trainer cannot be re-used for now in PyTorch-Lightning
        trainer_dlsom = Trainer(max_epochs=model.som_epochs,
                                accelerator=self.accelerator,
                                devices=self.devices,
                                deterministic=True,
                                logger=self.logger(experimenttracking, name=name),
                                enable_progress_bar=self._enable_progress_bar,
                                callbacks=[early_stop_callback])
        trainer_module_dlsom = self.TrainerModuleDLSOM(model=model, max_epochs=model.som_epochs, optimizer=None, dataaugmentations=dataaugmentations)
        trainer_dlsom.fit(trainer_module_dlsom,
                            DataLoader(self.DatasetFromArray(trainset), batch_size=model.som_batch_size, shuffle=True),
                            DataLoader(self.DatasetFromArray(validationset), batch_size=model.som_batch_size) if len(validationset.x) > 0 else None
                        )

    def subset_label(self, trainset, label_ratio, drop_unlabelled):
        print(f'{label_ratio=}')
        perms = np.random.permutation(len(trainset.y))
        labelperms = perms[0:int(len(trainset.y) * label_ratio)]

        if drop_unlabelled:
            labelset = RawDataModel.Data(x=trainset.x[labelperms],
                                         y=trainset.y[labelperms],
                                         info=trainset.info)
        else:
            labelmask = np.zeros_like(trainset.y, dtype=np.bool)
            labelmask[labelperms] = True
            labels = trainset.y.copy()
            labels[~labelmask] = np.nan
            labelset = RawDataModel.Data(x=trainset.x,
                                         y=labels,
                                         info=trainset.info)

        print(f'Labelled vectors: {len([y for y in labelset.y if not np.isnan(y).any()])} / {len(labelset.y)} of {len(trainset.y)}')
        return labelset

    def label_som(self, model, trainset, validationset, epochs, batch_size, optimizer, dataaugmentations=None, experimenttracking=None, name: str=None):
        if len(trainset.x) < 1:
            print('Warning: labelset is empty, not labelling SOM')
            return

        old_labels = model.som.som_labelling.labels.argmax(-1).clone()
        print(f'{old_labels=}')

        # Label SOM
        print(f'{cf.bold}Labelling SOM{cf.reset}')
        # Trainer cannot be re-used for now in PyTorch-Lightning
        trainer_dlsom_labelling = Trainer(max_epochs=1,
                                          accelerator=self.accelerator,
                                          devices=self.devices,
                                          deterministic=True,
                                          logger=self.logger(experimenttracking, name=name),
                                          enable_progress_bar=self._enable_progress_bar)
        trainer_module_dlsom_labelling = self.TrainerModuleDLSOMLabelling(model=model, max_epochs=1, optimizer=None, dataaugmentations=dataaugmentations, num_classes=trainset.y.shape[-1])
        trainer_dlsom_labelling.fit(trainer_module_dlsom_labelling,
                            DataLoader(self.DatasetFromArray(trainset), batch_size=model.dl_batch_size, shuffle=True),
                            DataLoader(self.DatasetFromArray(validationset), batch_size=model.dl_batch_size) if len(validationset.x) > 0 else None
                        )

        new_labels = model.som.som_labelling.labels.argmax(-1).clone()
        print(f'{new_labels=}')
        print('Modified labels:', (new_labels != old_labels).sum())

        print(f'{cf.bold}Evaluating DLSOM on labelling dataset{cf.reset}')
        super().evaluate(model,
                        testset=trainset,
                        batch_size=model.dl_batch_size,
                        accelerator=self.accelerator,
                        devices=self.devices,
                        dataaugmentations=dataaugmentations,
                        experimenttracking=experimenttracking,
                        name=name)

    def train(self, model, trainset, *args, **kwargs):
        if model.dl_epochs > 0:
            self.train_dl(model, trainset=trainset, *args, **kwargs)
        else:
            self.evaluate_dl(model, trainset=trainset, *args, **kwargs)
        self.freeze_fm(model)
        if model.som_epochs > 0:
            self.train_som(model, trainset=trainset, *args, **kwargs)
        labelset = self.subset_label(trainset, self.label_ratio, drop_unlabelled=True)
        self.label_som(model, trainset=labelset, *args, **kwargs)

        #self.plot_neurons(model.som)
