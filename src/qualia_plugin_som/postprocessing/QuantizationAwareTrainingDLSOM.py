import colorful as cf
from qualia_core.postprocessing import QuantizationAwareTraining
from torch.utils.data import DataLoader


class QuantizationAwareTrainingDLSOM(QuantizationAwareTraining):
    # Entire model should be "trained" for one epoch without optimizer, except for SOMLabelling which does not need quantization
    # and expects labels for training.
    # Ugly hack. For DLSOM, activation ranges have to be updated by putting model in train mode without optimizer
    # However we must prevent the SOM itself from actually modifying its parameter, so the "disable_som_training" parameter of the
    # underlying SOM layer is set to True.
    def _update_activation_ranges(self, trainresult, quantized_model):
        from pytorch_lightning import Trainer

        print(f"{cf.bold}{self.__class__.__name__}: Updating activation ranges{cf.reset}")

        framework = trainresult.framework
        name = f'{trainresult.name}_q{self.width}_r{trainresult.i}'

        quantized_model.som.som.disable_som_training = True

        # Make sure weights of DLSOM feature model are frozen
        framework.freeze_fm(quantized_model)

        # Must train DLSOM to update activation ranges, SOMLabelling excluded
        trainer_dlsom = Trainer(max_epochs=1,
                                accelerator=framework.accelerator,
                                devices=framework.devices,
                                deterministic=True,
                                logger=framework.logger(trainresult.experimenttracking, name=name))
        trainer_module_dlsom = framework.TrainerModuleDLSOM(model=quantized_model,
                                                            max_epochs=1,
                                                            optimizer=None, # Disable optimizer
                                                            dataaugmentations=trainresult.dataaugmentations,
                                                            num_classes=trainresult.trainset.y.shape[-1])
        trainer_dlsom.fit(trainer_module_dlsom,
                            DataLoader(framework.DatasetFromArray(trainresult.trainset), batch_size=quantized_model.som_batch_size, shuffle=True),
                            DataLoader(framework.DatasetFromArray(trainresult.testset), batch_size=quantized_model.som_batch_size) if len(trainresult.testset.x) > 0 else None
                        )



        quantized_model.som.som.disable_som_training = False

    def _quantization_aware_training(self, trainresult, quantized_model):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        framework = trainresult.framework
        name = f'{trainresult.name}_q{self.width}_r{trainresult.i}'

        # Make sure weights of DLSOM feature model are frozen
        framework.freeze_fm(quantized_model)

        if quantized_model.quantize_dl == False and quantized_model.quantize_som == True:
            print(f"{cf.bold}{self.__class__.__name__}: Performing quantization-aware-training for {self.epochs} epochs{cf.reset}")
            # You can plot the quantize training graph on tensorboard

            # Train SOM model from feature model of dl model
            print(f'{cf.bold}Training SOM{cf.reset}')

            early_stop_callback = EarlyStopping(monitor="trainmse", mode='min', min_delta=0.0, patience=9999, check_finite=True, strict=True, )
            # Trainer cannot be re-used for now in PyTorch-Lightning
            trainer_dlsom = Trainer(max_epochs=self.epochs,
                                    accelerator=framework.accelerator,
                                    devices=framework.devices,
                                    deterministic=True,
                                    logger=framework.logger(trainresult.experimenttracking, name=name),
                                    callbacks=[early_stop_callback])
            trainer_module_dlsom = framework.TrainerModuleDLSOM(model=quantized_model,
                                        max_epochs=self.epochs,
                                        optimizer=None,
                                        dataaugmentations=trainresult.dataaugmentations,
                                        num_classes=trainresult.trainset.y.shape[-1])
            trainer_dlsom.fit(trainer_module_dlsom,
                                DataLoader(framework.DatasetFromArray(trainresult.trainset), batch_size=self.batch_size, shuffle=True),
                                DataLoader(framework.DatasetFromArray(trainresult.testset), batch_size=self.batch_size*16) if len(trainresult.testset.x) > 0 else None
                            )

            #framework.train_som(quantized_model,
            #                trainset=trainresult.trainset,
            #                validationset=trainresult.testset,
            #                epochs=self.epochs,
            #                batch_size=self.batch_size,
            #                optimizer=None,
            #                dataaugmentations=trainresult.dataaugmentations,
            #                experimenttracking=trainresult.experimenttracking)

            # Relabel SOM
            labelset = framework.subset_label(trainresult.trainset, framework.label_ratio, drop_unlabelled=True)
            framework.label_som(quantized_model,
                                trainset=labelset,
                                validationset=trainresult.testset,
                                epochs=1,
                                batch_size=self.batch_size,
                                optimizer=None,
                                dataaugmentations=trainresult.dataaugmentations,
                                experimenttracking=trainresult.experimenttracking)
        else:
            raise NotImplemented()
