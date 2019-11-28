import colorful as cf
import numpy as np
from qualia_core.datamodel import RawDataModel
from qualia_core.postprocessing import PostProcessing
from qualia_core.utils.logger import CSVLogger


class DLSOMFineTuning(PostProcessing):
    logger = CSVLogger(name='DLSOMFineTuning', fields=('name', 'i', 'step', 'subject', 'shuffle', 'test_ratio', 'label_ratio', 'subject_label_ratio', 'acc', 'avgclsacc', 'classes_acc'))

    def __init__(self,
                 epochs,
                 subject: str,
                 test_ratio: float=0.5,
                 source: str='test',
                 relabel: str=None,
                 dsom: dict={},
                 shuffle: bool=False,
                 subject_label_ratio: float=None,
                 label_ratio: float=None,
                 drop_unlabelled: bool=False,
                 replay: dict={}):
        super().__init__()

        self.epochs = epochs
        self.subject = subject
        self.source = source
        self.test_ratio = test_ratio
        self.relabel = relabel
        self.dsom_params = dsom
        self.shuffle = shuffle
        self.label_ratio = label_ratio
        self.subject_label_ratio = subject_label_ratio
        self.drop_unlabelled = drop_unlabelled # Drop unlabelled samples from subject training set
        self.replaysource = replay.get('source', None)
        self.replayduplicate = int(replay.get('duplicate', 1))

    def log(self, name, i, step, metrics):
        ncm = metrics['ncm']
        classes_acc = np.diagonal(ncm)
        self.logger(name, i, step, self.subject, self.shuffle, self.test_ratio, self.label_ratio, self.subject_label_ratio, metrics['testacc'], metrics['testavgclsacc'], classes_acc)


    def extract_subject(self, subject: str, dataset, test_ratio: float):
        if len(dataset.info.shape) > 1: # Reduce over windows
            predicate = lambda winfo: all(info['subject'] == subject for info in winfo)
        else: # Already 1 subject per vector
            predicate = lambda info: info['subject'] == subject
        subjectX = [x for x, info in zip(dataset.x, dataset.info) if predicate(info)]
        subjectY = [y for y, info in zip(dataset.y, dataset.info) if predicate(info)]
        subjectinfo = [info for info in dataset.info if predicate(info)]
        if len(subjectX) < 1:
            raise ValueError(f'Subject {subject} not found in dataset')

        # Index to shuffle and split data
        perms = np.random.permutation(len(subjectX))
        testperms = perms[0:int(len(perms)*test_ratio)]
        trainperms = perms[len(testperms):]

        sets = {'train': RawDataModel.Data(x=np.array(subjectX)[trainperms],
                                           y=np.array(subjectY)[trainperms],
                                           info=np.array(subjectinfo)[trainperms]),
                'test': RawDataModel.Data(x=np.array(subjectX)[testperms],
                                          y=np.array(subjectY)[testperms],
                                          info=np.array(subjectinfo)[testperms])}
        return RawDataModel(sets=RawDataModel.Sets(**sets))

    def relabel_som(self, trainresult, trainset):
        print(f'{cf.bold}Relabelling SOM using {self.relabel} train set{cf.reset}')
        trainresult.framework.label_som(model=trainresult.model,
                                        trainset=trainset,
                                        validationset=trainresult.datamodel.sets.test,
                                        epochs=None, # label_som() will use 1 epoc
                                        batch_size=None, # label_som() will use model.som_batch_size
                                        optimizer=None, # No optimizer for label_som()
                                        dataaugmentations=trainresult.dataaugmentations,
                                        experimenttracking=trainresult.experimenttracking
                                        )

    def __call__(self, trainresult, model_conf):
        from qualia_core.learningmodel.pytorch.layers.som import DSOM
        from pytorch_lightning import Trainer
        from torch.utils.data import DataLoader

        framework = trainresult.framework

        if not isinstance(trainresult.model.som.som, DSOM):
            raise ValueError('DLSOMFineTuning requires a DLSOM model with DSOM layer')

        # Make sure weights of DLSOM feature model are frozen
        framework.freeze_fm(trainresult.model)

        # Save original parameters before online learning
        old_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
        old_weights = trainresult.model.som.som.neurons.clone()

        # Subject-specific dataset
        subjectdatamodel = self.extract_subject(subject=self.subject,
                                                dataset=getattr(trainresult.datamodel.sets, self.source),
                                                test_ratio = self.test_ratio)
        print(f'{self.subject} train set: {len(subjectdatamodel.sets.train.x)} vectors')
        print(f'{self.subject} test set: {len(subjectdatamodel.sets.test.x)} vectors')

        if self.subject_label_ratio is not None:
            subjectdatamodel.sets.train = framework.subset_label(subjectdatamodel.sets.train, self.subject_label_ratio, self.drop_unlabelled)

        print(f'{cf.bold}Evaluation of DL model on subject {self.subject} test set before fine-tuning{cf.reset}')
        metrics = framework.evaluate(trainresult.model.dl,
                                 subjectdatamodel.sets.test,
                                 batch_size=trainresult.batch_size,
                                 dataaugmentations=trainresult.dataaugmentations)
        self.log(trainresult.name, trainresult.i, 'original_dl', metrics)

        print(f'{cf.bold}Evaluation of DLSOM model on subject {self.subject} test set before fine-tuning{cf.reset}')
        metrics = framework.evaluate(trainresult.model,
                                 subjectdatamodel.sets.test,
                                 batch_size=trainresult.batch_size,
                                 dataaugmentations=trainresult.dataaugmentations)
        self.log(trainresult.name, trainresult.i, 'original_dlsom', metrics)

        # Apply new training parameters
        print(f'{cf.bold}Applying new training parameters to DSOM{cf.reset}')
        if 'learning_rate' in self.dsom_params:
            print(f'{self.dsom_params["learning_rate"]=}')
            trainresult.model.som.som.learning_rate.copy_(trainresult.model.som.som.learning_rate.new_tensor(self.dsom_params['learning_rate']))
        if 'elasticity' in self.dsom_params:
            print(f'{self.dsom_params["elasticity"]=}')
            trainresult.model.som.som.elasticity_squared.copy_(trainresult.model.som.som.elasticity_squared.new_tensor(self.dsom_params['elasticity']).square())

        # Generate labelset which can be used as replay source for training
        if self.relabel == 'original': # Use original train set to relabel
            labelset = trainresult.datamodel.sets.train
        elif self.relabel == 'subject': # Use subject train set to relabel
            labelset = subjectdatamodel.sets.train
        if self.label_ratio:
            labelset = framework.subset_label(labelset, self.label_ratio, self.drop_unlabelled)

        # Add replay vectors
        if self.replaysource == 'labelled':
            print('{labelset.y.shape}, {subjectdatamodel.sets.train.y.shape}')
            if self.replayduplicate:
                finetuneset = RawDataModel.Data(
                                x=labelset.x.repeat(self.replayduplicate, axis=0),
                                y=labelset.y.repeat(self.replayduplicate, axis=0),
                                info=labelset.info.repeat(self.replayduplicate, axis=0))
            else:
                finetuneset = RawDataModel.Data(
                                x=labelset.x.copy(),
                                y=labelset.y.copy(),
                                info=labelset.info.copy())
            print(f'Adding {len(finetuneset.y)} vectors from "{self.replaysource}" set to fine-tuning set')
            finetuneset.x = np.concatenate([subjectdatamodel.sets.train.x, finetuneset.x])
            finetuneset.y = np.concatenate([subjectdatamodel.sets.train.y, finetuneset.y])
            finetuneset.info = np.concatenate([subjectdatamodel.sets.train.info, finetuneset.info])
            print(f'{len(finetuneset.y)} total vectors in fine-tuning set')
            print(f'{finetuneset.x.shape}, {finetuneset.y.shape}')

        else:
            finetuneset = subjectdatamodel.sets.train
        
        # Train unlabelled SOM model from feature model of dl model
        print(f'{cf.bold}Fine-tuning SOM on subject {self.subject} train set{cf.reset}')
        if len(subjectdatamodel.sets.train.x) < 1:
            print(f'{cf.bold}Warning:{cf.reset} No data to train on!')
        # Trainer cannot be re-used for now in PyTorch-Lightning
        trainer_dlsom = Trainer(max_epochs=self.epochs,
                                accelerator=framework.accelerator,
                                devices=framework.devices,
                                deterministic=True,
                                logger=trainresult.experimenttracking.logger if trainresult.experimenttracking else None)
        trainer_module_dlsom = framework.TrainerModuleDLSOM(model=trainresult.model,
                                                            max_epochs=self.epochs,
                                                            optimizer=None,
                                                            dataaugmentations=trainresult.dataaugmentations,
                                                            num_classes=finetuneset.y.shape[-1])

        trainer_dlsom.fit(trainer_module_dlsom,
                            DataLoader(framework.DatasetFromArray(finetuneset), batch_size=trainresult.model.som_batch_size, shuffle=self.shuffle),
                            DataLoader(framework.DatasetFromArray(subjectdatamodel.sets.test), batch_size=trainresult.model.som_batch_size) if len(subjectdatamodel.sets.test.x) > 0 else None
                        )

        # Show differences after online learning
        new_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
        new_weights = trainresult.model.som.som.neurons.clone()
        print('Modified labels:', (new_labels != old_labels).sum())
        print('Modified neurons:', (new_weights != old_weights).any(-1).sum())

        print(f'{cf.bold}Evaluation on subject {self.subject} test set after fine-tuning{cf.reset}')
        metrics = framework.evaluate(trainresult.model,
                                 subjectdatamodel.sets.test,
                                 batch_size=trainresult.batch_size,
                                 dataaugmentations=trainresult.dataaugmentations)
        self.log(trainresult.name, trainresult.i, 'fine_tuned', metrics)

        if self.relabel:
            # Save original parameters before relabelling
            old_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
            old_weights = trainresult.model.som.som.neurons.clone()

            self.relabel_som(trainresult, trainset=labelset)

            # Show differences after relabelling
            new_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
            new_weights = trainresult.model.som.som.neurons.clone()
            print('Modified labels:', (new_labels != old_labels).sum())
            print('Modified neurons:', (new_weights != old_weights).any(-1).sum())

            print(f'{cf.bold}Evaluation on subject {self.subject} test set after relabelling with {self.relabel}{cf.reset}')
            metrics = framework.evaluate(trainresult.model,
                                     subjectdatamodel.sets.test,
                                     batch_size=trainresult.batch_size,
                                     dataaugmentations=trainresult.dataaugmentations)
            self.log(trainresult.name, trainresult.i, f'relabelled_{self.relabel}', metrics)


        return trainresult
