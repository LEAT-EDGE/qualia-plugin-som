import colorful as cf
import numpy as np
from qualia_core.datamodel import RawDataModel
from qualia_core.postprocessing import PostProcessing


class DNNFineTuning(PostProcessing):
    def __init__(self, epochs, subject: str, test_ratio: float=0.5, source: str='test', train_params: str=None):
        super().__init__()
        self.epochs = epochs
        self.subject = subject
        self.source = source
        self.test_ratio = test_ratio
        self.train_params = train_params

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

    def __call__(self, trainresult, model_conf):
        from pytorch_lightning import Trainer

        framework = trainresult.framework

        # Subject-specific dataset
        subjectdatamodel = self.extract_subject(subject=self.subject,
                                                dataset=getattr(trainresult.datamodel.sets, self.source),
                                                test_ratio = self.test_ratio)
        print(f'{self.subject} train set: {len(subjectdatamodel.sets.train.x)} vectors')
        print(f'{self.subject} test set: {len(subjectdatamodel.sets.test.x)} vectors')

        print(f'{cf.bold}Evaluation of DL model on subject {self.subject} test set before fine-tuning{cf.reset}')
        acc = framework.evaluate(trainresult.model, subjectdatamodel.sets.test, batch_size=trainresult.batch_size)

        # Freeze weights of feature model
        for name, param in trainresult.model.named_parameters():
            if name not in self.train_params:
                param.requires_grad = False

        # Train specific layer of DL model
        print(f'{cf.bold}Fine-tuning \'{self.train_params}\' params on subject {self.subject} train set{cf.reset}')
        # Trainer cannot be re-used for now in PyTorch-Lightning
        framework.train(trainresult.model,
                        trainset=subjectdatamodel.sets.train,
                        validationset=subjectdatamodel.sets.test,
                        epochs=self.epochs,
                        batch_size=trainresult.batch_size,
                        optimizer=trainresult.optimizer,
                        dataaugmentations=trainresult.dataaugmentations,
                        experimenttracking=trainresult.experimenttracking)

        print(f'{cf.bold}Evaluation on subject {self.subject} test set after fine-tuning{cf.reset}')
        acc = framework.evaluate(trainresult.model, subjectdatamodel.sets.test, batch_size=trainresult.batch_size)

        return trainresult
