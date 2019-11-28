import random
from pathlib import Path

import colorful as cf
import numpy as np
from qualia_core.datamodel import RawDataModel
from qualia_core.postprocessing import PostProcessing
from qualia_core.utils.logger import CSVLogger


class DLSOMCORe50FineTuning(PostProcessing):
    log = CSVLogger(name='DLSOMCORe50FineTuning', fields=('name', 'i', 'step', 'session', 'acc'))

    def __init__(self,
                 epochs,
                 first_batch: str=None,
                 batches: list=None,
                 shuffle_batches: bool=True,
                 relabel: str=None,
                 dsom: dict={}):
        super().__init__()
        self.epochs = epochs
        self.first_batch = first_batch
        self.batches = batches
        self.shuffle_batches = shuffle_batches
        if relabel and relabel not in ['original', 'batch']:
            raise ValueError(f'{self.__class__.__name__} unknown relabel {relabel}, must be original or batch')
        self.relabel = relabel
        self.dsom_params = dsom
        self.pp = None

    def __del__(self):
        if self.pp:
            self.pp.close()

    def plot_neurons(self, trainresult, batch, step, neurons2d=None, clusters=None, modified_labels=None, outdir=Path('out')/'DLSOMCORe50FineTuning'/'tsne'):
        if modified_labels is None:
            modified_labels = np.zeros_like(trainresult.model.som.som.neurons.numpy(), dtype=np.bool).flatten()
        else:
            modified_labels = modified_labels.numpy().flatten()

        # Compute T-SNE
        if neurons2d is None:
            from sklearn.manifold import TSNE
            tsne = TSNE()
            neurons2d = tsne.fit_transform(trainresult.model.som.som.neurons.numpy())

        # Compute clustering
        if clusters is None:
            import hdbscan
            import sklearn.cluster as sklcluster
            #cluster_func = sklcluster.OPTICS(min_samples=5, xi=0.000005)
            #cluster_func = sklcluster.DBSCAN(eps=2.0)
            cluster_func = hdbscan.HDBSCAN(min_samples=2)
            #clusters = cluster_func.fit_predict(trainresult.model.som.som.neurons.numpy())
            clusters = cluster_func.fit_predict(neurons2d)

        # Plot 2D neurons
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        BASE_COLORS_W_TO_BEIGE = mcolors.BASE_COLORS
        mcolors.BASE_COLORS['w'] = mcolors.CSS4_COLORS['beige']
        colors = [*mcolors.TABLEAU_COLORS.values(), *BASE_COLORS_W_TO_BEIGE.values()]

        if not self.pp:
            outdir.mkdir(parents=True, exist_ok=True)
            self.pp = PdfPages(outdir/f'{trainresult.name}.pdf')
        fig = plt.figure(figsize=(8.25, 4.7))
        plt.title(f'{trainresult.name}_{trainresult.i}, session {batch}, {step}')
        plt.scatter(*neurons2d.T, marker='x', c=[colors[ci % len(colors)] for ci in clusters])
        for label, u, changed in zip(trainresult.model.som.som_labelling.labels.argmax(-1).flatten().numpy(), neurons2d, modified_labels):
            plt.annotate(label, u, c='red' if changed else 'black')

        # Legend
        cluster_indices = sorted(set(clusters))
        print(f'Number of clusters: {len(cluster_indices)}')
        handles = [plt.Rectangle((0,0),1,1, color=colors[ci % len(colors)]) for ci in cluster_indices]
        plt.legend(handles, cluster_indices)

        self.pp.savefig(fig, pad_inches=0, bbox_inches='tight')
        plt.close(fig)

        # Plot neurons grid
        neurons_grid = np.indices(trainresult.model.som.som.out_features).reshape(2,-1).T
        fig = plt.figure(figsize=(8.25, 4.7))
        plt.scatter(*neurons_grid.T, marker='o', c=[colors[ci % len(colors)] for ci in clusters])
        for label, u, changed in zip(trainresult.model.som.som_labelling.labels.argmax(-1).flatten().numpy(), neurons_grid, modified_labels):
            plt.annotate(label, u, c='red' if changed else 'black')


        self.pp.savefig(fig, pad_inches=0, bbox_inches='tight')
        plt.close(fig)

        #self.pp.close()
        #self.pp = None

        return neurons2d, clusters

    def extract_session(self, session: str, dataset):
        predicate = lambda info: info['session'] == session
        sessionX = [x for x, info in zip(dataset.x, dataset.info) if predicate(info)]
        sessionY = [y for y, info in zip(dataset.y, dataset.info) if predicate(info)]
        sessioninfo = [info for info in dataset.info if predicate(info)]
        if len(sessionX) < 1:
            raise ValueError(f'Session {session} not found in dataset')

        sets = {'train': RawDataModel.Data(x=np.array(sessionX),
                                           y=np.array(sessionY),
                                           info=np.array(sessioninfo))}
        return RawDataModel(sets=RawDataModel.Sets(**sets))

    def relabel_som(self, trainresult, trainset):
        print(f'{cf.bold}Relabelling SOM using {self.relabel} train set{cf.reset}')
        trainresult.framework.label_som(model=trainresult.model,
                                        trainset=trainset,
                                        validationset=trainresult.datamodel.sets.test,
                                        epochs=None, # label_som() will use 1 epoch
                                        batch_size=None, # label_som() will use model.som_batch_size
                                        optimizer=None, # No optimizer for label_som()
                                        dataaugmentations=trainresult.dataaugmentations,
                                        experimenttracking=trainresult.experimenttracking
                                        )

    def train_batch(self, trainresult, model_conf, batch):
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
        batchdatamodel = self.extract_session(session=batch, dataset=trainresult.datamodel.sets.train)
        print(f'{batch}: {len(batchdatamodel.sets.train.x)} vectors')

        # Apply new training parameters except for first batch
        if batch == self.first_batch:
            epochs = trainresult.model.som_epochs
        else:
            epochs = self.epochs
            print(f'{cf.bold}Applying new training parameters to DSOM{cf.reset}')
            if 'learning_rate' in self.dsom_params:
                print(f'{self.dsom_params["learning_rate"]=}')
                trainresult.model.som.som.learning_rate.copy_(trainresult.model.som.som.learning_rate.new_tensor(self.dsom_params['learning_rate']))
            if 'elasticity' in self.dsom_params:
                print(f'{self.dsom_params["elasticity"]=}')
                trainresult.model.som.som.elasticity_squared.copy_(trainresult.model.som.som.elasticity_squared.new_tensor(self.dsom_params['elasticity']).square())
        
        # Train unlabelled SOM model from feature model of dl model
        print(f'{cf.bold}Fine-tuning SOM on batch {batch}{cf.reset}')
        # Trainer cannot be re-used for now in PyTorch-Lightning
        trainer_dlsom = Trainer(max_epochs=epochs,
                                accelerator=framework.accelerator,
                                devices=framework.devices,
                                deterministic=True,
                                logger=trainresult.framework.logger(trainresult.experimenttracking,
                                name=f'{trainresult.name}_{trainresult.i}_finetune_{batch}'),
                                log_every_n_steps=1)
        trainer_module_dlsom = framework.TrainerModuleDLSOM(model=trainresult.model,
                                                            max_epochs=epochs,
                                                            optimizer=None,
                                                            dataaugmentations=trainresult.dataaugmentations)

        trainer_dlsom.fit(trainer_module_dlsom,
                            DataLoader(framework.DatasetFromArray(batchdatamodel.sets.train), batch_size=trainresult.model.som_batch_size, shuffle=True),
                            DataLoader(framework.DatasetFromArray(trainresult.datamodel.sets.test), batch_size=trainresult.model.som_batch_size) if len(trainresult.datamodel.sets.test.x) > 0 else None
                        )

        # Show differences after online learning
        new_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
        new_weights = trainresult.model.som.som.neurons.clone()
        print('Modified labels:', (new_labels != old_labels).sum())
        print('Modified neurons:', (new_weights != old_weights).any(-1).sum())

        print(f'{cf.bold}Evaluation on test set after fine-tuning on batch {batch}{cf.reset}')
        acc = framework.evaluate(trainresult.model,
                                 trainresult.datamodel.sets.test,
                                 batch_size=trainresult.batch_size,
                                 dataaugmentations=trainresult.dataaugmentations,
                                 name=f'{trainresult.name}_{trainresult.i}_finetune_{batch}_eval')
        self.log(trainresult.name, trainresult.i, 'fine_tuned', batch, acc)

        print(f'{cf.bold}Evaluation on full train set after fine-tuning on batch {batch}{cf.reset}')
        acc = framework.evaluate(trainresult.model,
                                 trainresult.datamodel.sets.train,
                                 batch_size=trainresult.batch_size,
                                 dataaugmentations=trainresult.dataaugmentations,
                                 name=f'{trainresult.name}_{trainresult.i}_finetune_{batch}_eval-fulltrain')
        self.log(trainresult.name, trainresult.i, 'fine_tuned-fulltrain', batch, acc)

        # Plot with T-SNE and save 2D representation and clusters to reuse for after relabelling
        neurons2d, clusters = self.plot_neurons(trainresult, batch, 'fine_tuned')

        if self.relabel: # Use original train set to relabel
            # Save original parameters before relabelling
            old_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
            old_weights = trainresult.model.som.som.neurons.clone()

            if self.relabel == 'original':
                self.relabel_som(trainresult, trainresult.datamodel.sets.train)
            elif self.relabel == 'batch':
                self.relabel_som(trainresult, batchdatamodel.sets.train)

            # Show differences after relabelling
            new_labels = trainresult.model.som.som_labelling.labels.argmax(-1).clone()
            new_weights = trainresult.model.som.som.neurons.clone()
            print('Modified labels:', (new_labels != old_labels).sum())
            print('Modified neurons:', (new_weights != old_weights).any(-1).sum())

            print(f'{cf.bold}Evaluation on test set after relabelling{cf.reset}')
            acc = framework.evaluate(trainresult.model,
                                     trainresult.datamodel.sets.test,
                                     batch_size=trainresult.batch_size,
                                     dataaugmentations=trainresult.dataaugmentations,
                                     name=f'{trainresult.name}_{trainresult.i}_relabel_{self.relabel}_{batch}')
            self.log(trainresult.name, trainresult.i, f'relabelled_{self.relabel}', batch, acc)


            print(f'{cf.bold}Evaluation on full train set after relabelling{cf.reset}')
            acc = framework.evaluate(trainresult.model,
                                     trainresult.datamodel.sets.train,
                                     batch_size=trainresult.batch_size,
                                     dataaugmentations=trainresult.dataaugmentations,
                                     name=f'{trainresult.name}_{trainresult.i}_relabel_{self.relabel}-fulltrain_{batch}')
            self.log(trainresult.name, trainresult.i, 'relabelled_{self.relabel}-fulltrain', batch, acc)

            # Plot with T-SNE
            self.plot_neurons(trainresult, batch, f'relabel_{self.relabel}', neurons2d, clusters, modified_labels=new_labels != old_labels)

        return trainresult

    def __call__(self, trainresult, model_conf):
        if self.shuffle_batches:
            batches = random.sample(self.batches, len(self.batches))
        else:
            batches = self.batches
        if self.first_batch:
            batches = [self.first_batch] + batches
        print(f'Batches: {batches}')

        # Plot with T-SNE
        self.plot_neurons(trainresult, 'original', '')

        for batch in batches:
            self.train_batch(trainresult, model_conf, batch)
        return trainresult
