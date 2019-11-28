from pathlib import Path

from qualia_core.postprocessing import PostProcessing

from .DLSOMCORe50FineTuning import DLSOMCORe50FineTuning


class PlotSOMNeurons(PostProcessing):
    def __init__(self):
        self.d = DLSOMCORe50FineTuning(0)

    def __call__(self, trainresult, model_conf):
        label_ratio = trainresult.framework.label_ratio
        self.d.plot_neurons(trainresult, batch=f'{label_ratio=}', step='', outdir=Path('out')/'PlotSOMNeurons')
        return trainresult
