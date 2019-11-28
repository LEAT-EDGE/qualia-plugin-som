from pathlib import Path
from typing import Optional

import qualia_core
from qualia_codegen_core.graph import ModelGraph
from qualia_codegen_plugin_som import Converter
from qualia_codegen_plugin_som.graph import TorchModelGraph


class QualiaCodeGen(qualia_core.postprocessing.QualiaCodeGen):
    """QualiaCodeGen converter calling Qualia-CodeGen-Plugin-SOM to handle SOM layers."""

    def convert_model_to_modelgraph(self, model) -> Optional[ModelGraph]:
        return TorchModelGraph(model).convert()

    def convert_modelgraph_to_c(self, modelgraph: ModelGraph) -> Optional[str]:
        converter = Converter(output_path=Path('out')/'qualia_codegen'/self._name)
        return converter.convert_model(modelgraph)
