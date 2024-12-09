from __future__ import annotations

import sys

import qualia_core.postprocessing.QualiaCodeGen
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

    from qualia_codegen_core.graph import ModelGraph  # noqa: TCH002
    from torch import nn  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class QualiaCodeGen(qualia_core.postprocessing.QualiaCodeGen):
    """QualiaCodeGen converter calling Qualia-CodeGen-Plugin-SOM to handle SOM layers."""

    @override
    def convert_model_to_modelgraph(self, model: nn.Module) -> ModelGraph | None:
        from qualia_codegen_plugin_som.graph import TorchModelGraph
        return TorchModelGraph(model).convert()

    @override
    def convert_modelgraph_to_c(self, modelgraph: ModelGraph, output_path: Path) -> str | None:
        from qualia_codegen_plugin_som import Converter
        converter = Converter(output_path=output_path)
        return converter.convert_model(modelgraph)
