from __future__ import annotations

import sys
import typing

from qualia_core.typing import ModelConfigDict, ModelParamsConfigDict

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class DSOMLayerParamsConfigDict(TypedDict):
    learning_rate: float
    elasticity: float

class KSOMLayerParamsConfigDict(TypedDict):
    learning_rate: list[float]
    neighbourhood_width: list[float]

class SOMLayerConfigDict(TypedDict):
    kind: str
    params: typing.Union[DSOMLayerParamsConfigDict, KSOMLayerParamsConfigDict]  # noqa: UP007

class SOMModelParamsConfigDict(ModelParamsConfigDict):
    som_layer: SOMLayerConfigDict
    neurons: list[int]
    label_sigma: float

class SOMModelConfigDict(ModelConfigDict):
    params: SOMModelParamsConfigDict
