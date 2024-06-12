from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import torch
from qualia_core.typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from qualia_plugin_som.learningmodel.pytorch.layers.SOMLabelling import SOMLabelling  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SOM(ABC, nn.Module):

    def __init__(self) -> None:
        self.call_super_init = True # Support multiple inheritance from nn.Module
        super().__init__()

    @abstractmethod
    @override
    def forward(self,
                input: torch.Tensor,  # noqa: A002
                current_epoch: int | None = None, # We always receive current_epoch even though we do not use it in DSOM
                max_epochs: int | None = None, # We always receive max_epochs even though we do not use it in DSOM
                targets: torch.Tensor | None = None, # Unused for unsupervised learning
                som_labelling: SOMLabelling | None = None, # Unused for unsupervised learning,
                return_position: bool = True,
                return_value: bool = True) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
