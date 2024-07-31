from __future__ import annotations

import logging
import sys

from qualia_core.learningframework import PyTorch
from qualia_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # noqa: TCH002
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT  # noqa: TCH002
    from qualia_core.learningframework.PyTorch import CheckpointMetricConfigDict  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class PyTorchUnlabelledSOM(PyTorch):
    import qualia_plugin_som.learningmodel.pytorch as learningmodels
    learningmodels.__dict__.update(PyTorch.learningmodels.__dict__) # Merge core models back. Warning: module name changes too!

    class TrainerModule(PyTorch.TrainerModule):
        @override
        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor | None:
            x, y = batch
            bmu_indices, bmu_vectors = self(x, current_epoch=self.current_epoch, max_epochs=self.trainer.max_epochs)
            self.train_metrics(bmu_vectors, x.reshape(x.shape[0], -1))
            self.log_dict(self.train_metrics, prog_bar=True)
            return None # No loss

        @override
        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            bmu_indices, bmu_vectors = self(x)
            self.val_metrics(bmu_vectors, x.reshape(x.shape[0], -1))
            self.log_dict(self.val_metrics, prog_bar=True)

        @override
        def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
            x, y = batch
            bmu_indices, bmu_vectors = self(x)
            self.test_metrics(bmu_vectors, x.reshape(x.shape[0], -1))
            self.log_dict(self.test_metrics, prog_bar=True)

        @override
        def forward(self, x: torch.Tensor,
                    current_epoch: int | None = None,
                    max_epochs: int | None = None) -> torch.Tensor:
            return self.model(x, current_epoch=current_epoch, max_epochs=max_epochs)

    def __init__(self,  # noqa: PLR0913
                 use_best_epoch: bool = False,  # noqa: FBT001, FBT002
                 enable_progress_bar: bool = True,  # noqa: FBT001, FBT002
                 progress_bar_refresh_rate: int = 1,
                 accelerator: str = 'auto',
                 devices: int | str | list[int] = 'auto',
                 precision: _PRECISION_INPUT = 32,
                 metrics: list[str] | None = None,
                 checkpoint_metric: CheckpointMetricConfigDict | None = None) -> None:
        super().__init__(use_best_epoch=use_best_epoch,
                       enable_progress_bar=enable_progress_bar,
                       progress_bar_refresh_rate=progress_bar_refresh_rate,
                       accelerator=accelerator,
                       devices=devices,
                       precision=precision,
                       metrics=metrics,
                       loss=None,
                       enable_confusion_matrix=False,
                       checkpoint_metric=checkpoint_metric)
