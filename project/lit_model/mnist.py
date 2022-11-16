"""
=================================================
@path   : pytorch_lighting_example -> mnist
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/9 11:05
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from argparse import ArgumentParser
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerTypeUnion
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from torch_model.mnist import MNISTModule


class LitMNISTModel(LightningModule):
    def __init__(self, dims, learning_rate=1e-4, **kwargs):
        super().__init__(**kwargs)
        # Set our init args as class attributes
        self.save_hyperparameters()
        self.model = MNISTModule(dims)

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        pass

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        self.val_accuracy.update(prediction, y)
        accu = accuracy(prediction, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc1", self.val_accuracy, prog_bar=True)
        self.log("val_acc2", accu, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        self.test_accuracy.update(prediction, y)
        accu = accuracy(prediction, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc1", self.test_accuracy, prog_bar=True)
        self.log("test_acc2", accu, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_idx: int = 0,
            optimizer_closure: Optional[Callable[[], Any]] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
            ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure)
        pass

    def lr_scheduler_step(
            self,
            scheduler: LRSchedulerTypeUnion,
            optimizer_idx: int,
            metric: Optional[Any],
            ) -> None:
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)
        pass
