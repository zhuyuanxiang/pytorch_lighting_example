"""
=================================================
@path   : pytorch_lighting_example -> mnist_model
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/10/24 17:29
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.cli import LRSchedulerTypeUnion
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import CSVLogger
from torch.nn import functional as func
from torch.optim import Optimizer
from torchmetrics import Accuracy

from data_module.mnist_data_module import MNISTDataModule
from models.mnist_module import MNISTModule


class LitMNIST(LightningModule):
    def __init__(self,
                 hidden_size=64,
                 num_classes=10,
                 dims=(1, 28, 28),
                 learning_rate=2e-4,
                 ):
        super().__init__()
        # Set our init args as class attributes
        self.save_hyperparameters()

        # Hardcode some dataset specific attributes
        channels, width, height = dims
        self.model = MNISTModule(channels, height, width, hidden_size, num_classes)
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        pass

    def forward(self, x):
        x = self.model(x)
        return func.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.nll_loss(logits, y)
        # self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.nll_loss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        self.val_accuracy.update(prediction, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = func.nll_loss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        self.test_accuracy.update(prediction, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

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
        super(LitMNIST, self).optimizer_step(
                epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure
                )
        # self.log("learning rate", optimizer.param_groups[0]['lr'], prog_bar=True)
        pass

    def lr_scheduler_step(
            self,
            scheduler: LRSchedulerTypeUnion,
            optimizer_idx: int,
            metric: Optional[Any],
            ) -> None:
        super(LitMNIST, self).lr_scheduler_step(scheduler, optimizer_idx, metric)
        pass


def lit_mnist_main():
    data_module = MNISTDataModule()
    model = LitMNIST()
    trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=20,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            logger=CSVLogger(save_dir='logs/'),
            )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    metrics.set_index("epoch", inplace=True)
    print(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")
    plt.show()
    pass


if __name__ == '__main__':
    lit_mnist_main()
