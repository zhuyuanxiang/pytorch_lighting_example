"""
=================================================
@path   : pytorch_lighting_example -> lit_gan
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/10/27 17:12
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data_module.mnist import MNISTDataModule
from lit_model.gan import GAN


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path='../config/train', config_name='gan')
def lit_mnist_main(config):
    pl.seed_everything(config.seed)
    dm = MNISTDataModule(
            batch_size=config.mnist_dataset.batch_size,
            num_workers=config.mnist_dataset.num_workers,
            )
    model = GAN(
            dims=(
                    config.lit_classifier.in_channels,
                    config.lit_classifier.in_height,
                    config.lit_classifier.in_width
                    )
            )
    trainer = Trainer(
            default_root_dir=config.trainer.default_root_dir,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=config.trainer.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=10)],
            )
    trainer.fit(model, dm)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    lit_mnist_main()
