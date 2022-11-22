"""
=================================================
@path   : pytorch_lighting_example -> train_gan
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

from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitGAN
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerGAN
from lit_data_module.mnist import MNISTDataModule
from lit_model.gan import GAN
from parameters import HYDRA_PATH


@dataclass
class ConfigGAN(Config):
    trainer: TrainerGAN = TrainerGAN
    dataset: MNISTDataset = MNISTDataset
    lit_module: LitGAN = LitGAN
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigGAN)


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name='train_model')
def lit_mnist_main(config: ConfigGAN):
    pl.seed_everything(config.seed)
    dm = MNISTDataModule(
            data_dir=config.dataset.path,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            )
    model = GAN(
            dims=(
                    config.lit_module.input_channels,
                    config.lit_module.input_height,
                    config.lit_module.input_width
                    )
            )
    trainer = Trainer(
            default_root_dir=config.trainer.default_root_dir,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=config.trainer.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=10)],
            )
    trainer.fit(model, dm)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    lit_mnist_main()
