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
from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from hydra_conf.datasets import DatasetConfig
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitConfig
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerConfig
from lit_data_module.mnist import MNISTDataModule
from lit_model.mnist import LitMNISTModel
from parameters import HYDRA_PATH


@dataclass
class LitMNIST(LitConfig):
    checkpoint_path: str = 'saved_models/MNIST/'
    pass


@dataclass
class TrainerMNIST(TrainerConfig):
    max_epochs: int = 52
    pass


@dataclass
class ConfigMNIST(Config):
    trainer: TrainerConfig = TrainerMNIST
    dataset: DatasetConfig = MNISTDataset
    lit_mnist: LitConfig = LitMNIST
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigMNIST)


@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name="train_model")
def lit_mnist_main(config: ConfigMNIST):
    data_module = MNISTDataModule(
            data_dir=config.dataset.path,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers
            )
    dims = (
            config.lit_mnist.in_channels,
            config.lit_mnist.in_height,
            config.lit_mnist.in_width,
            config.lit_mnist.num_categories
            )
    lit_model = LitMNISTModel(dims, learning_rate=config.lit_mnist.learning_rate)
    trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=config.trainer.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            # logger=CSVLogger(save_dir=config.train.default_root_dir,),
            default_root_dir=config.trainer.default_root_dir,
            )
    trainer.fit(lit_model, data_module)
    trainer.test(lit_model, data_module)
    pass


if __name__ == '__main__':
    lit_mnist_main()
