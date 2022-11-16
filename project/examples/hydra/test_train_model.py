"""
=================================================
@path   : pytorch_lighting_example -> test_train_model
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/16 11:11
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
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf import OmegaConf

from hydra_conf.datasets import AnomalyDataset
from hydra_conf.datasets import Dataset
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitConfig
from hydra_conf.lit_config import LitMNIST
from hydra_conf.lit_config import LitTransformer
from hydra_conf.trainer import Trainer
from hydra_conf.trainer import TrainerMNIST
from hydra_conf.trainer import TrainerTransformer


@dataclass
class Config:
    trainer: Trainer = MISSING
    dataset: Dataset = MISSING
    seed: int = 1234
    debug: bool = False
    pass


@dataclass
class ConfigTransformer(Config):
    trainer: Trainer = TrainerTransformer
    dataset: Dataset = AnomalyDataset
    lit_transformer: LitConfig = LitTransformer()
    pass


@dataclass
class ConfigMNIST(Config):
    trainer: Trainer = TrainerMNIST
    dataset: Dataset = MNISTDataset
    lit_mnist: LitConfig = LitMNIST()
    pass


cs = ConfigStore.instance()
# cs.store(name='base_config', node=Config)
# cs.store(name="base_config", node=ConfigTransformer)
cs.store(name="base_config", node=ConfigMNIST)


@hydra.main(version_base=None, config_path='conf', config_name="train_model")
def my_app(cfg: ConfigMNIST) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
