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
import os
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import ListConfig
from omegaconf import MISSING
from omegaconf import OmegaConf

from hydra_conf.datasets import AnomalyDataset
from hydra_conf.datasets import DatasetConfig
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitMNIST
from hydra_conf.lit_config import LitModule
from hydra_conf.lit_config import LitTransformer
from hydra_conf.trainer_config import TrainerConfig
from hydra_conf.trainer_config import TrainerMNIST
from hydra_conf.trainer_config import TrainerTransformer


@dataclass
class Config:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    lit_module: LitModule = MISSING
    seed: int = 1234
    debug: bool = False
    root: str = 'root'
    path: str = 'path'
    todo: list = ListConfig([0, 1, 2])
    tmp: list = ListConfig([0, 1, 2])
    pass


@dataclass
class ConfigTransformer(Config):
    trainer: TrainerTransformer = TrainerTransformer
    dataset: AnomalyDataset = AnomalyDataset
    lit_module: LitTransformer = LitTransformer()
    pass


@dataclass
class ConfigMNIST(Config):
    trainer: TrainerMNIST = TrainerMNIST
    dataset: MNISTDataset = MNISTDataset
    lit_module: LitMNIST = LitMNIST()
    file: str = os.path.join(os.getcwd(), Config.root, Config.path, 'file')
    # optimizer 没有设置，所以结果为：???
    pass


cs = ConfigStore.instance()
# cs.store(name='base_config', node=Config)
# cs.store(name="base_config", node=ConfigTransformer)
cs.store(name="base_config", node=ConfigMNIST)


@hydra.main(version_base=None, config_path='conf', config_name="train_model")
def my_app(cfg: ConfigMNIST) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg)


if __name__ == "__main__":
    my_app()
