"""
=================================================
@path   : pytorch_lighting_example -> train_config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 15:00
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
from hydra.conf import RunDir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from hydra_conf.datasets import AnomalyDataset
from hydra_conf.datasets import Dataset
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitConfig
from hydra_conf.trainer import Trainer


@dataclass
class RunHydraConfig(RunDir):
    dir: str = r'logs/hydra_logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'


@dataclass
class TrainHydraConfig(HydraConfig):
    output_subdir = None
    run = RunHydraConfig()


@dataclass
class TrainClassifier:
    seed: int = 1234
    dataset: Dataset = MNISTDataset()
    lit_config: LitConfig = LitConfig()
    trainer: Trainer = Trainer()
    pass


@dataclass
class TrainMNIST:
    seed: int = 1234
    dataset: Dataset = MNISTDataset()
    lit_config: LitConfig = LitConfig()
    trainer: Trainer = Trainer()
    pass


@dataclass
class TrainTransformer:
    # hydra = TrainHydraConfig()    # 这里设置不起作用
    seed: int = 1234
    dataset: Dataset = AnomalyDataset()
    lit_config: LitConfig = LitConfig()
    trainer: Trainer = Trainer()
    pass


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="classifier", node=TrainClassifier)
cs.store(name="mnist", node=TrainMNIST)
cs.store(name="transformer", node=TrainTransformer)
cs.store(name='test_hydra', node=TrainTransformer)


@hydra.main(version_base="1.2", config_path=os.path.join(os.getcwd(), 'config'), config_name="test_hydra")
def my_app(cfg: TrainTransformer) -> None:
    print(OmegaConf.to_yaml(cfg))
    # print(f"seed={cfg.seed}, dataset={cfg.dataset.batch_size}")
    # print(f"seed={cfg.seed}, trainer={cfg.trainer.default_root_dir}")
    # assert cfg.seed == 1234
    # print(cfg['seed'])
    pass


if __name__ == "__main__":
    my_app()
