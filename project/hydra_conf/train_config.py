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
from dataclasses import dataclass

from omegaconf import MISSING

from hydra_conf.datasets import DatasetConfig
from hydra_conf.lit_config import LitModule
from hydra_conf.trainer_config import TrainerConfig


@dataclass
class Config:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    lit_module: LitModule = MISSING
    seed: int = 1234
    debug: bool = False
    pass
