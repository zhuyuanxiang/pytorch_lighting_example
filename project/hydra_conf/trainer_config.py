"""
=================================================
@path   : pytorch_lighting_example -> trainer_config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 14:29
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


@dataclass
class TrainerConfig:
    default_root_dir: str = "logs"
    max_epochs: int = MISSING
    pass


@dataclass
class TrainerTransformer(TrainerConfig):
    max_epochs: int = 53
    pass


@dataclass
class TrainerClassifier(TrainerConfig):
    max_epochs: int = 51
    pass


@dataclass
class TrainerMNIST(TrainerConfig):
    max_epochs: int = 52
    pass


@dataclass
class TrainerGAN(TrainerConfig):
    max_epochs: int = 52
    pass
