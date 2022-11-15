"""
=================================================
@path   : pytorch_lighting_example -> train
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

from hydra.core.config_store import ConfigStore


@dataclass
class Trainer:
    default_root_dir: str = "logs"
    max_epochs: int = 1
    pass


# cs = ConfigStore.instance()
# # Registering the Config class with the name 'config'.
# cs.store(name="trainer", node=Trainer)
