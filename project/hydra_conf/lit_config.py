"""
=================================================
@path   : pytorch_lighting_example -> lit_config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 14:56
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from dataclasses import dataclass


@dataclass
class LitConfig:
    in_channels: int = 1
    num_categories: int = 10
    in_height: int = 28
    in_width: int = 28
    hidden_dim: int = 128
    learning_rate: float = 0.0001
    pass
