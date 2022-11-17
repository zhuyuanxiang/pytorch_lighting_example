"""
=================================================
@path   : pytorch_lighting_example -> optimizer_config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/17 9:43
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
class Optimizers:
    max_iters: int = 1
    warmup: int = 50
    learning_rate: float = 1e-4
    pass


@dataclass
class OptimizersTransformer(Optimizers):
    learning_rate: float = 5e-4
    pass
