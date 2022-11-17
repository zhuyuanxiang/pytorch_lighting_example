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

from omegaconf import MISSING

from hydra_conf.optimizer_config import Optimizers
from hydra_conf.optimizer_config import OptimizersTransformer


@dataclass
class LitModule:
    input_channels: int = 1
    num_categories: int = 10
    input_height: int = 28
    input_width: int = 28
    hidden_dim: int = 128
    optimizer: Optimizers = MISSING
    checkpoint_path: str = MISSING  # Path to the folder where the pretrained torch_model are saved
    pass


@dataclass
class LitTransformer(LitModule):
    checkpoint_path: str = 'saved_models/Transformer/'
    optimizer: OptimizersTransformer = OptimizersTransformer
    pass


@dataclass
class LitTransformerReverse(LitTransformer):
    model_channels: int = 32
    num_heads: int = 1
    num_layers: int = 1
    dropout: float = 0.0
    input_dropout: float = 0.0
    pass


@dataclass
class LitClassifier(LitModule):
    checkpoint_path: str = 'saved_models/Classifier/'
    pass


@dataclass
class LitMNIST(LitModule):
    checkpoint_path: str = 'saved_models/MNIST/'
    pass


@dataclass
class LitGAN(LitModule):
    checkpoint_path: str = 'saved_models/MNIST/'
    pass
