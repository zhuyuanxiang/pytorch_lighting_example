"""
=================================================
@path   : pytorch_lighting_example -> datasets
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 14:55
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
class DatasetConfig:
    path: str = 'datasets'  # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    batch_size: int = 32  # 数据集每个批次加载数据的个数
    num_workers: int = 0  # MNIST 无法并行读取数据，会报错“Failed to load image Python extension:”
    pass


@dataclass
class MNISTDataset(DatasetConfig):
    pass


@dataclass
class AnomalyDataset(DatasetConfig):
    set_size: int = 10
    pass


@dataclass
class ReverseDataset(DatasetConfig):
    num_categories: int = 10
    sequence_len: int = 16

    pass
