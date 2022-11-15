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

from hydra.core.config_store import ConfigStore


@dataclass
class Dataset:
    path: str = 'datasets'
    batch_size: int = 32
    num_workers: int = 0  # MNIST 无法并行读取数据，会报错“Failed to load image Python extension:”
    pass


@dataclass
class MNISTDataset(Dataset):
    pass


@dataclass
class AnomalyDataset(Dataset):
    set_size: int = 10
    pass


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
            group="datasets",
            name="mnist",
            node=MNISTDataset,
            )
    cs.store(
            group="datasets",
            name="anomaly",
            node=AnomalyDataset,
            )
    pass
