"""
=================================================
@path   : pytorch_lighting_example -> test_hydra
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/8 15:15
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
from datetime import datetime

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf import OmegaConf

from toolbox import get_hydra_path

HYDRA_PATH = get_hydra_path(os.path.dirname(__file__))


@dataclass(frozen=True)
class SerialPort:
    baud_rate: int = 19200
    data_bits: int = 8
    stop_bits: int = 1


cs = ConfigStore.instance()
cs.store(name="serialport", node=SerialPort)
print(cs)


@hydra.main(version_base=None, config_name="serialport")
def test_frozen_config(config: SerialPort) -> None:
    # 不允许修改类中的参数
    # config.baud_rate=100
    print(config)


# @hydra.main(version_base=None, config_path=HYDRA_PATH, config_name='tmp')
@hydra.main(config_path=HYDRA_PATH, config_name='tmp')
def test_something(config: DictConfig):
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    print(f"to_absolute_path('config')   : {hydra.utils.to_absolute_path('config')}")
    print(f"to_absolute_path('/config/train')  : {hydra.utils.to_absolute_path('/config/train')}")
    assert isinstance(config.group1.float1, float)
    assert config.group1.float1 == 0.0001
    assert config['group1']['float1'] == 0.0001

    assert isinstance(config.group2.str2, str)
    assert config.group2.str2 == 'logs'

    assert isinstance(config.group1.int1, int)
    assert config.group1.int1 == 1

    print("config.group1.int1=", config.group1.int1)
    print("config.group1.float1=", config.group1.float1)
    print("config.group2.list2=", config.group2.list2)
    print("config.group2.str2=", config.group2.str2)
    # print(hydra.utils.HydraConfig.get_state())
    pass


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'config'),config_name='train')
def test_group(config):
    print(OmegaConf.to_yaml(config))
    pass


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    test_something()
    # test_frozen_config()
    # test_group()
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("FILE_PATH=", os.path.dirname(__file__))
    print("HYDRA_PATH=", HYDRA_PATH)
    __author__ = 'zYx.Tom'
    main(__author__)
