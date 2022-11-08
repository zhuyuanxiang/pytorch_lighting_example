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


@hydra.main(version_base=None, config_path=os.getcwd() + '/config/train', config_name='classifier')
def test_something(config: DictConfig):
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    print(f"to_absolute_path('config')   : {hydra.utils.to_absolute_path('config/train')}")
    print(f"to_absolute_path('/config/train')  : {hydra.utils.to_absolute_path('/config/train')}")
    assert isinstance(config.lit_classifier.learning_rate, float)
    assert config['lit_classifier']['learning_rate'] == 0.0001
    assert config.lit_classifier.learning_rate == 0.0001

    assert config.trainer.default_root_dir == 'logs'
    assert isinstance(config.trainer.default_root_dir, str)

    print(hydra.utils.HydraConfig.get_state())
    pass


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    # test_something()
    test_frozen_config()
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
