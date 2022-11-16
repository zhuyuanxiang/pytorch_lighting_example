"""
=================================================
@path   : pytorch_lighting_example -> test_hydra
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/8 14:37
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import os
import unittest

from hydra import compose
from hydra import initialize
from hydra import initialize_config_module
from hydra.core.config_store import ConfigStore

from hydra_conf.train_config import ConfigMNIST
from toolbox import get_hydra_path

HYDRA_PATH = get_hydra_path(os.path.dirname(__file__))
HYDRA_MODULE_PATH = 'config.train'

cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigMNIST)


class MyTestCase(unittest.TestCase):
    def test_hierarchy(self) -> None:
        with initialize(version_base='1.2', config_path='../config'):
            config = compose(config_name='train_model')
            print("config.trainer=",config.trainer)

    def test_with_initialize(self) -> None:
        """
        1. initialize will add config_path the config search path within the context
        2. The module with your configs should be importable. it needs to have a __init__.py (can be empty).
        3. The config path is relative to the file calling initialize (this file)

        :return:
        """
        with initialize(version_base=None, config_path=HYDRA_PATH):
            config = compose(config_name='classifier')
            self.assertEqual(
                    config, {'seed': 1234,
                             'datasets_path': 'datasets/',
                             'lit_classifier': {'in_channels': 1, 'out_channels': 10, 'in_height': 28, 'in_width': 28,
                                                'hidden_dim': 128, 'learning_rate': 0.0001},
                             'train': {'default_root_dir': 'logs', 'max_epochs': 1},
                             'mnist_dataset': {'batch_size': 32, 'num_workers': 0}}
                    )

    def test_with_initialize_config_module(self) -> None:
        """
        1. initialize_with_module will add the config module to the config search path within the context
        2. The module with your configs should be importable. it needs to have a __init__.py (can be empty).
        3. The module should be absolute
        4. This approach is not sensitive to the location of this file, the test can be relocated freely.

        :return:
        """
        with initialize_config_module(version_base=None, config_module=HYDRA_MODULE_PATH):
            # config is relative to a module
            config = compose(config_name='classifier')
            self.assertEqual(
                    config, {'seed': 1234,
                             'datasets_path': 'datasets/',
                             'lit_classifier': {'in_channels': 1, 'out_channels': 10, 'in_height': 28, 'in_width': 28,
                                                'hidden_dim': 128, 'learning_rate': 0.0001},
                             'train': {'default_root_dir': 'logs', 'max_epochs': 1},
                             'mnist_dataset': {'batch_size': 32, 'num_workers': 0}}
                    )


if __name__ == '__main__':
    unittest.main()
