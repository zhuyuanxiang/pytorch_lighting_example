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


class MyTestCase(unittest.TestCase):
    def test_with_initialize(self) -> None:
        """
        1. initialize will add config_path the config search path within the context
        2. The module with your configs should be importable. it needs to have a __init__.py (can be empty).
        3. The config path is relative to the file calling initialize (this file)

        :return:
        """
        with initialize(version_base=None, config_path='../config/train'):
            config = compose(config_name='classifier')
            self.assertEqual(
                    config, {'seed': 1234,
                             'lit_classifier': {'hidden_dim': 128, 'learning_rate': 0.0001},
                             'trainer': {'default_root_dir': 'logs', 'max_epochs': 1},
                             'mnist_dataset': {'batch_size': 32}
                             }
                    )

    def test_with_initialize_config_module(self) -> None:
        """
        1. initialize_with_module will add the config module to the config search path within the context
        2. The module with your configs should be importable. it needs to have a __init__.py (can be empty).
        3. The module should be absolute
        4. This approach is not sensitive to the location of this file, the test can be relocated freely.

        :return:
        """
        # ToDo: config_module 无法完成
        with initialize_config_module(version_base=None, config_module='config.train'):
            # config is relative to a module
            config = compose(config_name='classifier', overrides=["app.user=test_user"])
            self.assertEqual(
                    config, {'seed': 1234,
                             'lit_classifier': {'hidden_dim': 128, 'learning_rate': 0.0001},
                             'trainer': {'default_root_dir': 'logs', 'max_epochs': 1},
                             'mnist_dataset': {'batch_size': 32}
                             }
                    )


if __name__ == '__main__':
    unittest.main()
