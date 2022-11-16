"""
=================================================
@path   : pytorch_lighting_example -> lit_image_classifier
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/10/24 17:29
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
ToDo: 完成从config中更新到args中的过程，实现用参数生成Trainer()
==================================================
"""

import argparse
from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from hydra_conf.datasets import DatasetConfig
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitConfig
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerConfig
from lit_model.classifier import LitModuleBackboneClassifier
from parameters import HYDRA_PATH

TRAINER_NAME = 'train'


@dataclass
class LitClassifier(LitConfig):
    checkpoint_path: str = 'saved_models/Classifier/'
    pass


@dataclass
class TrainerClassifier(TrainerConfig):
    max_epochs: int = 1
    pass


@dataclass
class ConfigClassifier(Config):
    trainer: TrainerConfig = TrainerClassifier
    dataset: DatasetConfig = MNISTDataset
    lit_classifier: LitConfig = LitClassifier
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigClassifier)


@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name='train_model')
def cli_main(config:ConfigClassifier):
    parser = argparse.ArgumentParser(
            prog='train_mnist_classifier',
            description='pytorch-lightning mnist classifier example',
            epilog='contact with zhuyuanxiang@gmail.com'
            )
    parser = pl.Trainer.add_argparse_args(parser)
    if TRAINER_NAME is config:
        for flag in config[TRAINER_NAME]:
            value = config[TRAINER_NAME][flag]
            if flag in parser.parse_args():
                parser.set_defaults(flag=value)
            else:
                parser.add_argument('--' + flag, type=type(value), default=value)

    pl.seed_everything(config.seed)

    # ------------
    # datasets
    # ------------
    dataset = MNIST(root=config.dataset.path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(root=config.dataset.path, train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=config.dataset.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=config.dataset.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=config.dataset.batch_size)

    # ------------
    # model
    # ------------
    model = LitModuleBackboneClassifier(config.lit_classifier.hidden_dim, config.lit_classifier.learning_rate)

    # ------------
    # training
    # ------------
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
