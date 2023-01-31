"""
=================================================
@path   : pytorch_lighting_example -> argument_parser
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/15 10:38
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from argparse import ArgumentParser

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from parameters import HYDRA_PATH

DATASET_NAME = 'mnist_dataset'
MODULE_NAME = 'lit_mnist'
TRAINER_NAME = 'train'

DictConfig


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name="mnist")
def main(config):
    parser = ArgumentParser(
            prog='ProgramName',
            description='What the program does',
            epilog='Text at the bottom of help',
            )
    parser = pl.Trainer.add_argparse_args(parser)
    if MODULE_NAME in config:
        for j in config[MODULE_NAME]:
            value = config[MODULE_NAME][j]
            if j in parser.parse_args():
                parser.set_defaults(j=value)
            else:
                parser.add_argument('--' + j, type=type(value), default=value)
    if TRAINER_NAME is config:
        for j in config[TRAINER_NAME]:
            value = config[TRAINER_NAME][j]
            if j in parser.parse_args():
                parser.set_defaults(j=value)
            else:
                parser.add_argument('--' + j, type=type(value), default=value)
    args = parser.parse_args()
    print(args)
    trainer = pl.Trainer.from_argparse_args(args)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main()
