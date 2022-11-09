"""
=================================================
@path   : pytorch_lighting_example -> mnist_model
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/10/24 17:29
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from data_module.mnist import MNISTDataModule
from lit_model.mnist import LitMNISTModel
from torch_model.mnist import MNISTModule


@hydra.main(version_base=None, config_path='../config/train', config_name="mnist")
def lit_mnist_main(config):
    data_module = MNISTDataModule(
            batch_size=config.mnist_dataset.batch_size,
            num_workers=config.mnist_dataset.num_workers
            )
    dims = (
            config.lit_classifier.in_channels,
            config.lit_classifier.in_height,
            config.lit_classifier.in_width,
            config.lit_classifier.out_channels
            )
    model = MNISTModule(dims)
    lit_model = LitMNISTModel(model, learning_rate=config.lit_classifier.learning_rate)
    trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=config.trainer.max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            # logger=CSVLogger(save_dir=config.trainer.default_root_dir,),
            default_root_dir=config.trainer.default_root_dir,
            )
    trainer.fit(lit_model, data_module)
    trainer.test(lit_model, data_module)
    pass


if __name__ == '__main__':
    lit_mnist_main()
