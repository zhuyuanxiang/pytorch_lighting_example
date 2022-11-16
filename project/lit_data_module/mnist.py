"""
=================================================
@path   : pytorch_lighting_example -> mnist
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/11 16:41
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = '.',
                 batch_size: int = 256,
                 num_workers: int = 0,
                 ) -> None:
        super(MNISTDataModule, self).__init__()
        self.save_hyperparameters()
        self.mnist_train, self.mnist_val, self.mnist_test = None, None, None
        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,)),
                 ]
                )

    def prepare_data(self):
        # download
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
                dataset=self.mnist_train, shuffle=True,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers
                )

    def val_dataloader(self):
        return DataLoader(
                dataset=self.mnist_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers
                )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.mnist_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers
                )
