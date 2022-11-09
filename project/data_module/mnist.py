"""
=================================================
@path   : pytorch_lighting_example -> mnist_data_module
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/4 19:08
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

from datetime import datetime

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from config import BATCH_SIZE
from config import NUM_WORKERS
from config import PATH_DATASETS


def mnist(batch_size=32):
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    val_loader = DataLoader(mnist_val, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class MNISTDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = PATH_DATASETS,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
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
        return DataLoader(self.mnist_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
