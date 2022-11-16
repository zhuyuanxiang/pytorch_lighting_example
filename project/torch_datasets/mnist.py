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
import os

import hydra
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

config_path = os.path.join(os.getcwd(), 'config', 'train')


def mnist(batch_size=32, root=''):
    dataset = MNIST(root, train=True, download=False, transform=transforms.ToTensor())
    mnist_test = MNIST(root, train=False, download=False, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size)
    val_loader = DataLoader(mnist_val, batch_size)
    test_loader = DataLoader(mnist_test, batch_size)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=config_path, config_name='mnist')
def main(config):
    train_loader, val_loader, test_loader = mnist(
            config.mnist_dataset.batch_size,
            config.datasets_path
            )
    print("train_loader=", train_loader)
    print("val_loader=", val_loader)
    print("test_loader=", test_loader)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
