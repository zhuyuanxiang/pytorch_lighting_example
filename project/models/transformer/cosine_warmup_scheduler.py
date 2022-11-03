"""
=================================================
@path   : pytorch_lighting_example -> cosine_warmup_scheduler
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 16:20
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.nn.parameter import Parameter
from torch.optim import SGD
import seaborn as sns


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizers, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizers)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def generate_xy(_optimizer, _scheduler):
    x, y = [], []
    epoch = 0
    _optimizer.zero_grad()
    for epoch in range(0, 23):
        _optimizer.step()
        x.append(epoch)
        y.append(_optimizer.param_groups[0]['lr'])
        _scheduler.step()
        pass
    x.append(epoch + 1)
    y.append(_optimizer.param_groups[0]['lr'])
    return x, y


def plot_xy(x, y):
    plt.plot(x, y)
    plt.show()
    # from pprint import pprint
    # pprint(x, compact=True)
    # pprint(y, compact=True)


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, lr=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup=True, max_iters=5)
    plt.plot(*generate_xy(optimizer, scheduler))

    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(optimizers=optimizer, warmup=100, max_iters=2000)

    # Plotting
    epochs = list(range(2000))
    sns.reset_orig()
    sns.set()
    plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
