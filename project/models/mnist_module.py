"""
=================================================
@path   : pytorch_lighting_example -> mnist_module
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

from torch import nn


class MNISTModule(nn.Module):
    def __init__(self, dims, hidden_size=64, ):
        super(MNISTModule, self).__init__()
        in_channels, in_height, in_width, out_channels = dims
        # Define PyTorch model
        self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * in_width * in_height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, out_channels),
                )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
