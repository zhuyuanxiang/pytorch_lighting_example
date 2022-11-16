"""
=================================================
@path   : pytorch_lighting_example -> preposition
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/3 16:55
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

from datetime import datetime

import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
