"""
=================================================
@path   : pytorch_lighting_example -> config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/10/24 17:32
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

import os
from datetime import datetime

PROJECT_PATH = os.getcwd()
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
# NUM_WORKERS = int(os.cpu_count() / 2)
# NUM_WORKERS = 0  # MNIST 无法并行读取数据，会报错“Failed to load image Python extension:”


def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    print(__path__)
    print("__path__=", __path__)
    main(__author__)
