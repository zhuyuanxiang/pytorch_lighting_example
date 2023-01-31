"""
=================================================
@path   : pytorch_lighting_example -> config
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/4 13:18
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

import matplotlib
import matplotlib_inline
import pytorch_lightning as pl
import seaborn as sns
import torch.backends.cudnn
from matplotlib import pyplot as plt

plt.set_cmap("cividis")
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # for export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

NUM_CATEGORIES = 10
SEQUENCE_LEN = 16
MAX_EPOCHS = 10

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# DATASET_PATH = os.environ.get("PATH_DATASETS", "datasets/")
# Path to the folder where the pretrained torch_model are saved
# CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/Transformers/")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
