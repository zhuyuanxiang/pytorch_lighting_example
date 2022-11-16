"""
=================================================
@path   : pytorch_lighting_example -> tools
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/9 11:20
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def show_metrics(logger_path: str):
    """读取CSVLogger输出的文件，并且展示出来

    :param logger_path:
    :return:
    """
    metrics = pd.read_csv(f"{logger_path}/metrics.csv")
    metrics.set_index("epoch", inplace=True)
    print(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")
    plt.show()
    pass
