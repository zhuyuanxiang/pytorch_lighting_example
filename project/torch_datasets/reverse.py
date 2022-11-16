"""
=================================================
@path   : pytorch_lighting_example -> reverse_data_module
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 15:57
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   : 随机生成的反转数据集
            Input data: tensor([9, 8, 4, 3, 3, 4, 4, 8, 8, 7, 2, 8, 0, 9, 5, 0])
            Labels:     tensor([0, 5, 9, 0, 8, 2, 7, 8, 8, 4, 4, 3, 3, 4, 8, 9])
@History:
@Plan   :
==================================================
"""
# OS
from datetime import datetime
from functools import partial

# PyTorch
import torch
from torch.utils import data as data


class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, sequence_len, data_size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = sequence_len
        self.size = data_size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_data = self.data[idx]
        labels = torch.flip(input_data, dims=(0,))
        return input_data, labels


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    dataset = partial(ReverseDataset, 10, 16)  # partial() 函数允许你给一个或多个参数设置固定的值，减少接下来被调用时的参数个数。
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(dataset(1000), batch_size=128)
    test_loader = data.DataLoader(dataset(10000), batch_size=128)
    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
