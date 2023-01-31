"""
=================================================
@path   : pytorch_lighting_example -> tools
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 15:57
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import math
from datetime import datetime

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.nn import functional as func
import pytorch_lightning as pl

# ----------------------------------------------------------------------
def visualize_examples(_indices, orig_dataset):
    images = [orig_dataset[idx][0] for idx in _indices.reshape(-1)]
    images = torch.stack(images, dim=0)
    images = images * TORCH_DATA_STD + TORCH_DATA_MEANS

    img_grid = torchvision.utils.make_grid(images, nrow=SET_SIZE, normalize=True, pad_value=0.5, padding=16)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(12, 8))
    plt.title("Anomaly examples on CIFAR100")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()


def visualize_prediction(idx, _indices, _predictions, _attention_maps):
    visualize_examples(_indices[idx: idx + 1], test_set)
    print("Prediction:", _predictions[idx].item())
    plot_attention_maps(input_data=None, attn_maps=_attention_maps, idx=idx)
    pass


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin="lower", vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def scaled_dot_product(query, key, value, mask=None):
    """缩放点积注意力

    Args:
        query (_type_): 查询
        key (_type_): 键
        value (_type_): 值
        mask (_type_, optional): 掩码. Defaults to None.

    Returns:
        values_out (): 值
        attention_out (): 注意力
    """
    d_key = query.size()[-1]
    attn_logits = torch.matmul(query, key.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_key)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention_out = func.softmax(attn_logits, dim=-1)
    values_out = torch.matmul(attention_out, value)
    return values_out, attention_out


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    test_scaled_dot_product()
    pass


def test_scaled_dot_product():
    seq_len, d_k = 3, 2
    pl.seed_everything(42)
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print("-->Q:\n", q)
    print("-->K:\n", k)
    print("-->V:\n", v)
    print("-->Values:\n", values)
    print("-->Attention:\n", attention)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
