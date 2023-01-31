"""
=================================================
@path   : pytorch_lighting_example -> encoder_block
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 16:05
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

from datetime import datetime

from torch import nn as nn

from torch_model.transformer.multi_head_attention import MultiheadAttention


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_channels: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_channels, input_channels, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
                nn.Linear(input_channels, dim_feedforward),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(dim_feedforward, input_channels),
                )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
