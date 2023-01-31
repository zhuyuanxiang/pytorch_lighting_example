"""
=================================================
@path   : pytorch_lighting_example -> transformer
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/11 15:39
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import torch
from torch import nn

from torch_model.transformer.encoder import TransformerEncoder
from torch_model.transformer.encoding import PositionalEncoding


class TransformerModule(nn.Module):
    def __init__(self,
                 input_channels:int,
                 model_channels:int,
                 num_classes:int,
                 num_heads:int,
                 num_layers:int,
                 dropout:float=0.0,
                 input_dropout:float=0.0,
                 ):
        """_summary_

        Args:
            input_channels (int): 整个模型的输入维度
            model_channels (int): 编码器的输入维度
            num_classes (int): 序列中预测元素的类别
            num_heads (int): 多头注意力模块中头的个数
            num_layers (int): 编码模型的层数
            dropout (float, optional): 模型中 Dropout 的概率. Defaults to 0.0.
            input_dropout (float, optional): 输入网络中 Dropout 的概率. Defaults to 0.0.
        """
        super().__init__()
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(input_channels, model_channels)
                )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_channels)

        # Transformer
        self.transformer = TransformerEncoder(
                num_layers=num_layers,
                input_channels=model_channels,
                dim_feedforward=2 * model_channels,
                num_heads=num_heads,
                dropout=dropout,
                )

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
                nn.Linear(model_channels, model_channels),
                nn.LayerNorm(model_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(model_channels, num_classes),
                )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_channels]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps
