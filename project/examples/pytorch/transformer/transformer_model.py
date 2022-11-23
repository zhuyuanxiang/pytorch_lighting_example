"""
=================================================
@path   : pytorch_lighting_example -> transformer_model
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/23 9:58
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import torch
from torch import nn as nn

from examples.pytorch.transformer.positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, d_model=128, dropout=0.5):
        super().__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)
        # 定义位置编码器，词典数为10。我们只预测一位整数。
        self.pos_encoder = PositionalEncoding(d_model, dropout=0)
        # 定义Transformer
        self.transformer = nn.Transformer(
                d_model=128,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=512,
                batch_first=True
                )

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(128, 10)
        pass

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(
                src, tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
                )

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask
