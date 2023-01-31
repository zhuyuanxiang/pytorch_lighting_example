"""
=================================================
@path   : pytorch_lighting_example -> transformer_predictor
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 16:06
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

from datetime import datetime

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim as optim

from hydra_conf.lit_config import LitTransformerReverse
from torch_model.transformer.encoder import TransformerEncoder
from torch_model.transformer.encoding import PositionalEncoding
from torch_model.transformer.scheduler import CosineWarmupScheduler


class TransformerPredictor(pl.LightningModule):
    def __init__(self, module_parameters: LitTransformerReverse):
        """
        Args:
            lr: 优化器的学习率
            warmup: 优化器热身的步数，属于[50,500]
            max_iters: 模型训练时的最大迭代次数，用于 CosineWarmup 调整策略
            :param module_parameters:
        """
        super().__init__()
        self.module_parameters = module_parameters
        self._create_model()
        self.lr_scheduler = None

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
                nn.Dropout(self.module_parameters.input_dropout),
                nn.Linear(
                        in_features=self.module_parameters.num_categories,
                        out_features=self.module_parameters.model_channels
                        )
                )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.module_parameters.model_channels)
        # Transformer
        self.transformer = TransformerEncoder(
                num_layers=self.module_parameters.num_layers,
                input_channels=self.module_parameters.model_channels,
                dim_feedforward=2 * self.module_parameters.model_channels,
                num_heads=self.module_parameters.num_heads,
                dropout=self.module_parameters.dropout,
                )
        # Output classifier per sequence element
        self.output_net = nn.Sequential(
                nn.Linear(
                        in_features=self.module_parameters.model_channels,
                        out_features=self.module_parameters.model_channels
                        ),
                nn.LayerNorm(normalized_shape=self.module_parameters.model_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(self.module_parameters.dropout),
                nn.Linear(
                        in_features=self.module_parameters.model_channels,
                        out_features=self.module_parameters.num_categories
                        ),
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.module_parameters.optimizer.learning_rate)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
                optimizer,
                warmup=self.module_parameters.optimizer.warmup,
                max_iters=self.module_parameters.optimizer.max_iters
                )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
