"""
=================================================
@path   : pytorch_lighting_example -> reverse
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/17 8:57
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from torch.nn import functional as func

from lit_model.transformer import TransformerPredictor


class ReversePredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        # Fetch datasets and transform categories to one-hot vectors
        input_data, labels = batch
        input_data = func.one_hot(input_data, num_classes=self.module_parameters.num_categories).float()

        # Perform prediction and calculate loss and accuracy
        predictions = self.forward(input_data, add_positional_encoding=True)
        loss = func.cross_entropy(predictions.view(-1, predictions.size(-1)), labels.view(-1))
        accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, accuracy)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
