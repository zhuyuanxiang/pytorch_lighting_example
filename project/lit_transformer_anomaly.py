"""
=================================================
@path   : pytorch_lighting_example -> lit_transformer_anomaly
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 16:01
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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as func

# ----------------------------------------------------------------------
from lit_model.transformer import TransformerPredictor


class AnomalyPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch
        # No positional encodings as it is a set, not a sequence!
        preds = self.forward(img_sets, add_positional_encoding=False)
        preds = preds.squeeze(dim=-1)  # Shape: [Batch_size, set_size]
        loss = func.cross_entropy(preds, labels)  # Softmax/CE over set dimension
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


def train_anomaly(**kwargs):
    # Create a PyTorch Lightning train with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=100,
            gradient_clip_val=2,
            progress_bar_refresh_rate=1,
            )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = AnomalyPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = AnomalyPredictor(max_iters=trainer.max_epochs * len(train_anom_loader), **kwargs)
        trainer.fit(model, train_anom_loader, val_anom_loader)
        model = AnomalyPredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, dataloaders=train_anom_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=val_anom_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_anom_loader, verbose=False)
    result = {
            "test_acc": test_result[0]["test_acc"],
            "val_acc": val_result[0]["test_acc"],
            "train_acc": train_result[0]["test_acc"],
            }

    model = model.to(device)
    return model, result


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
