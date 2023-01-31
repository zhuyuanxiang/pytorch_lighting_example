"""
=================================================
@path   : pytorch_lighting_example -> lit_transformer
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/2 10:23
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

# Standard libraries
import os
from dataclasses import dataclass

import hydra
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import pytorch_lightning as pl
# PyTorch Lightning
import seaborn as sns
import torch
# PyTorch
import torch.nn.functional as func
import torch.utils.data as data
from hydra.core.config_store import ConfigStore

from hydra_conf.datasets import AnomalyDataset
from hydra_conf.lit_config import LitTransformer
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerTransformer
from lit_transformer_anomaly import train_anomaly
from parameters import HYDRA_PATH
from torch_datasets.anomaly import anomaly
from torch_model.tools import visualize_examples
from torch_model.tools import visualize_prediction


# Plotting


@dataclass
class ConfigTransformer(Config):
    trainer: TrainerTransformer = TrainerTransformer
    dataset: AnomalyDataset = AnomalyDataset
    lit_module: LitTransformer = LitTransformer
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigTransformer)


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name='train_model')
def cli_main(config: ConfigTransformer):
    plt.set_cmap("cividis")
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
    matplotlib.rcParams["lines.linewidth"] = 2.0
    sns.reset_orig()

    os.environ["TORCH_HOME"] = config.lit_module.checkpoint_path

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(config.seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device:", device)

    train_loader, test_loader = anomaly(config.dataset.batch_size)

    train_anom_loader = data.DataLoader(
            train_anom_dataset,
            batch_size=config.anomaly_dataset.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
            )
    val_anom_loader = data.DataLoader(val_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_anom_loader = data.DataLoader(test_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    _, indices, _ = next(iter(test_anom_loader))
    visualize_examples(indices[:4], test_set)

    anomaly_model, anomaly_result = train_anomaly(
            input_channels=train_set.img_feats.shape[-1],
            model_channels=256,
            num_heads=4,
            num_classes=1,
            num_layers=4,
            dropout=0.1,
            input_dropout=0.1,
            lr=5e-4,
            warmup=100,
            )

    print("Train accuracy: %4.2f%%" % (100.0 * anomaly_result["train_acc"]))
    print("Val accuracy:   %4.2f%%" % (100.0 * anomaly_result["val_acc"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * anomaly_result["test_acc"]))

    inp_data, indices, labels = next(iter(test_anom_loader))
    inp_data = inp_data.to(device)

    anomaly_model.eval()

    with torch.no_grad():
        preds = anomaly_model.forward(inp_data, add_positional_encoding=False)
        preds = func.softmax(preds.squeeze(dim=-1), dim=-1)

        # Permut input datasets
        permut = np.random.permutation(inp_data.shape[1])
        perm_inp_data = inp_data[:, permut]
        perm_preds = anomaly_model.forward(perm_inp_data, add_positional_encoding=False)
        perm_preds = func.softmax(perm_preds.squeeze(dim=-1), dim=-1)

    assert (preds[:, permut] - perm_preds).abs().max() < 1e-5, "Predictions are not permutation equivariant"

    print("Preds\n", preds[0, permut].cpu().numpy())
    print("Permuted preds\n", perm_preds[0].cpu().numpy())

    attention_maps = anomaly_model.get_attention_maps(inp_data, add_positional_encoding=False)
    predictions = preds.argmax(dim=-1)

    visualize_prediction(0, indices, predictions, attention_maps)

    mistakes = torch.where(predictions != 9)[0].cpu().numpy()
    print("Indices with mistake:", mistakes)
    visualize_prediction(mistakes[-1], indices, predictions, attention_maps)
    print("Probabilities:")
    for i, p in enumerate(preds[mistakes[-1]].cpu().numpy()):
        print("Image %i: %4.2f%%" % (i, 100.0 * p))
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cli_main()
