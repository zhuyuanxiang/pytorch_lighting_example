"""
=================================================
@path   : pytorch_lighting_example -> train_transformer_reverse
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/16 17:07
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

# External libraries
import hydra
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline
import pytorch_lightning as pl
import seaborn as sns
import torch
from hydra.core.config_store import ConfigStore
from torch.nn import functional as func

# My libraries
from hydra_conf.datasets import ReverseDataset
from hydra_conf.lit_config import LitTransformerReverse
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerTransformer
from lit_data_module.reverse import ReverseDataModule
from lit_model.reverse import ReversePredictor
from parameters import HYDRA_PATH
from torch_model.tools import plot_attention_maps


@dataclass
class ConfigTransformerReverse(Config):
    trainer: TrainerTransformer = TrainerTransformer
    dataset: ReverseDataset = ReverseDataset
    lit_module: LitTransformerReverse = LitTransformerReverse
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigTransformerReverse)


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name='train_model')
def cli_main(config: ConfigTransformerReverse):
    plt.set_cmap("cividis")
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
    matplotlib.rcParams["lines.linewidth"] = 2.0
    sns.reset_orig()

    os.environ["TORCH_HOME"] = config.lit_module.checkpoint_path

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(config.seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    data_module = ReverseDataModule(config.dataset.num_categories, config.dataset.sequence_len)
    data_module.setup()

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(config.lit_module.checkpoint_path, "ReverseTask.ckpt1")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor(module_parameters=config.lit_module)

    # Create a PyTorch Lightning train with the generation callback
    root_dir = os.path.join(config.lit_module.checkpoint_path, "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    callbacks = [pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")]
    trainer = pl.Trainer(
            accelerator="auto",
            default_root_dir=root_dir,
            callbacks=callbacks,
            devices=1 if str(device).startswith("cuda") else 0,
            max_epochs=config.trainer.max_epochs,
            gradient_clip_val=5,
            # progress_bar_refresh_rate=1,
            )
    if not os.path.isfile(pretrained_filename):
        trainer.fit(model, data_module)

    # Test best model on validation and test set
    val_result = trainer.test(model, data_module.val_dataloader(), verbose=False)
    test_result = trainer.test(model, data_module.test_dataloader(), verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    print("Val accuracy:  %4.2f%%" % (100.0 * result["val_acc"]))
    print("Test accuracy: %4.2f%%" % (100.0 * result["test_acc"]))

    data_input, labels = next(iter(data_module.val_dataloader()))
    input_data = func.one_hot(data_input, num_classes=model.module_parameters.num_categories).float()
    attention_maps = model.get_attention_maps(input_data)

    print("attention_maps[0].shape=", attention_maps[0].shape)

    plot_attention_maps(data_input, attention_maps, idx=0)
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cli_main()
