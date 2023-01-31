"""
=================================================
@path   : pytorch_lighting_example -> lit_rte_boolq_module
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/4 15:44
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import os
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import datasets
import pytorch_lightning as pl
import sentencepiece as sp  # noqa: F401 # isort: split
import torch
from datasets import logging as datasets_logging
from packaging.version import Version
from pytorch_lightning.utilities import rank_zero_warn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import logging as transformers_logging
from transformers.tokenization_utils_base import BatchEncoding

if Version(torch.__version__) == Version("1.12.0") or torch.__version__.startswith("1.12.0"):
    # we need to use a patched version of AdamW to fix https://github.com/pytorch/pytorch/issues/80809
    # and allow examples to succeed with torch 1.12.0 (this torch bug is fixed in 1.12.1)
    from fts_examples.patched_adamw import AdamW
else:
    from torch.optim.adamw import AdamW
from datetime import datetime

import torch
# Import the `FinetuningScheduler` PyTorch Lightning extension module we want to use. This will import all necessary
# callbacks.
import finetuning_scheduler as fts  # isort: split

# set notebook-level variables
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"

# reduce hf logging verbosity to focus on tutorial-relevant code/messages
for hflogger in [transformers_logging, datasets_logging]:
    hflogger.set_verbosity_error()
# ignore warnings related tokenizers_parallelism/DataLoader parallelism trade-off and
# expected logging behavior
for warnf in [
        r".*does not have many workers.*",
        r".*The number of training samples.*",
        r".*converting to a fast.*",
        r".*number of training batches.*",
        ]:
    warnings.filterwarnings("ignore", warnf)


class RteBoolqDataModule(pl.LightningDataModule):
    """A ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face datasets."""

    TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("question", "passage")}
    LOADER_COLUMNS = (
            "datasets_idx",
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
            "labels",
            )

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = DEFAULT_TASK,
            max_seq_length: int = 128,
            train_batch_size: int = 16,
            eval_batch_size: int = 16,
            tokenizers_parallelism: bool = True,
            **dataloader_kwargs: Any,
            ):
        r"""Initialize the ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            model_name_or_path (str):
                Can be either:
                    - A string, the ``model id`` of a pretrained model hosted inside a model repo on huggingface.co.
                        Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced
                        under
                        a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a ``directory`` containing model weights saved using
                        :meth:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
            task_name (str, optional): Name of the SuperGLUE task to execute. This module supports 'rte' or 'boolq'.
                Defaults to DEFAULT_TASK which is 'rte'.
            max_seq_length (int, optional): Length to which we will pad sequences or truncate input. Defaults to 128.
            train_batch_size (int, optional): Training batch size. Defaults to 16.
            eval_batch_size (int, optional): Batch size to use for validation and testing splits. Defaults to 16.
            tokenizers_parallelism (bool, optional): Whether to use parallelism in the tokenizer. Defaults to True.
            \**dataloader_kwargs: Arguments passed when initializing the dataloader
        """
        super().__init__()
        task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.text_fields = self.TASK_TEXT_FIELD_MAP[task_name]
        self.dataloader_kwargs = {
                "num_workers": dataloader_kwargs.get("num_workers", 0),
                "pin_memory": dataloader_kwargs.get("pin_memory", False),
                }
        self.save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.hparams.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_name_or_path, use_fast=True, local_files_only=False
                )

    def prepare_data(self):
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.hparams.task_name)

    def setup(self, stage):
        """Setup our dataset splits for training/validation."""
        self.dataset = datasets.load_dataset("super_glue", self.hparams.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                    self._convert_to_features, batched=True, remove_columns=["label"]
                    )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.hparams.eval_batch_size, **self.dataloader_kwargs)

    def _convert_to_features(self, example_batch: datasets.arrow_dataset.Batch) -> BatchEncoding:
        """Convert raw text examples to a :class:`~transformers.tokenization_utils_base.BatchEncoding` container
        (derived from python dict) of features that includes helpful methods for translating between word/character
        space and token space.

        Args:
            example_batch ([type]): The set of examples to convert to token space.

        Returns:
            ``BatchEncoding``: A batch of encoded examples (note default tokenizer batch_size=1000)
        """
        text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
                text_pairs, max_length=self.hparams.max_seq_length, padding="longest", truncation=True
                )
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        return features


class RteBoolqModule(pl.LightningModule):
    """A ``LightningModule`` that can be used to fine-tune a foundational model on either the RTE or BoolQ
    SuperGLUE tasks using Hugging Face implementations of a given model and the `SuperGLUE Hugging Face dataset."""

    def __init__(
            self,
            model_name_or_path: str,
            optimizer_init: Dict[str, Any],
            lr_scheduler_init: Dict[str, Any],
            model_cfg: Optional[Dict[str, Any]] = None,
            task_name: str = DEFAULT_TASK,
            experiment_tag: str = "default",
            ):
        """
        Args:
            model_name_or_path (str): Path to pretrained model or identifier from https://huggingface.co/models
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
        """
        super().__init__()
        if task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(f"Invalid task_name {task_name!r}. Proceeding with the default task: {DEFAULT_TASK!r}")
            task_name = DEFAULT_TASK
        self.num_labels = TASK_NUM_LABELS[task_name]
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.init_hparams = {
                "optimizer_init": optimizer_init,
                "lr_scheduler_init": lr_scheduler_init,
                "model_config": self.model.config,
                "model_name_or_path": model_name_or_path,
                "task_name": task_name,
                "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}",
                }
        self.save_hyperparameters(self.init_hparams)
        self.metric = datasets.load_metric(
                "super_glue", self.hparams.task_name, experiment_id=self.hparams.experiment_id
                )
        self.no_decay = ["bias", "LayerNorm.weight"]

    @property
    def finetuningscheduler_callback(self) -> fts.FinetuningScheduler:
        fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]
        return fts_callback[0] if fts_callback else None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.finetuningscheduler_callback:
            self.log("finetuning_schedule_depth", float(self.finetuningscheduler_callback.curr_depth))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metric_dict, prog_bar=True)

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
                {
                        "params": [
                                p
                                for n, p in self.model.named_parameters()
                                if not any(nd in n for nd in self.no_decay) and p.requires_grad
                                ],
                        "weight_decay": self.hparams.optimizer_init["weight_decay"],
                        },
                {
                        "params": [
                                p
                                for n, p in self.model.named_parameters()
                                if any(nd in n for nd in self.no_decay) and p.requires_grad
                                ],
                        "weight_decay": 0.0,
                        },
                ]

    def configure_optimizers(self):
        # the phase 0 parameters will have been set to require gradients during setup
        # you can initialize the optimizer with a simple requires.grad filter as is often done,
        # but in this case we pass a list of parameter groups to ensure weight_decay is
        # not applied to the bias parameter (for completeness, in this case it won't make much
        # performance difference)
        optimizer = AdamW(params=self._init_param_groups(), **self.hparams.optimizer_init)
        scheduler = {
                "scheduler": CosineAnnealingWarmRestarts(optimizer, **self.hparams.lr_scheduler_init),
                "interval": "epoch",
                }
        return [optimizer], [scheduler]


# ----------------------------------------------------------------------
def main(name):
    print(f'Hi, {name} 训练模型！', datetime.now())
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
