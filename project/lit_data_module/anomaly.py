"""
=================================================
@path   : pytorch_lighting_example -> anomaly
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/11 16:41
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
import os
from typing import Optional

import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.utils import data as data
from torchvision import transforms
# Torchvision
from torchvision.datasets import CIFAR100

from torch_datasets.anomaly import AnomalyDataset
from torch_datasets.tools import extract_features


class AnomalyDataModule(LightningDataModule):
    # Resize to 224x224, and normalize to ImageNet statistic
    # ImageNet statistics
    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])
    # As torch tensors for later preprocessing
    TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1, 3, 1, 1)
    TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1, 3, 1, 1)

    def __init__(self,
                 dataset_path: str = 'datasets/',
                 checkpoint_path: str = 'saved_models/Transformers/',
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train, self.data_val, self.data_test, self.data_predict = None, None, None, None

        self.transform = transforms.Compose(
                [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(self.DATA_MEANS, self.DATA_STD),
                        ]
                )
        pass

    def prepare_data(self) -> None:
        CIFAR100(root=self.hparams.dataset_path, train=True, download=True)
        CIFAR100(root=self.hparams.dataset_path, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Loading the training dataset.
        train_set = CIFAR100(root=self.hparams.dataset_path, train=True, transform=self.transform)

        # Loading the test set
        test_set = CIFAR100(root=self.hparams.dataset_path, train=False, transform=self.transform)

        pretrained_model = torchvision.models.resnet34(pretrained=True)
        # Remove classification layer
        # In some torch_model, it is called "fc", others have "classifier"
        # Setting both to an empty sequential represents an identity map of the final features.

        pretrained_model.fc = nn.Sequential()
        pretrained_model.classifier = nn.Sequential()
        # To GPU
        pretrained_model = pretrained_model

        # Only eval, no gradient required
        pretrained_model.eval()
        for p in pretrained_model.parameters():
            p.requires_grad = False

        train_feat_file = os.path.join(self.hparams.checkpoint_path, "train_set_features.tar")
        train_set_feats = extract_features(train_set, train_feat_file)

        test_feat_file = os.path.join(self.hparams.checkpoint_path, "test_set_features.tar")
        self.test_feats = extract_features(test_set, test_feat_file)

        print("Train:", train_set_feats.shape)
        print("Test: ", self.test_feats.shape)

        # Split train into train+val
        # Get labels from train set
        labels = train_set.targets

        # Get indices of images per class
        labels = torch.LongTensor(labels)
        num_labels = labels.max() + 1
        sorted_indices = torch.argsort(labels).reshape(num_labels, -1)  # [classes, num_imgs per class]

        # Determine number of validation images per class
        num_val_exmps = sorted_indices.shape[1] // 10

        # Get image indices for validation and training
        val_indices = sorted_indices[:, :num_val_exmps].reshape(-1)
        train_indices = sorted_indices[:, num_val_exmps:].reshape(-1)

        # Group corresponding image features and labels
        self.train_feats, self.train_labels = train_set_feats[train_indices], labels[train_indices]
        self.val_feats, self.val_labels = train_set_feats[val_indices], labels[val_indices]

        self.test_labels = torch.LongTensor(test_set.targets)
        if stage == "fit" or stage is None:
            self.data_train = AnomalyDataset(self.train_feats, self.train_labels, self.set_size)
            self.data_val = AnomalyDataset(self.val_feats, self.val_labels, self.set_size, train=False)
            pass
        if stage == "test" or stage is None:
            self.data_test = AnomalyDataset(self.test_feats, self.test_labels, self.set_size, train=False)
            pass
        if stage == "predict" or stage is None:
            self.data_predict = AnomalyDataset(self.test_feats, self.test_labels, self.set_size, train=False)
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return data.DataLoader(self.data_train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_val, batch_size=128, drop_last=True, pin_memory=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_test, batch_size=128, drop_last=True, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_predict, batch_size=128, drop_last=True, pin_memory=True)
