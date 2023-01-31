"""
=================================================
@path   : pytorch_lighting_example -> anomaly
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/8 10:43
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data as data
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from preposition import device


def anomaly(batch_size=32):
    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose(
            [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(DATA_MEANS, DATA_STD),
                    ]
            )
    train_set = CIFAR100('', train=True, transform=transform, download=True)
    test_set = CIFAR100('', train=False, transform=transform, download=True)

    pretained_model = load_pretained_model()
    return train_set, test_set


def load_pretained_model():
    pretrained_model = torchvision.models.resnet34(pretrained=True)
    # Remove classification layer
    # In some models, it is called "fc", others have "classifier"
    # Setting both to an empty sequential represents an identity map of the final features.
    pretrained_model.fc = nn.Sequential()
    pretrained_model.classifier = nn.Sequential()
    # To GPU
    pretrained_model = pretrained_model.to(device)
    # Only eval, no gradient required
    pretrained_model.eval()
    for p in pretrained_model.parameters():
        p.requires_grad = False
        pass
    pass


class AnomalyDataset(data.Dataset):
    def __init__(self, img_feats, labels, set_size=10, train=True):
        """
        Args:
            img_feats: Tensor of shape [num_imgs, img_dim]. Represents the high-level features.
            labels: Tensor of shape [num_imgs], containing the class labels for the images
            set_size: Number of elements in a set. N-1 are sampled from one class, and one from another one.
            train: If True, a new set will be sampled every time __getitem__ is called.
        """
        super().__init__()
        self.img_feats = img_feats
        self.labels = labels
        self.set_size = set_size - 1  # The set size is here the size of correct images
        self.train = train

        # Tensors with indices of the images per class
        self.num_labels = labels.max() + 1
        self.img_idx_by_label = torch.argsort(self.labels).reshape(self.num_labels, -1)

        if not train:
            self.test_sets = self._create_test_sets()

    def _create_test_sets(self):
        # Pre-generates the sets for each image for the test set
        test_sets = []
        num_imgs = self.img_feats.shape[0]
        np.random.seed(42)
        test_sets = [self.sample_img_set(self.labels[idx]) for idx in range(num_imgs)]
        test_sets = torch.stack(test_sets, dim=0)
        return test_sets

    def sample_img_set(self, anomaly_label):
        """Samples a new set of images, given the label of the anomaly.

        The sampled images come from a different class than anomaly_label
        """
        # Sample class from 0,...,num_classes-1 while skipping anomaly_label as class
        set_label = np.random.randint(self.num_labels - 1)
        if set_label >= anomaly_label:
            set_label += 1

        # Sample images from the class determined above
        img_indices = np.random.choice(self.img_idx_by_label.shape[1], size=self.set_size, replace=False)
        img_indices = self.img_idx_by_label[set_label, img_indices]
        return img_indices

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        anomaly = self.img_feats[idx]
        if self.train:  # If train => sample
            img_indices = self.sample_img_set(self.labels[idx])
        else:  # If test => use pre-generated ones
            img_indices = self.test_sets[idx]

        # Concatenate images. The anomaly is always the last image for simplicity
        img_set = torch.cat([self.img_feats[img_indices], anomaly[None]], dim=0)
        indices = torch.cat([img_indices, torch.LongTensor([idx])], dim=0)
        label = img_set.shape[0] - 1

        # We return the indices of the images for visualization purpose. "Label" is the index of the anomaly
        return img_set, indices, label
