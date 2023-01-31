"""
=================================================
@path   : pytorch_lighting_example -> reverse
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
from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils import data as data
from tqdm import tqdm

from torch_datasets.reverse import ReverseDataset


class ReverseDataModule(LightningDataModule):
    def __init__(self, num_categories, sequence_len):
        super().__init__()
        self.data_train, self.data_val, self.data_test, self.data_predict = None, None, None, None
        self.num_categories = num_categories
        self.sequence_len = sequence_len
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """不同的状态初始化不同的数据集，如果没有状态就初始化所有数据集

        :param stage: 调用数据集时的模型状态
        :return:
        """
        if stage == "fit" or stage is None:
            self.data_train = ReverseDataset(self.num_categories, self.sequence_len, data_size=50000)
            self.data_val = ReverseDataset(self.num_categories, self.sequence_len, data_size=1000)
            pass
        if stage == "test" or stage is None:
            self.data_test = ReverseDataset(self.num_categories, self.sequence_len, data_size=10000)
            pass
        if stage == "predict" or stage is None:
            self.data_predict = ReverseDataset(self.num_categories, self.sequence_len, data_size=1000)
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return data.DataLoader(self.data_train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_val, batch_size=128)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_test, batch_size=128)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.data_predict, batch_size=128)


# ----------------------------------------------------------------------
def main():
    dm = ReverseDataModule(num_categories=10, sequence_len=16)
    dm.setup("fit")
    input_data, labels = dm.train_dataloader().dataset[0]
    print("Input datasets:", input_data)
    print("Labels:    ", labels)
    print("len(train_loader)=", len(dm.train_dataloader()))
    for index, (input_data, labels) in tqdm(enumerate(dm.train_dataloader())):
        if index < 3:
            print("--->", index, "<---")
            print("Input datasets:", input_data[0])
            print("Labels:    ", labels[0])
        else:
            break
    pass


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
