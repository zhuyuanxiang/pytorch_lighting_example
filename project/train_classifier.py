from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore

from hydra_conf.datasets import DatasetConfig
from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitConfig
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerConfig
from lit_model.classifier import LitModuleClassifier
from parameters import HYDRA_PATH
from torch_datasets.mnist import mnist


@dataclass
class LitClassifier(LitConfig):
    checkpoint_path: str = 'saved_models/Classifier/'
    pass


@dataclass
class TrainerClassifier(TrainerConfig):
    max_epochs: int = 51
    pass


@dataclass
class ConfigClassifier(Config):
    trainer: TrainerConfig = TrainerClassifier
    dataset: DatasetConfig = MNISTDataset
    lit_classifier: LitConfig = LitClassifier
    pass


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigClassifier)


@hydra.main(version_base=None, config_path=HYDRA_PATH, config_name="train_model")
def cli_main(config: ConfigClassifier):
    pl.seed_everything(config.seed)

    # ------------
    # datasets
    # ------------
    train_loader, val_loader, test_loader = mnist(
            config.dataset.batch_size,
            config.dataset.path
            )

    # ------------
    # model
    # ------------
    model = LitModuleClassifier(
            config.lit_classifier.hidden_dim,
            config.lit_classifier.learning_rate
            )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
            default_root_dir=config.trainer.default_root_dir,
            max_epochs=config.trainer.max_epochs
            )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(model, test_loader)


if __name__ == '__main__':
    cli_main()
