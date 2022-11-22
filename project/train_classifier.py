from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore

from hydra_conf.datasets import MNISTDataset
from hydra_conf.lit_config import LitClassifier
from hydra_conf.train_config import Config
from hydra_conf.trainer_config import TrainerClassifier
from lit_model.classifier import LitModuleClassifier
from parameters import HYDRA_PATH
from torch_datasets.mnist import mnist


@dataclass
class ConfigClassifier(Config):
    trainer: TrainerClassifier = TrainerClassifier
    dataset: MNISTDataset = MNISTDataset
    lit_module: LitClassifier = LitClassifier
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
            config.lit_module.hidden_dim,
            config.lit_module.optimizer.learning_rate
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
