from argparse import ArgumentParser

import hydra
import pytorch_lightning as pl

from data_module.mnist import mnist
from lit_model.classifier import LitClassifier


@hydra.main(version_base=None, config_path='../config/train', config_name="classifier")
def cli_main(config):
    pl.seed_everything(config.seed)

    # ------------
    # data
    # ------------
    train, val, test = mnist(config.mnist_dataset.batch_size)

    # ------------
    # model
    # ------------
    model = LitClassifier(
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
    trainer.fit(model, train, val)

    # ------------
    # testing
    # ------------
    trainer.test(model, test)


if __name__ == '__main__':
    cli_main()
