import unittest

from hydra import compose
from hydra import initialize_config_module
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from lit_model.classifier import LitModuleClassifier
from torch_datasets.mnist import mnist


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with initialize_config_module(version_base=None, config_module='config.test'):
            self.config = compose(config_name='classifier')
            seed_everything(self.config.seed)

    def test_lit_classifier(self):
        model = LitModuleClassifier()
        train, val, test = mnist(
                self.config.mnist_dataset.batch_size,
                self.config.datasets_path
                )
        trainer = Trainer(
                default_root_dir=self.config.trainer.default_root_dir,
                limit_train_batches=self.config.trainer.limit_train_batches,
                limit_val_batches=self.config.trainer.limit_val_batches,
                max_epochs=self.config.trainer.max_epochs,
                )
        trainer.fit(model, train, val)

        results = trainer.test(model, test)
        self.assertGreater(results[0]['test_loss'], 0.7)


if __name__ == '__main__':
    unittest.main()
