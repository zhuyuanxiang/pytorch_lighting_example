"""
=================================================
@path   : pytorch_lighting_example -> gan
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022/11/11 10:37
@Version: v0.1
@License: (C)Copyright 2020-2022, SoonSolid
@Reference:
@Desc   :
@History:
@Plan   :
==================================================
"""
from typing import Optional

import torch
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import functional as F
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure

from torch_model.gan_models import Discriminator
from torch_model.gan_models import Generator


class GAN(LightningModule):
    def __init__(
            self,
            dims,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            **kwargs,
            ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=dims)
        self.discriminator = Discriminator(img_shape=dims)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.psnr = PeakSignalNoiseRatio().to(device=self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=self.device)

        self.g_loss = 0
        self.d_loss = 0
        pass

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = (self.hparams.b1, self.hparams.b2)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            generated_imgs = self(z)

            # log sampled images
            sample_imgs = generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)
            self.g_loss = self.g_loss + g_loss
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.d_loss = self.d_loss + d_loss
            return d_loss

    def on_train_epoch_end(self) -> None:
        train_dataloader_len = len(self.trainer.train_dataloader)
        self.log('g_loss', self.g_loss / train_dataloader_len)
        self.log('d_loss', self.d_loss / train_dataloader_len)

        self.g_loss = 0
        self.d_loss = 0

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return torch.tensor([0])

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
