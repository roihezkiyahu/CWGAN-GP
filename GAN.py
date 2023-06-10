import lightning as pl
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class GAN(pl.LightningModule):
    def __init__(
            self,
            generator,
            discriminator,
            wgan_mode=False,
            lambda_gp=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=2e-4,
            betas=(0.5, 0.999),
            n_critic=1,
            generator_loss=nn.BCELoss(),
            discriminator_loss=nn.BCELoss(),
            val_imgs=100,
            ncols=10,
            step_lr_gamma=0.975,
            save_folder="epoch_imgs",
            generator_lr_factor=1,
            fuzzy=(0.2, 0.2),
            class_names=None,
            optimzers=None,
            schedulers=None,
            save_interval=10,
            zip_folder=True,
            verbosity=2

    ):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss # will only be used if wgan_mode is False
        self.discriminator_loss = discriminator_loss # will only be used if wgan_mode is False
        self.wgan_mode = wgan_mode
        self.lambda_gp = lambda_gp
        self.device = device
        self.lr = lr
        self.betas = betas
        self.n_critic = n_critic
        self.validation_z = torch.randn(val_imgs, generator.latent_dim).to(device)
        self.label_2_idx = {name: idx for idx, name in enumerate(class_names)} if class_names else {}
        self.classes_z = torch.arange(len(class_names)).repeat(val_imgs // ncols).to(device) if class_names else None
        self.ncols = ncols
        self.g_losses_batch = []
        self.d_losses_batch = []
        self.d_losses_iters = []
        self.g_losses_iters = []
        self.step_lr_gamma = step_lr_gamma
        self.save_folder = save_folder
        self.fuzzy = fuzzy

        if save_folder != "":
            os.makedirs(save_folder, exist_ok=True)

        self.generator_lr_factor = generator_lr_factor
        self.optimzers_input = optimzers
        self.schedulers_input = schedulers
        self.save_interval = save_interval
        self.zip_folder = zip_folder
        self.verbosity = verbosity

    def forward(self, z, batch_labels):
        return self.generator(z, batch_labels)

    def compute_gradient_penalty(self, real_samples, fake_samples, batch_labels):
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).clone().detach().requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates, batch_labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradients_norm = ((torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) - 1) ** 2).mean()
        return gradients_norm

    def get_labels(self, batch_size, zeros=True):
        zero_fuzz, ones_fuzz = self.fuzzy
        if zeros:
            if not zero_fuzz:
                labels = torch.full((batch_size,), 0, device=self.device)
            else:
                labels = torch.rand(batch_size, device=self.device) * zero_fuzz
        else:
            if not ones_fuzz:
                labels = torch.full((batch_size,), 1, device=self.device)
            else:
                labels = 1 - torch.rand(batch_size, device=self.device) * ones_fuzz * 2 - ones_fuzz
        return labels

    def train_generator(self, optimizer_g, z, batch_labels):
        self.toggle_optimizer(optimizer_g)
        fake_imgs = self(z, batch_labels)
        fake_validity = self.discriminator(fake_imgs, batch_labels)
        batch_size = fake_imgs.shape[0]
        if self.wgan_mode:
            labels = self.get_labels(batch_size, True)
            g_loss = -torch.mean(fake_validity - labels)
        else:
            labels = self.get_labels(batch_size, False)
            g_loss = self.generator_loss(fake_validity.view(-1), labels)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        g_loss_cpu = g_loss.detach().cpu()
        self.g_losses_batch.append(g_loss_cpu)
        self.g_losses_iters = np.append(self.g_losses_iters, g_loss_cpu)
        return g_loss

    def train_discriminator(self, optimizer_d, z, imgs, batch_labels):
        self.toggle_optimizer(optimizer_d)
        fake_imgs = self(z, batch_labels)
        batch_size = fake_imgs.shape[0]
        real_validity = self.discriminator(imgs, batch_labels)
        fake_validity = self.discriminator(fake_imgs, batch_labels)
        labels_fake = self.get_labels(batch_size, True)
        labels_real = self.get_labels(batch_size, False)
        if self.wgan_mode:
            gradients_norm = self.compute_gradient_penalty(imgs.data, fake_imgs.data, batch_labels)
            d_loss = -torch.mean(real_validity - labels_real) + torch.mean(
                fake_validity - labels_fake) + self.lambda_gp * gradients_norm
        else:
            d_loss_real = self.discriminator_loss(real_validity.view(-1), labels_real)
            d_loss_fake = self.discriminator_loss(fake_validity.view(-1), labels_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        d_loss_cpu = d_loss.detach().cpu()
        self.d_losses_batch.append(d_loss_cpu)
        self.d_losses_iters = np.append(self.d_losses_iters, d_loss_cpu)
        return d_loss

    def training_step(self, batch, batch_idx):
        imgs, batch_labels = batch
        batch_size = imgs.shape[0]
        optimizer_g, optimizer_d = self.optimizers()
        z = torch.randn(batch_size, self.generator.latent_dim)
        z = z.type_as(imgs)
        if self.train_iter % self.n_critic == 0:
            g_loss = self.train_generator(optimizer_g, z, batch_labels)
        output = self.train_discriminator(optimizer_d, z, imgs, batch_labels)
        self.train_iter += 1
        return output

    def configure_optimizers(self):
        if isinstance(self.optimzers_input, type(None)):
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr * self.generator_lr_factor, betas=self.betas)
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        else:
            opt_g, opt_d = self.optimzers_input
        for group in opt_g.param_groups:
            group['initial_lr'] = self.lr * self.generator_lr_factor
        for group in opt_d.param_groups:
            group['initial_lr'] = self.lr
        if isinstance(self.schedulers_input, type(None)):
            scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, 1, self.step_lr_gamma)
            scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, 1, self.step_lr_gamma)
        else:
            scheduler_g, scheduler_d = self.schedulers_input
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def plot_grid(self, sample_imgs):
        img_grid = torchvision.utils.make_grid(sample_imgs.data, self.ncols).cpu()
        img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
        plt.imshow(img_grid)
        label = f'Generated set of images (epoch {self.current_epoch})'
        plt.title(label)
        if self.save_folder != "":
            plt.savefig(os.path.join(self.save_folder, f'res_epoch_{self.current_epoch}.png'))
        plt.show()

    def schedulers_step(self):
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def get_cur_mean_loss(self):
        gen_loss = torch.tensor(self.g_losses_batch).mean()
        des_loss = torch.tensor(self.d_losses_batch).mean()
        return gen_loss, des_loss

    def plot_sample_imgs(self):
        z = self.validation_z
        with torch.no_grad():
            sample_imgs = self(z.to(self.device), self.classes_z.to(self.device)).detach()
        self.plot_grid((sample_imgs + 1) / 2)

    def get_cur_lr(self):
        learning_rate_generator = self.optimizers()[0].param_groups[0]['lr']
        learning_rate_discriminator = self.optimizers()[1].param_groups[0]['lr']
        return learning_rate_generator, learning_rate_discriminator

    def save_state(self):
        generator_file = os.path.join(self.save_folder, f"generator_epoch_{self.current_epoch}.pt")
        discriminator_file = os.path.join(self.save_folder, f"discriminator_epoch_{self.current_epoch}.pt")
        losses_file = os.path.join(self.save_folder, f"losses_epoch_{self.current_epoch}.npz")

        torch.save(self.generator.state_dict(), generator_file)
        torch.save(self.discriminator.state_dict(), discriminator_file)
        np.savez(losses_file, d_losses=self.d_losses_iters, g_losses=self.g_losses_iters)

    def load_model(self, discriminator_path, generator_path):
        self.discriminator.load_state_dict(torch.load(discriminator_path))
        self.generator.load_state_dict(torch.load(generator_path))
        print('Generator model loaded from {}.'.format(generator_path))
        print('Discriminator model loaded from {}-'.format(discriminator_path))

    def zip_save_folder(self):
        zip_file = f"results.zip"
        shutil.make_archive(self.save_folder, 'zip', self.save_folder)
        os.rename(f"{self.save_folder}.zip", zip_file)

    def on_train_epoch_end(self):
        gen_loss, des_loss = self.get_cur_mean_loss()
        self.schedulers_step()
        learning_rate_generator, learning_rate_discriminator = self.get_cur_lr()
        if self.verbosity in [1, 2]:
            print(
                f'''Epoch {self.current_epoch}: Generator loss: {gen_loss.item():.4f}, Discriminator loss: {des_loss.item():.4f},
            learning_rate_generator is: {learning_rate_generator}, learning_rate_discriminator is: {learning_rate_discriminator}''')
        self.g_losses_batch, self.d_losses_batch = [], []
        if self.verbosity in [2, 3]:
            self.plot_sample_imgs()
        if self.current_epoch % self.save_interval == 0:
            self.save_state()
            if self.zip_folder:
                self.zip_save_folder()
