import lightning as pl
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class GAN(pl.LightningModule):
    """
    GAN class for training and generating images using a Generative Adversarial Network.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        wgan_mode (bool): Whether to use Wasserstein GAN (WGAN) mode. Default is False.
        lambda_gp (float): Gradient penalty coefficient for WGAN-GP. Default is 0.
        device (str): Device to use for training. Default is "cuda" if available, otherwise "cpu".
        lr (float): Learning rate for the Adam optimizer. Default is 2e-4.
        betas (Tuple[float, float]): Coefficients for computing running averages of gradient and its square. Default is (0.5, 0.999).
        n_critic (int): Number of iterations to train the discriminator for each generator iteration. Default is 1.
        generator_loss (nn.Module): Loss function for the generator. Default is nn.BCELoss().
        discriminator_loss (nn.Module): Loss function for the discriminator. Default is nn.BCELoss().
        val_imgs (int): Number of images to generate during validation. Default is 100.
        ncols (int): Number of columns in the generated image grid during validation. Default is 10.
        step_lr_gamma (float): Gamma factor for learning rate scheduler. Default is 0.975.
        save_folder (str): Folder to save generated images and model checkpoints. Default is "epoch_imgs".
        generator_lr_factor (float): Learning rate factor for the generator. Default is 1.
        fuzzy (Tuple[float, float]): Fuzziness range for label smoothing. Default is (0.2, 0.2).
        class_names (List[str]): List of class names for label generation. Default is None.
        optimzers (Tuple[torch.optim.Optimizer, torch.optim.Optimizer]): Custom optimizers for the generator and discriminator. Default is None.
        schedulers (Tuple[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler]): Custom learning rate schedulers for the generator and discriminator. Default is None.
        save_interval (int): Interval (in epochs) for saving model checkpoints. Default is 10.
        zip_folder (bool): Whether to zip the save folder containing images and checkpoints. Default is True.
        verbosity (int): Verbosity level for printing training progress. Default is 2.
    """
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
        self.device_ = device
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
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Input noise tensor.
            batch_labels (torch.Tensor): Labels for conditional generation.

        Returns:
            torch.Tensor: Generated images.
        """
        return self.generator(z, batch_labels)

    def compute_gradient_penalty(self, real_samples, fake_samples, batch_labels):
        """
        Compute the gradient penalty for WGAN-GP.

        Args:
            real_samples (torch.Tensor): Real images from the dataset.
            fake_samples (torch.Tensor): Generated images.
            batch_labels (torch.Tensor): Labels for conditional generation.

        Returns:
            torch.Tensor: Computed gradient penalty.
        """
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device_)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).clone().detach().requires_grad_(True)
        interpolates = interpolates.to(self.device_)
        d_interpolates = self.discriminator(interpolates, batch_labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device_)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device_)
        gradients_norm = ((torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) - 1) ** 2).mean()
        return gradients_norm

    def get_labels(self, batch_size, zeros=True):
        """
         Generate labels for conditional generation.

         Args:
             batch_size (int): Number of labels to generate.
             zeros (bool): Whether to generate labels for real images (zeros) or fake images (ones). Default is True.

         Returns:
             torch.Tensor: Generated labels.
         """
        zero_fuzz, ones_fuzz = self.fuzzy
        if zeros:
            if not zero_fuzz:
                labels = torch.full((batch_size,), 0, device=self.device_)
            else:
                labels = torch.rand(batch_size, device=self.device_) * zero_fuzz
        else:
            if not ones_fuzz:
                labels = torch.full((batch_size,), 1, device=self.device_)
            else:
                labels = 1 - torch.rand(batch_size, device=self.device_) * ones_fuzz * 2 - ones_fuzz
        return labels

    def train_generator(self, optimizer_g, z, batch_labels):
        """
        Train the generator.

        Args:
            optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
            z (torch.Tensor): Input noise tensor.
            batch_labels (torch.Tensor): Labels for conditional generation.

        Returns:
            torch.Tensor: Loss value for the generator.
        """
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
        """
        Train the discriminator.

        Args:
            optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
            z (torch.Tensor): Input noise tensor.
            imgs (torch.Tensor): Real images from the dataset.
            batch_labels (torch.Tensor): Labels for conditional generation.

        Returns:
            torch.Tensor: Loss value for the discriminator.
        """
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
        """
        Perform a single training step.

        Args:
            batch: Batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the discriminator.
        """
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
        """
        Configure the optimizers and learning rate schedulers.

        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Optimizers for the generator and discriminator.
        """
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
        """
        Plots a grid of sample images.

        Args:
            sample_imgs (torch.Tensor): Tensor containing the sample images to plot.

        Returns:
            None
        """
        img_grid = torchvision.utils.make_grid(sample_imgs.data, self.ncols).cpu()
        img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
        plt.imshow(img_grid)
        label = f'Generated set of images (epoch {self.current_epoch})'
        plt.title(label)
        if self.save_folder != "":
            plt.savefig(os.path.join(self.save_folder, f'res_epoch_{self.current_epoch}.png'))
        plt.show()

    def schedulers_step(self):
        """
        Takes a step in the learning rate schedulers.

        This function steps the learning rate schedulers for both the generator and
        discriminator optimizers.

        Returns:
            None
        """
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def get_cur_mean_loss(self):
        """
        Calculates the current mean generator and discriminator losses.

        Returns:
            Tuple: A tuple containing the mean generator loss and mean discriminator loss.
        """
        gen_loss = torch.tensor(self.g_losses_batch).mean()
        des_loss = torch.tensor(self.d_losses_batch).mean()
        return gen_loss, des_loss

    def plot_sample_imgs(self):
        """
        Plots a grid of sample images using the current generator model.

        Returns:
            None
        """
        z = self.validation_z
        with torch.no_grad():
            sample_imgs = self(z.to(self.device_), self.classes_z.to(self.device_)).detach()
        self.plot_grid((sample_imgs + 1) / 2)

    def get_cur_lr(self):
        """
        Retrieves the current learning rates for the generator and discriminator.

        Returns:
            Tuple: A tuple containing the learning rate for the generator and discriminator.
        """
        learning_rate_generator = self.optimizers()[0].param_groups[0]['lr']
        learning_rate_discriminator = self.optimizers()[1].param_groups[0]['lr']
        return learning_rate_generator, learning_rate_discriminator

    def save_state(self):
        """
        Saves the current state of the generator and discriminator models, as well as the losses.

        Returns:
            None
        """
        generator_file = os.path.join(self.save_folder, f"generator_epoch_{self.current_epoch}.pt")
        discriminator_file = os.path.join(self.save_folder, f"discriminator_epoch_{self.current_epoch}.pt")
        losses_file = os.path.join(self.save_folder, f"losses_epoch_{self.current_epoch}.npz")

        torch.save(self.generator.state_dict(), generator_file)
        torch.save(self.discriminator.state_dict(), discriminator_file)
        np.savez(losses_file, d_losses=self.d_losses_iters, g_losses=self.g_losses_iters)

    def load_model(self, discriminator_path, generator_path):
        """
        Loads a pre-trained generator and discriminator models from the given paths.

        Args:
            discriminator_path (str): Path to the pre-trained discriminator model.
            generator_path (str): Path to the pre-trained generator model.

        Returns:
            None
        """
        self.discriminator.load_state_dict(torch.load(discriminator_path))
        self.generator.load_state_dict(torch.load(generator_path))
        print('Generator model loaded from {}.'.format(generator_path))
        print('Discriminator model loaded from {}-'.format(discriminator_path))

    def zip_save_folder(self):
        """
        Creates a zip file of the saved results folder.

        Returns:
            None
        """
        zip_file = f"results.zip"
        shutil.make_archive(self.save_folder, 'zip', self.save_folder)
        os.rename(f"{self.save_folder}.zip", zip_file)

    def on_train_epoch_end(self):
        """
        Performs necessary operations at the end of each training epoch.

        This function calculates the average generator and discriminator losses,
        updates the learning rate schedulers, and prints the current losses and
        learning rates if the verbosity level is set appropriately. It also plots
        a grid of sample images if the verbosity level is high enough.

        Additionally, it saves the current state of the generator and discriminator
        models, and optionally creates a zip file of the saved results.

        Returns:
            None
        """
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
