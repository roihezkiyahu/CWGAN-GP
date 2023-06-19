import torch
import torchvision
import torchvision.transforms as transforms
from Generator import GeneratorMNIST
from Discriminator import DiscriminatorMNIST
from GAN import GAN
import lightning as pl
from PIL import Image
from IPython.display import Image as DisplayImage, display
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def create_gif(folder_path, output_path, fps=5, display=True,pause_end=5, output_name='output.gif'):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    epoch_numbers = [int(re.findall(r'res_epoch_(\d+)\.png', f)[0]) for f in image_files]
    image_files_sorted = [f for _, f in sorted(zip(epoch_numbers, image_files))]

    images = []

    for file in image_files_sorted:
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        images.append(image)
    for i in range(pause_end*fps):
        images.append(image)
    output_path = os.path.join(output_path, output_name)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0)
    print(f"GIF created successfully at: {output_path}")
    if display:
        display(DisplayImage(filename=output_path))

def plot_losses(losees):
    if isinstance(losees, str):
        losees = np.load(losees)
    d_losses_iters = losees['d_losses']
    g_losses_iters = losees['g_losses']
    d_len = len(d_losses_iters)
    g_len = len(g_losses_iters)
    jump_facor = d_len//g_len
    plt.plot(range(0, g_len), d_losses_iters[::jump_facor], label='Discriminator Loss')
    plt.plot(range(0, g_len), g_losses_iters, label='Generator Loss')
    plt.xlabel('Generator iterations')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Config
    batch_size = 64
    learning_rate_wgan = 2e-4
    learning_rate_dcgan = 5e-5
    num_epochs = 150
    lambda_gp = 10
    latent_dim = 128
    dim = 128
    n_critics = 1
    step_lr_gamma = 0.975
    betas = (0.5, 0.999)
    acc_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(acc_device)

    # image processing
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.RandomAffine(5, (0.1, 0.1), scale=(0.95, 1.05)),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Configuring Dataset
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    full_dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    class_names = train_set.classes

    # CWGAN-GP
    generator_wgan_mnist = GeneratorMNIST((32, 32, 1), latent_dim, dim, batch_norm=True)
    discriminator_wgan_mnist = DiscriminatorMNIST((32, 32, 1), dim, batch_norm=False)
    wgan_mnist = GAN(generator_wgan_mnist, discriminator_wgan_mnist, True, lambda_gp, device, learning_rate_wgan,
                     n_critics, betas, step_lr_gamma=step_lr_gamma, class_names=class_names)

    # DCGAN
    GeneratorMNIST = GeneratorMNIST((32, 32, 1), latent_dim, dim, batch_norm=True)
    DiscriminatorMNIST = DiscriminatorMNIST((32, 32, 1), dim, batch_norm=True)
    dcgan_mnist = GAN(GeneratorMNIST, DiscriminatorMNIST, False, 0, device, learning_rate_dcgan, n_critics, betas,
                      step_lr_gamma=step_lr_gamma, class_names=class_names)

    # Training
    trainer = pl.Trainer(accelerator=acc_device, max_epochs=num_epochs)
    trainer.fit(wgan_mnist, train_loader)

    # Creating gif
    folder_path = r"C:\Users\Nir\Downloads\bs64-dim128-critic1"
    create_gif(folder_path, folder_path, display=False)

    # load_from epoch
    epoch = 150
    plot_losses(os.path.join(folder_path, f"losses_epoch_{epoch}.npz"))

    D_path = os.path.join(folder_path, f"discriminator_epoch_{epoch}.pt")
    G_path = os.path.join(folder_path, f"generator_epoch_{epoch}.pt")
    wgan_mnist.load_model(D_path, G_path)
    wgan_mnist.plot_sample_imgs()