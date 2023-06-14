import torch
import torch.nn as nn

class Conv2dTBlockMNIST(nn.Module):
    """
    Convolutional Transpose Block for MNIST GAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default: 4.
        stride (int): Stride value of the convolutional kernel. Default: 2.
        padding (int): Padding value of the convolution. Default: 1.
        output_padding (int): Additional size added to one side of the output shape. Default: 0.
        batch_norm (bool): Whether to use batch normalization. Default: False.
        skip_connect (bool): Whether to use skip connections. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0,
                 batch_norm=False, skip_connect=True):
        super(Conv2dTBlockMNIST, self).__init__()
        self.skip_connect = skip_connect
        self.stride = stride
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear') if stride == 2 else None
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if in_channels != out_channels and skip_connect else None

    def forward(self, x):
        """
        Forward pass of the Conv2dTBlockMNIST.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_up = self.upsampler(x) if self.stride == 2 else x
        if self.conv1x1 is not None:
            x_up = self.conv1x1(x_up)
        x = self.block(x)
        if self.skip_connect:
            x = x + x_up
        return x


class GeneratorMNIST(nn.Module):
    """
    Generator for MNIST GAN.

    Args:
        img_size (tuple): Tuple containing the size of the generated image (height, width).
        latent_dim (int): Dimensionality of the latent space.
        dim (int): Number of filters in the first layer. Default: 64.
        num_upsamp (int): Number of upsampling layers. Default: 3.
        batch_norm (bool): Whether to use batch normalization. Default: True.
        n_classes (int): Number of classes. Default: 10.
        emb_dim (int): Dimensionality of the embedding layer. Default: 10.
        conv_input (bool): Whether to use convolutional input. Default: False.
    """
    def __init__(self, img_size, latent_dim, dim=64, num_upsamp=3, batch_norm=True, n_classes=10, emb_dim=10,
                 conv_input=True):
        super(GeneratorMNIST, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.feature_sizes = (img_size[0] // 2 ** num_upsamp, img_size[1] // 2 ** num_upsamp)

        self.l2f = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 4 * dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(0.2)
        ) if conv_input else nn.Sequential(
            nn.Linear(latent_dim, 4 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.LeakyReLU(0.2)
        )

        self.block_1 = Conv2dTBlockMNIST(4 * dim + 1, 2 * dim, batch_norm=batch_norm)
        self.block_2 = Conv2dTBlockMNIST(2 * dim, dim, batch_norm=batch_norm)
        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(dim, img_size[2], 3, 2, 1, 1),
            nn.Tanh()
        )

        self.label_block = nn.Sequential(
            nn.Embedding(n_classes, emb_dim),
            nn.Linear(emb_dim, self.feature_sizes[0] * self.feature_sizes[1])
        )

    def features_to_image(self, x):
        """
        Convert features to an image.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x

    def forward(self, input_data, label):
        """
        Forward pass of the GeneratorMNIST.

        Args:
            input_data (torch.Tensor): Input data tensor.
            label (torch.Tensor): Label tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.l2f(input_data.view(-1, self.latent_dim, 1, 1))
        x = x.view(-1, 4 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        label_reshaped = self.label_block(label).view(-1, 1, self.feature_sizes[0], self.feature_sizes[1])
        x = torch.cat([x, label_reshaped], dim=1)
        x = self.features_to_image(x)
        return x