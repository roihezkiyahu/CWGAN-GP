import torch
import torch.nn as nn
import numpy as np

class Conv2dBlockMNIST(nn.Module):
    """
    Convolutional Block for MNIST Discriminator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default: 3.
        stride (int): Stride value of the convolutional kernel. Default: 2.
        padding (int): Padding value of the convolution. Default: 1.
        batch_norm (bool): Whether to use batch normalization. Default: False.
        skip_connect (bool): Whether to use skip connections. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, batch_norm=False,
                 skip_connect=True):
        super(Conv2dBlockMNIST, self).__init__()
        self.skip_connect = skip_connect
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.downsampler = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if in_channels != out_channels and skip_connect else None

    def forward(self, x):
        """
        Forward pass of the Conv2dBlockMNIST.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_down = self.downsampler(x) if self.stride == 2 else x
        if self.conv1x1 is not None and self.skip_connect:
            x_down = self.conv1x1(x_down)
        x = self.block(x)
        if self.skip_connect:
            x = x + x_down
        return x


class DiscriminatorMNIST(nn.Module):
    """
    Discriminator for MNIST GAN.

    Args:
        img_size (tuple): Tuple containing the size of the input image (height, width, channels).
        dim (int): Number of filters in the first layer. Default: 64.
        num_downsamp (int): Number of downsampling layers. Default: 3.
        batch_norm (bool): Whether to use batch normalization. Default: False.
        n_classes (int): Number of classes. Default: 10.
        emb_dim (int): Dimensionality of the embedding layer. Default: 10.
    """

    def __init__(self, img_size, dim=64, num_downsamp=3, batch_norm=False, n_classes=10, emb_dim=10):
        super(DiscriminatorMNIST, self).__init__()
        self.dim = dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] // 2 ** num_downsamp, self.img_size[1] // 2 ** num_downsamp)

        self.block_1 = Conv2dBlockMNIST(self.img_size[2]*2, dim)
        self.block_2 = Conv2dBlockMNIST(dim, 2 * dim, batch_norm=batch_norm)
        self.block_3 = Conv2dBlockMNIST(2 * dim, 4 * dim, batch_norm=batch_norm)
        self.predictor = nn.Sequential(nn.Linear(4 * np.prod(self.feature_sizes) * self.dim, 1),
                                       nn.Sigmoid())
        self.label_block = nn.Sequential(nn.Embedding(n_classes, emb_dim),
                                         nn.Linear(emb_dim, np.prod(img_size)))

    def image_to_features(self, x, add_noise=True):
        """
        Convert an image to features.

        Args:
            x (torch.Tensor): Input image tensor.
            add_noise (bool): Whether to add noise to the input image. Default: True.

        Returns:
            torch.Tensor: Output feature tensor.
        """
        if add_noise:
            x = x + torch.randn_like(x) * 0.1
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x

    def forward(self, input_data, label, add_noise=True):
        """
        Forward pass of the DiscriminatorMNIST.

        Args:
            input_data (torch.Tensor): Input data tensor.
            label (torch.Tensor): Label tensor.
            add_noise (bool): Whether to add noise to the input image. Default: True.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = input_data.view(-1, self.img_size[2], self.img_size[0], self.img_size[1])
        label_reshaped = self.label_block(label).view(-1, self.img_size[2], self.img_size[0], self.img_size[1])
        x = torch.cat([x, label_reshaped], dim=1)
        x = self.image_to_features(x, add_noise)
        x = x.view(-1, 4 * np.prod(self.feature_sizes) * self.dim)
        x = self.predictor(x)
        return x
