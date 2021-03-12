import torch
import numpy as np
import torch.nn as nn

from torchvision.models import resnet18
from torch.nn.utils import spectral_norm


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = list(resnet18(pretrained=True).children())
        self.backbone = nn.Sequential(*resnet[:-1])

    def forward(self, x):
        # TODO: add ImageNet Normalization
        return self.backbone(x).view(x.size(0), -1)


class Conv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True, use_sn=False):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if use_sn:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """ Residual Block with instance normalization. """

    def __init__(self, dim_in, dim_out, use_sn=False):
        super().__init__()
        self.main = nn.Sequential(
            Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False, use_sn=use_sn),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False, use_sn=use_sn),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """ Generator network. """

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, use_sn=False):
        super().__init__()

        self.head = nn.Sequential(
            Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False, use_sn=use_sn),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Down-sampling layers.
        curr_dim = conv_dim

        self.down_sampling = nn.ModuleList([
            nn.Sequential(
                Conv2d(curr_dim * (2 ** i), curr_dim * (2 ** (i + 1)), kernel_size=4, stride=2, padding=1, bias=False,
                       use_sn=use_sn),
                nn.InstanceNorm2d(curr_dim * (2 ** (i + 1)), affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(2)
        ])

        # Bottleneck layers.
        curr_dim *= 4
        self.bottleneck = nn.Sequential(*[ResidualBlock(dim_in=curr_dim, dim_out=curr_dim) for _ in range(repeat_num)])

        # Up-sampling layers.
        self.up_sampling = nn.ModuleList([
            nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                Conv2d(2 * curr_dim // (2 ** i), curr_dim // (2 ** (i + 1)),
                       kernel_size=3, padding=1, bias=False, use_sn=use_sn),
                nn.InstanceNorm2d(curr_dim // (2 ** (i + 1)), affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(2)
        ])
        curr_dim = curr_dim // 4

        self.tail = nn.Sequential(
            Conv2d(2 * curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False, use_sn=use_sn),
            nn.Tanh()
        )

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        skips = []

        x = self.head(x)
        skips.append(x.clone())

        for down in self.down_sampling:
            x = down(x)
            skips.append(x.clone())

        skips = skips[::-1]

        x = self.bottleneck(x)

        for i, up in enumerate(self.up_sampling):
            x = torch.cat([x, skips[i]], dim=1)
            x = up(x)

        return self.tail(torch.cat([x, skips[-1]], dim=1))


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, use_sn=True):
        super().__init__()
        layers = [Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, bias=True, use_sn=use_sn),
                  nn.LeakyReLU(0.01)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True, use_sn=use_sn))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False, use_sn=use_sn)
        self.conv2 = Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False, use_sn=use_sn)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
