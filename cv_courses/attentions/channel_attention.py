import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)

        return x * y


class CNNWithChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.ca = ChannelAttention(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x)
        x = self.conv2(x)

        return x


def main():
    cnn = CNNWithChannelAttention()
    x = torch.randn(1, 3, 32, 32)
    out = cnn(x)
    print(out.shape)


if __name__ == "__main__":
    main()
