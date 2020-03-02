import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()

    def forward(self, x):
        x = convblock(1, x)
        x = maxpool(x)
        x = convblock(1, x)
        x = maxpool(x)
        x = convblock(0, x)
        x = gap(2, x)
        x = linear(256, 10)
        return x


def convblock(padding, x):
    return nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=padding),  # RF = 3 - 14 - 36
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=padding),  # RF = 5 - 18 - 44
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.SeparableConv2d(64, 64, kernel_size=3, stride=1, dilation=2, bias=False, padding=padding, groups=64),
                                                                                                # RF = 9 - 26 - 60
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
    )(x)


def maxpool(x):
    return nn.MaxPool2d(kernel_size=2, stride=2)(x)  # RF = 10 - 28


def gap(kernel_size, x):
    return nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size))(x)  # RF = 60


def linear(in_features, x, num_classes=10):
    return nn.Linear(in_features, num_classes, bias=False)(x)  # RF = 60
