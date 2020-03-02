import torch
import torch.nn as nn
import torch.nn.functional as F

class FourConvBlockNet(nn.Module):
    def __init__(self):
          super(FourConvBlockNet, self).__init__()

    def forward(self, x):
        x = ConvBlock(1, x)
        x = Maxpool(x)
        x = ConvBlock(1, x)
        x = Maxpool(x)
        x = ConvBlock(0, x)
        x = Gap(2, x)
        x = FC(256, 10)
        return x


def ConvBlock(padding, x):
    return nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=padding),   #RF = 3 - 16 - 42
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=padding), #RF = 5 - 20 - 50
        nn.BatchNorm2d(34),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, dilation=1, bias=False, padding=padding), #RF = 7 - 24 - 58
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.SeparableConv2d(128, 256, kernel_size=3, stride=1, dilation=2, bias=False, padding=padding), #RF = 11 - 32 - 70
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
    )(x)

def Maxpool(x):
    return nn.MaxPool2d(kernel_size=2, stride=2)(x) #RF = 12 - 34

def Gap(kernel_size, x):
    return nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size))(x) #RF = 70

def FC(in_features,x,num_classes = 10):
    return nn.Linear(in_features, num_classes, bias=False)(x) #RF = 70
