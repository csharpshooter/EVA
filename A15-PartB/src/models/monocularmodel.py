import torch.nn as nn
import torch
import torch.nn.functional as F
from .depthwise_seperable_conv2d import DepthwiseSeparableConv2d


class MonocularModel(nn.Module):

    def __init__(self):
        super(MonocularModel, self).__init__()

        self.inputblock = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, bias=False, padding=1),
            # RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.convblock2 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(8, 8, 3, stride=1, bias=False, padding=1, groups=8),
            nn.Conv2d(8, 16, 1, stride=1, bias=False, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.convblock3 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, padding=1, groups=16),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.convblock4 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 3, 3, stride=1, bias=False, padding=1),
        )

    def forward(self, data):
        x1 = self.convblock2(self.inputblock(data[0]))
        x2 = self.convblock2(self.inputblock(data[1]))

        final_x = torch.cat([x1, x2], 1)

        final_x = self.convblock4(self.convblock3(final_x))

        return final_x



    # def __init__(self):
    #     super(MonocularModel, self).__init__()
    #
    #     self.inputblock = nn.Sequential(
    #         # Defining a 2D convolution layer
    #         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False, padding=1),
    #         # RF = 3
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #     )
    #
    #     self.convblock2 = nn.Sequential(
    #         # Defining a 2D convolution layer
    #         nn.Conv2d(16, 32, 3, stride=1, bias=False, padding=1, groups=16),
    #         nn.Conv2d(32, 64, 1, stride=1, bias=False, padding=0),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #     )
    #
    #     self.convblock3 = nn.Sequential(
    #         # Defining a 2D convolution layer
    #         nn.Conv2d(128, 512, kernel_size=3, stride=1, bias=False, padding=1, groups=32),
    #         nn.BatchNorm2d(512),
    #         nn.ReLU(),
    #     )
    #
    #     self.convblock4 = nn.Sequential(
    #         # Defining a 2D convolution layer
    #         nn.Conv2d(512, 3, 3, stride=1, bias=False, padding=1),
    #     )
    #
    # def forward(self, data):
    #     x1 = self.convblock2(self.inputblock(data[0]))
    #     x2 = self.convblock2(self.inputblock(data[1]))
    #
    #     final_x = torch.cat([x1, x2], 1)
    #
    #     final_x = self.convblock4(self.convblock3(final_x))
    #
    #     return final_x
