import torch
import torch.nn as nn

# class DepthModel(nn.Module):
#
#     def __init__(self):
#         super(DepthModel, self).__init__()
#
#         self.inputblock = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         )
#
#         self.convblock2 = nn.Sequential(
#             # Defining a 2D convolution layer
#             nn.Conv2d(16, 32, 3, stride=1, bias=False, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#         )
#
#         self.convblock3 = nn.Sequential(
#             # Defining a 2D convolution layer
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#
#         self.convblock4 = nn.Sequential(
#             # Defining a 2D convolution layer
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#
#         self.convblockfinal = nn.Sequential(
#             # Defining a 2D convolution layer
#             nn.Conv2d(512, 3, 3, stride=1, bias=False, padding=1),
#         )
#
#     def forward(self, data):
#         x1 = self.convblock2(self.inputblock(data[0]))
#         x2 = self.convblock2(self.inputblock(data[1]))
#
#         final_x = torch.cat([x1, x2], 1)
#
#         final_x = self.convblock4(self.convblock3(final_x))
#
#         return final_x
