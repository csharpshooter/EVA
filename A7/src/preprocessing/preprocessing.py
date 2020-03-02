from torchvision import transforms
import torch


class Preprocessing(object):

    def gettraintransforms(self):
        # Train Phase transformations
        return transforms.Compose([
            transforms.Pad(padding=1, padding_mode="edge"),
            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def gettesttransforms(self):
        # Test Phase transformations
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
