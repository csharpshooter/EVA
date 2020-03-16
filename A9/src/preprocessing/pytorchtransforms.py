from torchvision import transforms
import torch


class PytorchTransforms(object):

    def gettraintransforms(self):
        # Train Phase transformations
        return transforms.Compose([
            transforms.Pad(padding=1, padding_mode="edge"),
            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
            transforms.RandomRotation(20),
            # transforms.RandomErasing(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def gettesttransforms(self):
        # Test Phase transformations
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
