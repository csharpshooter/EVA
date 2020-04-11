from torchvision import transforms

from .cutout import Cutout


class PytorchTransforms(object):

    def gettraintransforms(self, mean, std):
        # Train Phase transformations
        return transforms.Compose([
            transforms.Pad(padding=1, padding_mode="edge"),
            transforms.RandomHorizontalFlip(),  # randomly flip and rotate
            transforms.RandomRotation(20),
            # transforms.RandomErasing(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(1, 8)
        ])

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
