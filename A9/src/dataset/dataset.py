# import src.dataset
from torchvision import datasets


class Dataset(object):
    # def __init__(self, name):
    #     self.name = name
    #     print(name)

    # def gettraindataset(self, train_transforms):
    #     return src.dataset.CustomCifar10(root='data', train=True,
    #                                      download=True, transform=train_transforms)
    #
    # def gettestdataset(self, test_transforms):
    #     return src.dataset.CustomCifar10(root='data', train=False,
    #                                      download=True, transform=test_transforms)

    def gettraindataset(self, train_transforms):
        return datasets.CIFAR10(root='data', train=True,
                                         download=True, transform=train_transforms)

    def gettestdataset(self, test_transforms):
        return datasets.CIFAR10(root='data', train=False,
                                         download=True, transform=test_transforms)
