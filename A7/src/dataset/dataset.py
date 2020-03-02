from torchvision import datasets

def Dataset():

    def __init__(self):

        def gettraindataset(train_transforms):
            return  datasets.CIFAR10('data', train=True,
                                      download=True, transform=train_transforms)


        def gettraindataset(test_transforms):
            return  datasets.CIFAR10('data', train=False,
                                     download=True, transform=test_transforms)
