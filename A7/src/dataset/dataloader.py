import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class Cifar10Dataloader(object):

    def __init__(self, traindataset, testdataset):
        self.traindataset = traindataset
        self.testdataset = testdataset

        # number of subprocesses to use for data loading
        self.num_workers = 0
        # how many samples per batch to load
        self.batch_size = 64
        # percentage of training set to use as validation
        valid_size = 0.2

        seed = 1
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        if cuda:
            batch_size = 128
            num_workers = 4
            pin_memory = True
        else:
            shuffle = True
            batch_size = 64

        # obtain training indices that will be used for validation
        num_train = len(self.traindataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)  # For reproducibility
        split = int(np.floor(valid_size * num_train))
        train_idx, test_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def gettraindataloader(self):
        return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,
                                           sampler=self.train_sampler, num_workers=self.num_workers)

    def gettestdataloader(self):
        return torch.utils.data.DataLoader(dataset=self.testdataset, batch_size=self.batch_size,
                                           sampler=self.test_sampler, num_workers=self.num_workers)
