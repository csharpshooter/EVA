import torch
import numpy as np


class Dataloader(object):

    def __init__(self, traindataset, testdataset, batch_size=16, memory_format=torch.contiguous_format):
        self.traindataset = traindataset
        self.testdataset = testdataset
        self.batch_size = batch_size
        self.num_workers = 4
        self.pin_memory = True
        self.shuffle = True
        self.memory_format = memory_format
        # self.collate_fn = lambda b: self.fast_collate(b, memory_format)
        print(self.batch_size)

    def gettraindataloader(self):
        return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=self.shuffle,
                                           pin_memory=self.pin_memory,
                                           )

    def gettestdataloader(self):
        return torch.utils.data.DataLoader(dataset=self.testdataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=self.shuffle,
                                           pin_memory=self.pin_memory,
                                          )

    def fast_collate(self, batch):
        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        w = imgs[0].size[0]
        h = imgs[0].size[1]
        tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            if (nump_array.ndim < 3):
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)

            tensor[i] += torch.from_numpy(nump_array)

        return tensor, targets
