import os

import numpy as np
import torch.utils.data
from PIL import Image
from tqdm import trange, tqdm
import asyncio
import nest_asyncio
from PIL import ImageFile


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, ds_type, preload=False):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = images
        self.labels = labels
        self.cache_bg_fg = []
        self.cache_mask = []
        self.cache_dm = []
        self.cache_bg = {}
        self.preload = preload
        self.ds_type = ds_type
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self.preload:
            asyncio.ensure_future(self.preload_mask())
            # asyncio.ensure_future(self.preload_dm())
            asyncio.ensure_future(self.preload_bg())
            asyncio.ensure_future(self.preload_bg_fg())
            # loop = asyncio.new_event_loop()
            # preload_bg_fg = loop.create_task(self.preload_bg_fg())
            # # preload_mask = loop.create_task(self.preload_mask())
            # # preload_dm = loop.create_task(self.preload_dm())
            # preload_bg = loop.create_task(self.preload_bg())
            # loop.run_until_complete(asyncio.wait([preload_bg_fg, preload_bg]))
            # # loop.close()

        # (self.preload_dataset())

    def __getitem__(self, idx):
        images = []
        labels = []
        bg_fg = None
        bg = None
        mask = None
        dm = None
        # load images and masks

        # print(len(self.cache))

        if self.preload:
            bg_fg = self.cache_bg_fg[idx]  # .convert("RGB")
            # print(self.labels[idx]["labels"].split('_')[0])
            bg = self.cache_bg[self.labels[idx]["labels"].split('_')[0]]  # .convert("RGB")
            mask = self.cache_mask[idx].convert("RGB")
            # dm = self.cache_dm[idx]
        else:
            bg_fg = Image.open(self.images[idx])  # .convert("RGB")
            bg = Image.open(self.labels[idx]["bg_path"])  # .convert("RGB")
            mask = Image.open(self.labels[idx]["masks"]).convert("RGB")
            # dm = Image.open(self.labels[idx]["depth_mask"])

        # bg_fg = Image.open(self.images[idx])  # .convert("RGB")
        # bg = Image.open(self.labels[idx]["bg_path"])  # .convert("RGB")
        # mask = Image.open(self.labels[idx]["masks"]).convert("RGB")
        # dm = Image.open(self.labels[idx]["depth_mask"])

        if self.transforms is not None:
            bg_fg = self.transforms(bg_fg)

        if self.transforms is not None:
            bg = self.transforms(bg)

        if self.transforms is not None:
            mask = self.transforms(mask)

        # if self.transforms is not None:
        #     dm = self.transforms(dm)

        images.append(np.array(bg_fg, np.float32))
        images.append(np.array(bg, np.float32))
        images.append(np.array(mask, np.float32))
        # images.append(np.array(dm, np.float32))

        labels.append(self.images[idx])
        labels.append(self.labels[idx]["bg_path"])
        labels.append(self.labels[idx]["masks"])
        # labels.append(self.labels[idx]["depth_mask"])

        return images, labels

    def __len__(self):
        return len(self.images)

    async def preload_bg_fg(self):
        print("\nPreloading bg_fg from " + self.ds_type + " dataset...")
        async for idx in AsyncIterator(tqdm(self.images)):
            bg_fg = Image.open(idx)
            self.cache_bg_fg.append(bg_fg)
            # if len(self.cache_bg_fg) > 16:
            #     return

    async def preload_mask(self):
        print("\nPreloading masks from " + self.ds_type + " dataset...")
        async for idx in AsyncIterator(tqdm(self.labels)):
            mask = Image.open(idx["masks"])
            self.cache_mask.append(mask)
            # if len(self.cache_mask) > 16:
            #     return

    async def preload_dm(self):
        print("\nPreloading depth maps from " + self.ds_type + " dataset...")
        async for idx in AsyncIterator(tqdm(self.labels)):
            dm = Image.open(idx["depth_mask"])  # .convert("RGB")
            self.cache_dm.append(dm)
            # if len(self.cache_dm) > 16:
            #     return

    async def preload_bg(self):
        print("\nPreloading bg from " + self.ds_type + " dataset...")
        from src.utils import Utils
        file_paths, filenames = Utils.get_all_file_paths(self.labels[0]["bg_folder"])
        async for idx in AsyncIterator(tqdm(file_paths)):
            bg = Image.open(idx)
            # print(os.path.splitext(os.path.basename(idx)))
            self.cache_bg[os.path.splitext(os.path.basename(idx))[0]] = bg
            # if len(self.cache_bg) > 16:
            #     return

    def set_transforms(self, transforms=None):
        self.transforms = transforms


class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration
