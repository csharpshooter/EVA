import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        images = []
        labels = []
        # load images and masks

        bg_fg = Image.open(self.images[idx])  # .convert("RGB")
        bg = Image.open(self.labels[idx]["bg_path"])  # .convert("RGB")
        mask = Image.open(self.labels[idx]["masks"]).convert("RGB")

        labels.append(self.images[idx])
        labels.append(self.labels[idx]["bg_path"])
        labels.append(self.labels[idx]["masks"])

        if self.transforms is not None:
            bg_fg = self.transforms(bg_fg)

        if self.transforms is not None:
            bg = self.transforms(bg)

        if self.transforms is not None:
            mask = self.transforms(mask)

        images.append(np.array(bg_fg))
        images.append(np.array(bg))
        images.append(np.array(mask))

        # print("fg_bg-{0}".format(bg_fg.shape))
        # print("bg-{0}".format(bg.shape))
        # print("mask-{0}".format(mask.shape))

        return images, labels

    # def __getitem__(self, idx):
    #     # load images ad masks
    #     img_path = self.images[idx]
    #     mask_path = self.labels[idx]["masks"]
    #     img = Image.open(img_path).convert("RGB")
    #     # note that we haven't converted the mask to RGB,
    #     # because each color corresponds to a different instance
    #     # with 0 being background
    #     mask = Image.open(mask_path)
    #
    #     mask = np.array(mask)
    #     # instances are encoded as different colors
    #     obj_ids = np.unique(mask)
    #     # first id is the background, so remove it
    #     obj_ids = obj_ids[1:]
    #
    #     # split the color-encoded mask into a set
    #     # of binary masks
    #     masks = mask == obj_ids[:, None, None]
    #
    #     # get bounding box coordinates for each mask
    #     num_objs = len(obj_ids)
    #     boxes = []
    #     for i in range(num_objs):
    #         pos = np.where(masks[i])
    #         xmin = np.min(pos[1])
    #         xmax = np.max(pos[1])
    #         ymin = np.min(pos[0])
    #         ymax = np.max(pos[0])
    #         boxes.append([xmin, ymin, xmax, ymax])
    #
    #     boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #     # there is only one class
    #     labels = torch.ones((num_objs,), dtype=torch.int64)
    #     masks = torch.as_tensor(masks, dtype=torch.uint8)
    #
    #     image_id = torch.tensor([idx])
    #     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    #     # suppose all instances are not crowd
    #     iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    #
    #     target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
    #               "iscrowd": iscrowd}
    #
    #     if self.transforms is not None:
    #         img, target = self.transforms(img, target)
    #
    #     return img, target

    def __len__(self):
        return len(self.images)


