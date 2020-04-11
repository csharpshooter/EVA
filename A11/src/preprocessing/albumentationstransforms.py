import cv2
from albumentations import Compose, Flip, pytorch, Normalize, OneOf, MotionBlur, MedianBlur, Blur, \
    ShiftScaleRotate, OpticalDistortion, GridDistortion, HueSaturationValue, Cutout, GaussNoise, RandomCrop, PadIfNeeded


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std, p=1):
        # Train Phase transformations

        albumentations_transform = Compose([
            # RandomRotate90(),
            PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            RandomCrop(32, 32, True),
            Flip(),
            GaussNoise(p=0.6, mean=mean),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
            OneOf([
                OpticalDistortion(p=0.4),
                GridDistortion(p=0.2),
            ], p=0.3),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
            Cutout(always_apply=True, num_holes=1, max_h_size=8, max_w_size=8, fill_value=(255 * .6)),
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),

        ], p=p)

        return albumentations_transform;

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return Compose([
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])
