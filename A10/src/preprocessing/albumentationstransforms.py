from albumentations import Compose, ElasticTransform, Flip, CoarseDropout, RandomCrop, pytorch, Normalize, Resize, \
    HorizontalFlip, Rotate, PadIfNeeded, CenterCrop, Cutout, ChannelDropout
import cv2


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std):
        # Train Phase transformations

        albumentations_transform = Compose([
            # PadIfNeeded(40, 40, always_apply=True, border_mode=cv2.BORDER_REFLECT),
            HorizontalFlip(always_apply=True),
            Cutout(always_apply=True, num_holes=2, max_h_size=10, max_w_size=10, fill_value=(255*.5)),
            # ChannelDropout(channel_drop_range=(1, 2), always_apply=True, fill_value=(255 * mean[0])),
            # CoarseDropout(fill_value=(255 * mean[0])),
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])

        return albumentations_transform;

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return Compose([
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])
