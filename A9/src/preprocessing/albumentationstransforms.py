from albumentations import Compose, ElasticTransform, Flip, CoarseDropout, RandomCrop, pytorch, Normalize, Resize, \
    HorizontalFlip, Rotate, PadIfNeeded, CenterCrop, Cutout, ChannelDropout


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std):
        # Train Phase transformations

        albumentations_transform = Compose([
            # PadIfNeeded(42, 42, always_apply=True),
            HorizontalFlip(always_apply=True),
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
