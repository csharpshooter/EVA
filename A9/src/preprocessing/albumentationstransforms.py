from albumentations import Compose, ElasticTransform, Flip, CoarseDropout, RandomCrop, pytorch, Normalize, Resize, \
    HorizontalFlip, Rotate, PadIfNeeded, CenterCrop, Cutout


class AlbumentaionsTransforms(object):

    def gettraintransforms(self):
        # Train Phase transformations

        albumentations_transform = Compose([
            # PadIfNeeded(36, 36, always_apply=True),
            # Rotate(limit=(20, 20), always_apply=True),
            # HorizontalFlip(always_apply=True),
            # CoarseDropout(max_holes=6, min_holes=6, min_height=8, min_width=8, max_height=16, max_width=16, p=1, ),
            # CenterCrop(32, 32, always_apply=True),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
            # Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            # ElasticTransform(),
            # pytorch.ToTensor(
            #     {"std": [0.24703199, 0.24348481, 0.26158789], "mean": [0.49139765, 0.48215759, 0.44653141]}),
        ])

        return albumentations_transform;

    def gettesttransforms(self):
        # Test Phase transformations
        return Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
            # pytorch.ToTensor(
            #     {"std": [0.24703199, 0.24348481, 0.26158789], "mean": [0.49139765, 0.48215759, 0.44653141]}),
        ])
