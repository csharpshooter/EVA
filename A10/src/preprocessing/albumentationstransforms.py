from albumentations import Compose, Flip, pytorch, Normalize, RandomRotate90, OneOf, MotionBlur, MedianBlur, Blur, \
    ShiftScaleRotate, ToFloat, OpticalDistortion, GridDistortion, HueSaturationValue, Cutout, FromFloat


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std, p=1):
        # Train Phase transformations

        # albumentations_transform = Compose([
        #     # PadIfNeeded(40, 40, always_apply=True, border_mode=cv2.BORDER_REFLECT),
        #     HorizontalFlip(always_apply=True),
        #     Cutout(always_apply=True, num_holes=2, max_h_size=10, max_w_size=10, fill_value=(255 * .5)),
        #     GaussNoise(p=1,mean=mean),
        #     # ChannelDropout(channel_drop_range=(1, 2), always_apply=True, fill_value=(255 * mean[0])),
        #     # CoarseDropout(fill_value=(255 * mean[0])),
        #     Normalize(mean=mean, std=std, always_apply=True),
        #     pytorch.ToTensorV2(always_apply=True),
        #     # GaussianNoise(mean=mean, std=std),
        # ])

        albumentations_transform = Compose([
            # albumentations supports uint8 and float32 inputs. For the latter, all
            # values must lie in the range [0.0, 1.0]. To apply augmentations, we
            # first use a `ToFloat()` transformation, which will inspect the data
            # type of the input image and convert the image to a float32 ndarray where
            # all values lie in the required range [0.0, 1.0].
            # ToFloat(),

            # Alternatively, you can specify the maximum possible value for your input
            # and all values will be divided by it instead of using a predefined value
            # for a specific data type.
            # ToFloat(max_value=65535.0),

            # Then we will apply augmentations
            RandomRotate90(),
            Flip(),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
            ], p=0.2),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
            Cutout(always_apply=True, num_holes=2, max_h_size=10, max_w_size=10, fill_value=(255 * .5)),


            # You can convert the augmented image back to its original
            # data type by using `FromFloat`.
            # FromFloat(dtype='uint16'),

            # As in `ToFloat` you can specify a `max_value` argument and all input values
            # will be multiplied by it.
            # FromFloat(dtype='uint16', max_value=65535.0),
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
