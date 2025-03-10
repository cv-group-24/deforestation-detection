import numpy as np
np.bool = np.bool_
import imgaug.augmenters as iaa
import albumentations as A

# Dataset constants
RESIZE_SIZE = 300
RESIZE_SIZE_SENT2 = 450
INPUT_SIZE = 224
INPUT_SIZE_SENT2 = 320
AGGRESSIVE_INPUT_SIZE = 160
AGGRESSIVE_INPUT_SIZE_SENT2 = 256
SMALL_RESIZE_SIZE = 150
SMALL_RESIZE_SIZE_SENT2 = 240
LABEL_IGNORE_VALUE = 255

# Spatial Augmentations
SPATIAL_AUGMENTATIONS = {
    "none": [iaa.Identity()],
    "flip": [iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])],
    "affine": [iaa.SomeOf(2, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Rot90(1, 3)
    ])],
    "cloud": [iaa.Sequential([
        iaa.SomeOf(2, [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Rot90(1, 3)]),
        iaa.Sometimes(0.5, iaa.OneOf([iaa.Clouds(), iaa.Fog(), iaa.Snowflakes()]))
    ])],
    "affine_cloud": [iaa.Sequential([
        iaa.SomeOf(2, [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
            iaa.Affine(translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
            iaa.Rot90(1, 3)
        ]),
        iaa.Sometimes(0.5, iaa.OneOf([iaa.Clouds(), iaa.Fog(), iaa.Snowflakes()]))
    ])],
    "sap": [iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.01]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(rotate=(-5, 5), cval=LABEL_IGNORE_VALUE, mode='constant')
    ])],
    "aggressive": [iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.1]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(rotate=(-5, 5), cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.Affine(shear=(-5, 5), cval=LABEL_IGNORE_VALUE, mode='constant'),
        iaa.ElasticTransformation(alpha=(0.0, 40.0), sigma=(4.0, 8.0))
    ])]
}

# Pixel Augmentations
R_SHIFT_INTERVAL = (6, 16)
G_SHIFT_INTERVAL = (14, 25)
B_SHIFT_INTERVAL = (34, 46)
HAZY_AUGMENTATION = A.RGBShift(r_shift_limit=R_SHIFT_INTERVAL, g_shift_limit=G_SHIFT_INTERVAL, b_shift_limit=B_SHIFT_INTERVAL, p=0.2)

PIXEL_AUGMENTATIONS = {
    "none": [iaa.Identity()],
    "hazy": [HAZY_AUGMENTATION],
    "all": [
        iaa.Sometimes(0.2, iaa.SomeOf((0, 2), [iaa.MultiplyBrightness((0.9, 1.1)), iaa.LogContrast((0.9, 1.1))])),
        HAZY_AUGMENTATION
    ]
}

# Tensor conversion
# TOTENSOR_TRANSFORM = [T.ToTensor()]

DEPLOY_RESIZE = iaa.Resize((INPUT_SIZE, INPUT_SIZE))


def get_resize_crop_transform(resize, is_training, use_landsat):
    if use_landsat:
        input_size = INPUT_SIZE
        aggressive_input_size = AGGRESSIVE_INPUT_SIZE
        resize_size = RESIZE_SIZE
        small_resize_size = SMALL_RESIZE_SIZE
    else:
        input_size = INPUT_SIZE_SENT2
        aggressive_input_size = AGGRESSIVE_INPUT_SIZE_SENT2
        resize_size = RESIZE_SIZE_SENT2
        small_resize_size = SMALL_RESIZE_SIZE_SENT2

    resize_crop_transform_dict = {
        "none": {
            True: [iaa.CropToFixedSize(input_size, input_size)],
            False: [iaa.CenterCropToFixedSize(input_size, input_size)]
        },
        "aggressive": {
            True: [iaa.CropToFixedSize(aggressive_input_size, aggressive_input_size)],
            False: [iaa.CenterCropToFixedSize(aggressive_input_size, aggressive_input_size)]
        },
        "small": {
            True: [
                iaa.Sequential([
                    iaa.Resize((resize_size, resize_size)),
                    iaa.CropToFixedSize(input_size, input_size)
                ])
            ],
            False: [
                iaa.Sequential([
                    iaa.Resize((resize_size, resize_size)),
                    iaa.CenterCropToFixedSize(input_size, input_size)
                ])
            ]
        }
    }

    return resize_crop_transform_dict[resize][is_training]


def get_transforms(resize, spatial_augmentation, pixel_augmentation, is_training, use_landsat):
    transforms = []

    # Resizing
    transforms += get_resize_crop_transform(resize=resize, is_training=is_training, use_landsat=use_landsat)

    # Spatial and pixel augmentation
    if is_training:
        transforms += SPATIAL_AUGMENTATIONS[spatial_augmentation]
        transforms += PIXEL_AUGMENTATIONS[pixel_augmentation]

    return transforms


def valid_crop(crop, mask_bool):
    """
    Check if the cropped region contains any non-masked (valid) pixels.

    Args:
        crop (PIL.Image): The cropped image.
        mask_bool (np.array): The boolean mask array (True for valid pixels, False for masked).

    Returns:
        bool: True if the crop contains non-masked pixels, False if the crop is entirely black.
    """
    crop_np = np.array(crop)
    crop_height, crop_width = crop_np.shape[:2]

    # Check if the crop contains any valid (non-masked) pixels (not black)
    crop_mask = crop_np.sum(axis=-1) > 0  # Non-black pixels in the crop
    valid_pixels = np.sum(crop_mask & mask_bool[:crop_height, :crop_width])

    return valid_pixels > 0

