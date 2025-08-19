# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import logging
import numpy as np

from copy import deepcopy
from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
    RandomCrop,
    RandomResizedCrop,
    RandomFlip,
    RandomBrigtness,
    RandomContrast,
    RandomGamma,
    RandomGaussianBlurOrSharpen,
    RandomGaussianNoise,
    RandomRotation90,
    identity,
    TransformCompose
)

logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

    
class DataAugmentationDINO3D(object):
    def __init__(
        self,
        local_crops_number,
        initial_crop_size=256,
        global_crops_size=128,
        local_crops_size=64,
        batch_size=1,
        device=f"cuda",
        dtype=torch.float32,
        use_global=True,
        use_local=True,
        use_color=True
    ):
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")


        self.geometric_augmentation_initial = RandomCrop(
            (initial_crop_size, initial_crop_size, initial_crop_size),
            device=self.device, dtype=self.dtype)
        
        self.geometric_augmentation_global = RandomResizedCrop(
            (global_crops_size, global_crops_size, global_crops_size),
            0.32, 1., device=self.device, dtype=self.dtype
        ) if use_global else identity
        self.geometric_augmentation_local = RandomResizedCrop(
            (local_crops_size, local_crops_size, local_crops_size), 
            0.05, 0.32, device=self.device, dtype=self.dtype
        ) if use_local else identity
        # self.global_transfo1 = identity
        # self.global_transfo2 = identity
        # self.local_transfo = identity
        self.global_transfo1 = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype),
                RandomContrast(p=0.8, device=self.device, dtype=self.dtype),
                RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity
        self.global_transfo2 = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype),
                RandomContrast(p=0.8, device=self.device, dtype=self.dtype),
                RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity
        self.local_transfo = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype),
                RandomContrast(p=0.8, device=self.device, dtype=self.dtype),
                RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity

    def __call__(self, x):
        x = self.geometric_augmentation_initial(x)
        out = []
        for _ in range(self.batch_size):
            output = {}
            # global crops:
            im1_base = self.geometric_augmentation_global(deepcopy(x))#
            global_crop_1 = self.global_transfo1(im1_base)['image']

            im2_base = self.geometric_augmentation_global(deepcopy(x))#
            global_crop_2 = self.global_transfo2(im2_base)['image']

            output["global_crops"] = [global_crop_1, global_crop_2]

            # global crops for teacher:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

            # local crops:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(deepcopy(x)))['image'] for _ in range(self.local_crops_number)#
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()
            out.append(output)

        return out[0] if self.batch_size == 1 else out


class DataAugmentationDINO3DOpenmind(object):
    def __init__(
        self,
        local_crops_number,
        initial_crop_size=256,
        global_crops_size=128,
        local_crops_size=64,
        batch_size=1,
        device=f"cuda",
        dtype=torch.float32,
        use_global=True,
        use_local=True,
        use_color=True
    ):
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")


        self.geometric_augmentation_initial = RandomCrop(
            (initial_crop_size, initial_crop_size, initial_crop_size),
            device=self.device, dtype=self.dtype)
        
        self.geometric_augmentation_global = RandomResizedCrop(
            (global_crops_size, global_crops_size, global_crops_size),
            0.32, 1., device=self.device, dtype=self.dtype
        ) if use_global else identity
        self.geometric_augmentation_local = RandomResizedCrop(
            (local_crops_size, local_crops_size, local_crops_size), 
            0.05, 0.32, device=self.device, dtype=self.dtype
        ) if use_local else identity
        # self.global_transfo1 = identity
        # self.global_transfo2 = identity
        # self.local_transfo = identity
        self.global_transfo1 = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype, sigma_high=1.5, do_sharpen=False),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                # RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype, brightntess_low=0.1, brightness_high=0.3),
                # RandomContrast(p=0.8, device=self.device, dtype=self.dtype, contrast_low=0.6, contrast_high=0.7),
                # RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity
        self.global_transfo2 = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype, sigma_high=1.5, do_sharpen=False),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                # RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype, brightntess_low=0.1, brightness_high=0.3),
                # RandomContrast(p=0.8, device=self.device, dtype=self.dtype, contrast_low=0.6, contrast_high=0.7),
                # RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity
        self.local_transfo = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype, sigma_high=1.5, do_sharpen=False),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                # RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype, brightntess_low=0.1, brightness_high=0.3),
                # RandomContrast(p=0.8, device=self.device, dtype=self.dtype, contrast_low=0.6, contrast_high=0.7),
                # RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        ) if use_color else identity

    def __call__(self, x):
        x = self.geometric_augmentation_initial(x)
        out = []
        for _ in range(self.batch_size):
            output = {}
            # global crops:
            im1_base = self.geometric_augmentation_global(deepcopy(x))#
            global_crop_1 = self.global_transfo1(im1_base)['image']

            im2_base = self.geometric_augmentation_global(deepcopy(x))#
            global_crop_2 = self.global_transfo2(im2_base)['image']

            output["global_crops"] = [global_crop_1, global_crop_2]

            # global crops for teacher:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

            # local crops:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(deepcopy(x)))['image'] for _ in range(self.local_crops_number)#
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()
            out.append(output)

        return out[0] if self.batch_size == 1 else out
    

class DataAugmentation3DForClassification(object):
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.transform = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype),
                RandomContrast(p=0.8, device=self.device, dtype=self.dtype),
                RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        )

    def __call__(self, x):
        x = deepcopy(x) #copy?
        image = x['image']
        if len(image.shape) == 3:
            image = image[None, ...]
        x['image'] = image
        return self.transform(x)
    

class DataAugmentation3DForClassificationOpenmind(object):
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.transform = TransformCompose(
            [
                RandomFlip(p=0.5, device=self.device, dtype=self.dtype),
                RandomRotation90(p=0.5, device=self.device, dtype=self.dtype),
                RandomGaussianBlurOrSharpen(p=0.8, device=self.device, dtype=self.dtype, sigma_high=1.5, do_sharpen=False),
                RandomGaussianNoise(p=0.8, device=self.device, dtype=self.dtype),
                # RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype, brightntess_low=0.1, brightness_high=0.3),
                # # RandomContrast(p=0.8, device=self.device, dtype=self.dtype, contrast_low=0.6, contrast_high=0.7),
                # RandomGamma(p=0.8, device=self.device, dtype=self.dtype)
            ]
        )

    def __call__(self, x):
        x = deepcopy(x) #copy?
        image = x['image']
        if len(image.shape) == 3:
            image = image[None, ...]
        x['image'] = image
        return self.transform(x)


class DataAugmentation3DForClassificationVal(object):
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        x = deepcopy(x)
        image = x['image']
        if len(image.shape) == 3:
            image = image[None, ...]
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).to(self.device, dtype=self.dtype)
        x['image'] = image
        return x


class DataAugmentation3DForClassificationValOpenmind(object):
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.transform = TransformCompose(
            [
                # RandomBrigtness(p=0.8, device=self.device, dtype=self.dtype, brightntess_low=0.1, brightness_high=0.3)
                identity
            ]
        )
    def __call__(self, x):
        x = deepcopy(x)
        image = x['image']
        if len(image.shape) == 3:
            image = image[None, ...]
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).to(self.device, dtype=self.dtype)
        x['image'] = image
        return self.transform(x)       



