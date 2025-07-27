# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tv

from typing import Sequence
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


class Transform:
    def __call__(self, x):
        raise NotImplementedError

class RandomCrop(Transform):
    def __init__(self, shape, device=f"cuda", dtype=torch.float32):
        self.shape = shape
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if len(image.shape) == 4:
            _, d, h, w = image.shape
        else:
            d, h, w = image.shape
        ds, hs, ws = self.shape
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        image = image.to(self.device, dtype=self.dtype)
        pad_w = max(0, ws-w)
        pad_h = max(0, hs-h)
        pad_d = max(0, ds-d)
        # pad = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2)
        image = F.pad(image, padding)

        d0 = np.random.randint(0, d-ds+1) if pad_d == 0 else 0
        h0 = np.random.randint(0, h-hs+1) if pad_h == 0 else 0
        w0 = np.random.randint(0, w-ws+1) if pad_w == 0 else 0

        image = image[..., d0:d0+ds, h0:h0+hs, w0:w0+ws]
        image = image if len(image.shape) == 4 else image[None, ...]

        x['image'] = image
        return x
    
class RandomResizedCrop(Transform):
    def __init__(self, shape, scale_min, scale_max, device=f"cuda", dtype=torch.float32):
        self.shape = shape
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        crop_shape = (np.array(image.shape[-3:]) * np.random.uniform(self.scale_min ** (1/3), self.scale_max ** (1/3))).astype(np.int32).tolist()
        x = RandomCrop(crop_shape, self.device, self.dtype)(x)
        image = x['image']
        image = F.interpolate(image[None, ...], self.shape)#[0]
        x['image'] = image[0]
        return x

class RandomFlip(Transform):
    def __init__(self, p=0.5, device="cuda", dtype=torch.float32):
        self.p = p
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)
        to_flip = []
        for i in range(1, 4):
            if np.random.random() < self.p:
                to_flip.append(-i)
        image = image.flip(to_flip)
        image = image if len(image.shape) == 4 else image[None, ...]

        x['image'] = image
        return x

class RandomGaussianBlurOrSharpen(Transform):
    def __init__(
        self,
        p=0.8,
        device="cuda",
        dtype=torch.float32,
        voxel_spacing=np.array([1, 1, 1]),
        sigma_high=1.5,
        alpha_high=2.0,
        do_sharpen=True
    ):
        self.p = p
        self.device = device
        self.dtype = dtype
        self.voxel_spacing = voxel_spacing
        self.sigma_high = sigma_high
        self.alpha_high = alpha_high
        self.do_sharpen = do_sharpen

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)
        sigma = (np.random.uniform(0.0, self.sigma_high) / self.voxel_spacing[:2]).tolist()
        image_blurred = tv.gaussian_blur(image, kernel_size=[7, 7], sigma=sigma)
        
        image_ret = None
        if np.random.random() < self.p:
            image_ret = image_blurred
        else:
            if self.do_sharpen:
                alpha = np.random.uniform(0.0, self.alpha_high)
                image = image + alpha * (image - image_blurred)
                image_ret = image
            else:
                image_ret = image
            
        
        x['image'] = image_ret
        return x

class RandomGaussianNoise(Transform):
    def __init__(self, p=0.8, device="cuda", dtype=torch.float32):
        self.p = p
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)
        if np.random.uniform() < self.p:
            noise_sigma = np.random.uniform(0.0, 0.1)
            image = image + torch.randn_like(image) * noise_sigma
        x['image'] = image
        return x

class RandomBrigtness(Transform):
    def __init__(
        self, p=0.8,
        device="cuda",
        dtype=torch.float32,
        brightntess_low=0.8,
        brightness_high=1.2
    ):
        self.p = p
        self.device = device
        self.dtype = dtype
        self.brightness_low = brightntess_low
        self.brightness_high = brightness_high

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)

        if np.random.uniform() < self.p:
            brightness_factor = np.random.uniform(self.brightness_low, self.brightness_high)
            image = tv.adjust_brightness(image.transpose(0, 1), brightness_factor).transpose(0, 1)
        
        x['image'] = image
        return x
    
class RandomContrast(Transform):
    def __init__(
        self,
        p=0.8,
        device="cuda",
        dtype=torch.float32,
        contrast_low=0.8,
        contrast_high=1.2
    ):
        self.p = p
        self.device = device
        self.dtype = dtype
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high


    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)

        if np.random.uniform() < self.p:
            contrast_factor = np.random.uniform(self.contrast_low, self.contrast_high)
            image = tv.adjust_contrast(image.transpose(0, 1), contrast_factor).transpose(0, 1)
        x['image'] = image
        return x
    
class RandomGamma(Transform):
    def __init__(self, p=0.8, device="cuda", dtype=torch.float32):
        self.p = p
        self.device = device
        self.dtype = dtype
    
    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device, dtype=self.dtype)

        if np.random.uniform() < self.p:
            gamma = np.random.uniform(0.8, 1.25)
            eps = 1e-3
            image = tv.adjust_gamma(torch.clamp(image.transpose(0, 1), eps, 1 - eps), gamma).transpose(0, 1)
        x['image'] = image
        return x
    
class RandomRotation90(Transform):
    def __init__(self, p=0.5, device="cuda", dtype=torch.float32):
        self.p = p
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x):
        image = x['image']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.to(self.device, dtype=self.dtype)

        if np.random.uniform() < self.p:
            k = np.random.randint(1, 4)
            image = image.rot90(k, (1, 2))
        x['image'] = image
        return x

def identity(x):
    return x

class TransformCompose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms
    
    @torch.no_grad()
    def __call__(self, x):
        ret = self.transforms[0](x)
        for transform in self.transforms:
            ret = transform(ret)
        return ret
