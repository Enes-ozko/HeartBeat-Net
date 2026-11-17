import os
from random import random, randint

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    img_path = os.path.join(root, mode, 'Img')
    mask_path = os.path.join(root, mode, 'GT')

    images = sorted(f for f in os.listdir(img_path) if f.endswith('.png'))
    labels = sorted(f for f in os.listdir(mask_path) if f.endswith('.png'))

    items = []
    for it_im, it_gt in zip(images, labels):
        items.append((os.path.join(img_path, it_im), os.path.join(mask_path, it_gt)))
    return items


def make_unlabeled(root):
    img_path = os.path.join(root, 'train', 'Img-Unlabeled')
    images = sorted(f for f in os.listdir(img_path) if f.endswith('.png'))
    return [os.path.join(img_path, f) for f in images]


class MedicalImageDataset(Dataset):
    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False,
                 equalize=False, flip=True, scale=True, intensity=True):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.flip = flip
        self.scale = scale
        self.intensity = intensity
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if self.flip:
            if random() > 0.5:
                img = ImageOps.flip(img)
                mask = ImageOps.flip(mask)
            if random() > 0.5:
                img = ImageOps.mirror(img)
                mask = ImageOps.mirror(mask)

        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)

        if self.scale and random() > 0.5:
            zoom = 0.9 + random() * 0.2
            dx = randint(-10, 10)
            dy = randint(-10, 10)
            img = TF.affine(img, angle=0, translate=[dx, dy], scale=zoom, shear=[0.0],
                            interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[dx, dy], scale=zoom, shear=[0.0],
                             interpolation=InterpolationMode.NEAREST)

        if self.intensity and random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(0.8 + random() * 0.4)
            img = ImageEnhance.Contrast(img).enhance(0.8 + random() * 0.4)

        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return [img, mask, img_path]


class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.imgs = make_unlabeled(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('L')
        if self.transform:
            img = self.transform(img)
        return [img, self.imgs[index]]


class PseudoLabelDataset(MedicalImageDataset):
    def __init__(self, root_dir, masks, transform=None, augment=False, flip=True, scale=True, intensity=True):
        self.imgs = make_unlabeled(root_dir)
        self.masks = masks
        self.transform = transform
        self.augmentation = augment
        self.flip = flip
        self.scale = scale
        self.intensity = intensity

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('L')
        mask = Image.fromarray(self.masks[index])
        if self.augmentation:
            img, mask = self.augment(img, mask)
        if self.transform:
            img = self.transform(img)
        target = torch.from_numpy(np.array(mask)).long()
        return [img, target, self.imgs[index]]
