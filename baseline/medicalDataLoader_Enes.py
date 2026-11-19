import os
from random import random, randint

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def list_png(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png'))


def random_augment(img, mask):
    if random() > 0.5:
        angle = random() * 60 - 30
        img = img.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)

    if random() > 0.5:
        zoom = 0.9 + random() * 0.2
        shift = [randint(-10, 10), randint(-10, 10)]
        img = TF.affine(img, angle=0, translate=shift, scale=zoom, shear=0, interpolation=InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=0, translate=shift, scale=zoom, shear=0, interpolation=InterpolationMode.NEAREST)

    if random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(0.8 + random() * 0.4)
        img = ImageEnhance.Contrast(img).enhance(0.8 + random() * 0.4)

    return img, mask


class MedicalImageDataset(Dataset):
    def __init__(self, mode, root_dir, augment=False):
        self.imgs = list_png(os.path.join(root_dir, mode, 'Img'))
        self.masks = list_png(os.path.join(root_dir, mode, 'GT'))
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert('L')
        mask = Image.open(self.masks[i]).convert('L')
        if self.augment:
            img, mask = random_augment(img, mask)
        target = torch.from_numpy(np.array(mask) // 85).long()
        return TF.to_tensor(img), target


class UnlabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.imgs = list_png(os.path.join(root_dir, 'train', 'Img-Unlabeled'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert('L')
        return TF.to_tensor(img)


class PseudoLabelDataset(Dataset):
    def __init__(self, root_dir, masks):
        self.imgs = list_png(os.path.join(root_dir, 'train', 'Img-Unlabeled'))
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert('L')
        mask = Image.fromarray(self.masks[i])
        img, mask = random_augment(img, mask)
        target = torch.from_numpy(np.array(mask)).long()
        return TF.to_tensor(img), target
