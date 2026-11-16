import os
from random import random, randint

from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    img_path = os.path.join(root, mode, 'Img')
    mask_path = os.path.join(root, mode, 'GT')

    # On ne garde que les .png (le Mac ajoute des fichiers .DS_Store dans les dossiers)
    images = sorted(f for f in os.listdir(img_path) if f.endswith('.png'))
    labels = sorted(f for f in os.listdir(mask_path) if f.endswith('.png'))

    items = []
    for it_im, it_gt in zip(images, labels):
        items.append((os.path.join(img_path, it_im), os.path.join(mask_path, it_gt)))
    return items


class MedicalImageDataset(Dataset):
    """Images IRM et masques de segmentation (ACDC).

    augment   : active l'augmentation de données (entraînement seulement)
    flip      : retournements haut/bas et gauche/droite
    scale     : changement d'échelle (zoom) et petit décalage
    intensity : luminosité et contraste
    La rotation est toujours faite quand augment=True.
    """

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
        # Règle du cours : chaque transformation géométrique est faite sur l'image ET sur le masque.
        # Image : interpolation bilinéaire. Masque : plus proche voisin, pour garder les classes 0, 1, 2, 3.

        # 1. Retournements (mirroring)
        if self.flip:
            if random() > 0.5:
                img = ImageOps.flip(img)
                mask = ImageOps.flip(mask)
            if random() > 0.5:
                img = ImageOps.mirror(img)
                mask = ImageOps.mirror(mask)

        # 2. Rotation de -30° à +30°
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)

        # 3. Changement d'échelle (zoom de 0.9 à 1.1) et décalage de 10 pixels au plus
        if self.scale and random() > 0.5:
            zoom = 0.9 + random() * 0.2
            dx = randint(-10, 10)
            dy = randint(-10, 10)
            img = TF.affine(img, angle=0, translate=[dx, dy], scale=zoom, shear=[0.0],
                            interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[dx, dy], scale=zoom, shear=[0.0],
                             interpolation=InterpolationMode.NEAREST)

        # 4. Luminosité et contraste de 0.8 à 1.2 : seulement l'image, le masque ne change pas
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
