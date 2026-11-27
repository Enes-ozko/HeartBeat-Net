import os
import sys
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from UNet_Base_Enes import UNet
from medicalDataLoader_Enes import MedicalImageDataset, list_png
from utils_Enes import dice_per_class, hausdorff_per_class, asd_per_class, keep_largest_component

# ---------------- CONFIG ----------------
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else 'models/mean_teacher_final/last.pth'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'eval'   # 'eval' (avec masques) ou 'predict' (images seules)
DATA_DIR = '/kaggle/input/datasets/enesozkosar/acdc-heartbeat/Data/' if os.path.exists('/kaggle/input') else '../../Data/'
SPLIT = 'val'                                  # mode eval : dossier DATA_DIR/<SPLIT>/Img + GT
PREDICT_DIR = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA_DIR, 'val', 'Img')   # mode predict : dossier d'images à segmenter
OUT_DIR = 'predictions_test'
BATCH_SIZE = 8
SEED = 0
POST_PROCESS = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. Créer le modèle
def build_model():
    model = UNet(num_classes=4).to(device)
    print('Paramètres : {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model


# 2. Charger le checkpoint
def load_checkpoint(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError('Checkpoint introuvable : ' + os.path.abspath(path))
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    try:
        model.load_state_dict(ckpt)
    except RuntimeError as e:
        raise RuntimeError('Le checkpoint ne correspond pas à cette architecture UNet :\n' + str(e)[:500])
    model.eval()
    print('Checkpoint chargé :', path)
    return model


# 3. Charger les données (sans augmentation, ordre fixe)
class ImageFolder(Dataset):
    def __init__(self, folder):
        self.paths = list_png(folder)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF.to_tensor(Image.open(self.paths[i]).convert('L')), os.path.basename(self.paths[i])


def build_loader():
    if MODE == 'eval':
        dataset = MedicalImageDataset(SPLIT, DATA_DIR, augment=False)
    else:
        dataset = ImageFolder(PREDICT_DIR)
    print('Images :', len(dataset))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. Tester
@torch.no_grad()
def test(model, loader):
    preds, targets = [], []
    if MODE == 'predict':
        os.makedirs(OUT_DIR, exist_ok=True)
    for images, second in loader:
        pred = model(images.to(device)).argmax(dim=1).cpu()
        if POST_PROCESS:
            pred = keep_largest_component(pred)
        if MODE == 'eval':
            preds.append(pred)
            targets.append(second)
        else:
            for p, name in zip(pred, second):
                Image.fromarray((p.numpy() * 85).astype(np.uint8)).save(os.path.join(OUT_DIR, name))

    if MODE == 'eval':
        pred, target = torch.cat(preds), torch.cat(targets)
        dice = dice_per_class(pred, target)
        hd = hausdorff_per_class(pred, target)
        asd = asd_per_class(pred, target)
        print('Dice VD {:.3f} Myo {:.3f} VG {:.3f} | moyenne {:.3f}'.format(*dice, np.mean(dice)))
        print('Hausdorff (px) VD {:.1f} Myo {:.1f} VG {:.1f}'.format(*hd))
        print('ASD (px)       VD {:.2f} Myo {:.2f} VG {:.2f}'.format(*asd))
    else:
        print('Masques enregistrés dans', os.path.abspath(OUT_DIR))


if __name__ == '__main__':
    model = build_model()
    model = load_checkpoint(model, CHECKPOINT)
    loader = build_loader()
    test(model, loader)
