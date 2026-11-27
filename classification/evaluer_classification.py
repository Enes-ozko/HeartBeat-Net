import os
import sys
import random

import numpy as np
import torch

from UNet_Base_Enes import UNet
from classification_Enes import CLASSES, load_groups, find_scans, heart_features, FeatureClassifier

# ---------------- CONFIG ----------------
UNET_CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else '../semi-supervised/models/mean_teacher_final/last.pth'
CLASSIFIER_CHECKPOINT = sys.argv[2] if len(sys.argv) > 2 else 'models/classifier.pth'
MODE = sys.argv[3] if len(sys.argv) > 3 else 'eval'   # 'eval' (avec groupes.csv) ou 'predict'
DATA_DIR = '/kaggle/input/datasets/enesozkosar/acdc-heartbeat/Data/' if os.path.exists('/kaggle/input') else '../../Data/'
IMG_DIRS = [sys.argv[4]] if len(sys.argv) > 4 else [DATA_DIR + 'train/Img', DATA_DIR + 'train/Img-Unlabeled']   # images patientXXX_YY_ZZ.png
GROUPS_CSV = 'groupes.csv'                     # mode eval : vrai diagnostic de chaque patient
OUT_CSV = 'predictions_diagnostic.csv'         # mode predict : diagnostic prédit par patient
HIDDEN = 32
SEED = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. Créer les modèles : le UNet qui mesure le coeur et l'ensemble de petits réseaux qui classe
def build_models(n_classifiers):
    unet = UNet(num_classes=4).to(device)
    classifiers = [FeatureClassifier(10, HIDDEN).to(device) for _ in range(n_classifiers)]
    n_unet = sum(p.numel() for p in unet.parameters())
    n_clf = sum(p.numel() for c in classifiers for p in c.parameters())
    print('Paramètres : UNet {:,} + {} classifieurs {:,}'.format(n_unet, n_classifiers, n_clf))
    return unet, classifiers


# 2. Charger les checkpoints
def read_checkpoint(path, weights_only=True):
    if not os.path.isfile(path):
        raise FileNotFoundError('Checkpoint introuvable : ' + os.path.abspath(path))
    return torch.load(path, map_location=device, weights_only=weights_only)


def load_checkpoints(unet, classifiers, unet_ckpt, clf_ckpt):
    state = unet_ckpt
    if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
    try:
        unet.load_state_dict(state)
        for c, s in zip(classifiers, clf_ckpt['models']):
            c.load_state_dict(s)
    except RuntimeError as e:
        raise RuntimeError('Un checkpoint ne correspond pas à l\'architecture :\n' + str(e)[:500])
    unet.eval()
    for c in classifiers:
        c.eval()
    return unet, classifiers


# 3. Charger les données : les images ED et ES de chaque patient
def load_patients():
    scans = find_scans([d for d in IMG_DIRS if os.path.isdir(d)])
    scans = {p: s for p, s in scans.items() if len(s) == 2}
    print('Patients avec ED et ES :', len(scans))
    return scans


# 4. Tester
@torch.no_grad()
def test(unet, classifiers, clf_ckpt, scans):
    patients = sorted(scans)
    X = np.array([heart_features(unet, scans[p], device) for p in patients])
    x = torch.tensor((X - clf_ckpt['mean']) / clf_ckpt['std'], dtype=torch.float32, device=device)
    probs = torch.stack([torch.softmax(c(x), dim=1) for c in classifiers]).mean(dim=0).cpu()
    pred = probs.argmax(dim=1).numpy()

    if MODE == 'eval':
        groups = load_groups(GROUPS_CSV)
        known = [i for i, p in enumerate(patients) if p in groups]
        y = np.array([groups[patients[i]] for i in known])
        good = (pred[known] == y).sum()
        print('Bien classés : {} / {} = {:.3f}'.format(good, len(known), good / len(known)))
        for c, name in enumerate(CLASSES):
            if (y == c).any():
                print('{:<5} rappel {:.2f}'.format(name, (pred[known][y == c] == c).mean()))
    else:
        with open(OUT_CSV, 'w') as f:
            f.write('patient,groupe,confiance\n')
            for p, k, pr in zip(patients, pred, probs):
                f.write('{},{},{:.2f}\n'.format(p, CLASSES[k], pr[k].item()))
        print('Diagnostics enregistrés dans', os.path.abspath(OUT_CSV))


if __name__ == '__main__':
    unet_ckpt = read_checkpoint(UNET_CHECKPOINT)
    clf_ckpt = read_checkpoint(CLASSIFIER_CHECKPOINT, weights_only=False)
    unet, classifiers = build_models(len(clf_ckpt['models']))
    unet, classifiers = load_checkpoints(unet, classifiers, unet_ckpt, clf_ckpt)
    scans = load_patients()
    test(unet, classifiers, clf_ckpt, scans)
