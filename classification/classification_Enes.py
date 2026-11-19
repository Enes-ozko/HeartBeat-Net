import os
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ['NOR', 'MINF', 'DCM', 'HCM', 'RV']


def load_groups(csv_path):
    groups = {}
    for line in open(csv_path).read().split()[1:]:
        patient, group = line.split(',')
        groups[int(patient)] = CLASSES.index(group)
    return groups


def find_scans(folders):
    scans = {}
    for folder in folders:
        for f in os.listdir(folder):
            if f.endswith('.png'):
                patient, phase, z = f[:-4].split('_')
                scans.setdefault(int(patient[7:]), {}).setdefault(int(phase), {})[int(z)] = os.path.join(folder, f)
    return scans


def make_pairs(scans, patients):
    pairs = []
    for p in patients:
        ed, es = sorted(scans[p])
        for z in sorted(scans[p][ed]):
            pairs.append((p, scans[p][ed][z], scans[p][es][z]))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs, groups, augment=False):
        self.pairs = pairs
        self.groups = groups
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p, ed_path, es_path = self.pairs[i]
        ed = Image.open(ed_path).convert('L')
        es = Image.open(es_path).convert('L')
        if self.augment:
            angle = random.uniform(-15, 15)
            ed = ed.rotate(angle, resample=Image.BILINEAR)
            es = es.rotate(angle, resample=Image.BILINEAR)
        return TF.to_tensor(ed), TF.to_tensor(es), self.groups[p], p


class Classifier(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.Sequential(unet.enc1, unet.enc2, unet.enc3, unet.enc4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * 256, len(CLASSES)))

    def features(self, x):
        return self.pool(self.encoder(x)).flatten(1)

    def forward(self, ed, es):
        return self.head(torch.cat([self.features(ed), self.features(es)], dim=1))
