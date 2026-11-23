import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, label
from scipy.spatial.distance import directed_hausdorff, cdist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['Fond', 'VD', 'Myo', 'VG']


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        valid = (target != 255).unsqueeze(1).float()
        target = target.clone()
        target[target == 255] = 0
        probs = torch.softmax(logits, dim=1) * valid
        one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * valid
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * inter + 1e-6) / (total + 1e-6)
        return 1 - dice[1:].mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', ignore_index=255)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def dice_per_class(pred, target):
    dices = []
    for c in [1, 2, 3]:
        p = (pred == c)
        t = (target == c)
        total = p.sum().item() + t.sum().item()
        if total == 0:
            dices.append(1.0)
        else:
            dices.append(2 * (p & t).sum().item() / total)
    return dices


def hausdorff_per_class(pred, target):
    result = []
    for c in [1, 2, 3]:
        distances = []
        for p, t in zip(pred.numpy() == c, target.numpy() == c):
            if not p.any() or not t.any():
                continue
            p_border = np.argwhere(p & ~binary_erosion(p))
            t_border = np.argwhere(t & ~binary_erosion(t))
            distances.append(max(directed_hausdorff(p_border, t_border)[0],
                                 directed_hausdorff(t_border, p_border)[0]))
        result.append(np.mean(distances) if distances else float('nan'))
    return result


def asd_per_class(pred, target):
    result = []
    for c in [1, 2, 3]:
        distances = []
        for p, t in zip(pred.numpy() == c, target.numpy() == c):
            if not p.any() or not t.any():
                continue
            p_border = np.argwhere(p & ~binary_erosion(p))
            t_border = np.argwhere(t & ~binary_erosion(t))
            d = cdist(p_border, t_border)
            distances.append(np.concatenate([d.min(axis=1), d.min(axis=0)]).mean())
        result.append(np.mean(distances) if distances else float('nan'))
    return result


def keep_largest_component(pred):
    pred = pred.clone()
    for k in range(len(pred)):
        for c in [1, 2, 3]:
            zones, n = label((pred[k] == c).numpy())
            if n > 1:
                sizes = [(zones == i).sum() for i in range(1, n + 1)]
                small = (zones > 0) & (zones != np.argmax(sizes) + 1)
                pred[k][torch.from_numpy(small)] = 0
    return pred


@torch.no_grad()
def predict(net, loader):
    net.eval()
    preds, targets = [], []
    for images, labels in loader:
        preds.append(net(images.to(device)).argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(preds), torch.cat(targets)


def evaluate(net, loader, post_process=False):
    pred, target = predict(net, loader)
    if post_process:
        pred = keep_largest_component(pred)
    return dice_per_class(pred, target), hausdorff_per_class(pred, target)


@torch.no_grad()
def validation_ce(net, loader):
    net.eval()
    losses = []
    for images, labels in loader:
        losses.append(F.cross_entropy(net(images.to(device)), labels.to(device)).item())
    return np.mean(losses)


@torch.no_grad()
def make_pseudo_labels(net, loader, threshold):
    net.eval()
    masks = []
    for images in loader:
        probs = torch.softmax(net(images.to(device)), dim=1).cpu()
        conf, pred = probs.max(dim=1)
        pred = keep_largest_component(pred)
        pred[conf < threshold] = 255
        masks.append(pred.numpy().astype(np.uint8))
    return np.concatenate(masks)


@torch.no_grad()
def show_predictions(net, dataset, indexes, post_process=False):
    net.eval()
    images = torch.stack([dataset[i][0] for i in indexes])
    pred = net(images.to(device)).argmax(dim=1).cpu()
    if post_process:
        pred = keep_largest_component(pred)
    fig, axes = plt.subplots(3, len(indexes), figsize=(3 * len(indexes), 9))
    for k, i in enumerate(indexes):
        axes[0, k].imshow(images[k, 0], cmap='gray')
        axes[1, k].imshow(dataset[i][1], vmin=0, vmax=3)
        axes[2, k].imshow(pred[k], vmin=0, vmax=3)
        for ax in axes[:, k]:
            ax.axis('off')
    axes[0, 0].set_title('Image')
    axes[1, 0].set_title('Vrai masque')
    axes[2, 0].set_title('Prédiction')
    plt.show()


def show_confusion(net, loader, post_process=False):
    pred, target = predict(net, loader)
    if post_process:
        pred = keep_largest_component(pred)
    matrix = np.zeros((4, 4))
    for t in range(4):
        for p in range(4):
            matrix[t, p] = ((target == t) & (pred == p)).sum().item()
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    plt.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    for t in range(4):
        for p in range(4):
            plt.text(p, t, '{:.1f} %'.format(100 * matrix[t, p]), ha='center', va='center')
    plt.xticks(range(4), CLASSES)
    plt.yticks(range(4), CLASSES)
    plt.xlabel('Classe prédite')
    plt.ylabel('Vraie classe')
    plt.show()
