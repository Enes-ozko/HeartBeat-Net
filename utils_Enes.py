import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion, label
from scipy.spatial.distance import directed_hausdorff

CLASS_CMAP = ListedColormap(['black', 'tab:red', 'tab:green', 'tab:blue'])


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def getTargetSegmentation(batch):
    denom = 0.33333334
    return (batch / denom).round().long().squeeze(1)


def dice_per_class(pred, target):
    dices = []
    for c in [1, 2, 3]:
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        total = p.sum().item() + t.sum().item()
        if total == 0:
            dices.append(1.0)
        else:
            dices.append(2 * inter / total)
    return dices


def hausdorff_per_class(pred, target):
    pred, target = pred.numpy(), target.numpy()
    result = []
    for c in [1, 2, 3]:
        distances = []
        for p_img, t_img in zip(pred, target):
            p = (p_img == c)
            t = (t_img == c)
            if not p.any() or not t.any():
                continue
            p_border = np.argwhere(p & ~binary_erosion(p))
            t_border = np.argwhere(t & ~binary_erosion(t))
            d1 = directed_hausdorff(p_border, t_border)[0]
            d2 = directed_hausdorff(t_border, p_border)[0]
            distances.append(max(d1, d2))
        if distances:
            result.append(np.mean(distances))
        else:
            result.append(float('nan'))
    return result


def keep_largest_component(pred):
    pred = pred.clone()
    for k in range(len(pred)):
        for c in [1, 2, 3]:
            zones, n = label((pred[k] == c).numpy())
            if n > 1:
                sizes = [(zones == i).sum() for i in range(1, n + 1)]
                biggest = np.argmax(sizes) + 1
                small = (zones > 0) & (zones != biggest)
                pred[k][torch.from_numpy(small)] = 0
    return pred


@torch.no_grad()
def validate(net, loader, loss_fn, with_hausdorff=True, post_process=False):
    net.eval()
    losses, preds, targets = [], [], []
    for images, labels, _ in loader:
        images = to_var(images)
        target = getTargetSegmentation(to_var(labels))
        out = net(images)
        losses.append(loss_fn(out, target).item())
        preds.append(out.argmax(dim=1).cpu())
        targets.append(target.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    if post_process:
        preds = keep_largest_component(preds)
    dsc = dice_per_class(preds, targets)
    hd = hausdorff_per_class(preds, targets) if with_hausdorff else None
    return np.mean(losses), dsc, hd


def format_score(x):
    if np.isnan(x):
        return '-'
    if x <= 1:
        return '{:.2f}'.format(x)
    return '{:.0f}'.format(x)


@torch.no_grad()
def show_predictions(net, loader, n=4, title='', post_process=False):
    net.eval()
    images, labels, _ = next(iter(loader))
    images, labels = images[:n], labels[:n]
    probs = torch.softmax(net(to_var(images)), dim=1).cpu()
    pred = probs.argmax(dim=1)
    if post_process:
        pred = keep_largest_component(pred)
    target = getTargetSegmentation(labels)

    fig, axes = plt.subplots(n, 4, figsize=(13, 3.4 * n))
    for k in range(n):
        p = pred[k:k + 1]
        t = target[k:k + 1]
        dsc = dice_per_class(p, t)
        hd = hausdorff_per_class(p, t)
        for i, c in enumerate([1, 2, 3]):
            if not (p == c).any() and not (t == c).any():
                dsc[i] = float('nan')

        axes[k, 0].imshow(images[k, 0], cmap='gray')
        axes[k, 1].imshow(target[k], cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation='nearest')
        axes[k, 2].imshow(pred[k], cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation='nearest')
        axes[k, 2].set_xlabel('Dice  VD {}  Myo {}  VG {}\nHD    VD {}  Myo {}  VG {}'.format(
            *[format_score(d) for d in dsc], *[format_score(h) for h in hd]),
            fontsize=9, family='monospace')
        im = axes[k, 3].imshow(probs[k, 0], cmap='magma', vmin=0, vmax=1)
        for ax in axes[k]:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, name in zip(axes[0], ['Image', 'Vrai masque', 'Prédiction', 'Probabilité du fond']):
        ax.set_title(name)
    fig.colorbar(im, ax=axes[:, 3], shrink=0.6)
    legend = [Patch(color=CLASS_CMAP(0), label='Fond'), Patch(color=CLASS_CMAP(1), label='VD'),
              Patch(color=CLASS_CMAP(2), label='Myocarde'), Patch(color=CLASS_CMAP(3), label='VG')]
    fig.legend(handles=legend, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.45, 0.965))
    fig.suptitle(title, y=0.995)
    plt.show()


def show_augmentations(dataset, index=0, n=5):
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    for k in range(n):
        img, mask, _ = dataset[index]
        target = getTargetSegmentation(mask.unsqueeze(0))[0]
        axes[0, k].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        axes[1, k].imshow(target, cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation='nearest')
        axes[0, k].set_title('Tirage {}'.format(k + 1))
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel('Image')
    axes[1, 0].set_ylabel('Masque')
    fig.suptitle('Augmentation de données : une même image, {} tirages'.format(n))
    plt.show()


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * inter + 1e-6) / (total + 1e-6)
        return 1 - dice[1:].mean()


@torch.no_grad()
def show_confusion(net, loader, title='', min_pixels=50, post_process=False):
    net.eval()
    names = ['Fond', 'VD', 'Myo', 'VG']
    matrix = np.zeros((4, 4))
    found, missed, false_alarm, empty_ok = 0, 0, 0, 0

    for images, labels, _ in loader:
        pred = net(to_var(images)).argmax(dim=1).cpu()
        if post_process:
            pred = keep_largest_component(pred)
        target = getTargetSegmentation(labels)
        for t in range(4):
            for p in range(4):
                matrix[t, p] += ((target == t) & (pred == p)).sum().item()
        for k in range(len(target)):
            heart_true = (target[k] > 0).any().item()
            heart_pred = (pred[k] > 0).sum().item() >= min_pixels
            if heart_true and heart_pred:
                found += 1
            elif heart_true:
                missed += 1
            elif heart_pred:
                false_alarm += 1
            else:
                empty_ok += 1

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    for t in range(4):
        for p in range(4):
            color = 'white' if matrix[t, p] > 0.5 else 'black'
            ax.text(p, t, '{:.1f} %'.format(100 * matrix[t, p]), ha='center', va='center', color=color)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Vraie classe')
    ax.set_title('Matrice de confusion par pixel ' + title)
    plt.tight_layout()
    plt.show()

    print('Détection du cœur par image :')
    print('   images avec cœur  : trouvé {} / {}  (raté sur {})'.format(found, found + missed, missed))
    print('   images sans cœur  : rien prédit {} / {}  (faux positifs sur {})'.format(
        empty_ok, empty_ok + false_alarm, false_alarm))
