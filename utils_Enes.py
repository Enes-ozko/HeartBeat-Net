import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
#from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial

# from scipy.spatial.distance import directed_hausdorff


labels = {0: 'Background', 1: 'RV', 2: 'Myo', 3: 'LV'}
'''
def computeDSC(pred, gt):
    dscAll = []
    pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dscAll.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    DSC = np.asarray(dscAll)

    return DSC.mean()
'''

def dice_per_class(pred, target, num_classes=4):
    """pred et target : classes 0..3, forme [images, 256, 256].
    Renvoie le Dice de chaque classe sauf le fond : [VD, Myo, VG]."""
    dices = []
    for c in range(1, num_classes):           # 1 = VD, 2 = Myo, 3 = VG
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        total = p.sum().item() + t.sum().item()
        dices.append(1.0 if total == 0 else 2 * inter / total)
    return dices

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


from scipy import ndimage


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join('./Results/Images/', modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.view(labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

@torch.no_grad()
def validate(net, loader, loss_fn):
    """Passe toutes les images de validation dans le réseau.
    Renvoie la loss moyenne et le Dice [VD, Myo, VG]."""
    net.eval()
    losses, preds, targets = [], [], []
    for images, labels, _ in loader:
        images = to_var(images)
        target = getTargetSegmentation(to_var(labels))
        out = net(images)
        losses.append(loss_fn(out, target).item())
        preds.append(out.argmax(dim=1).cpu())      # classe la plus probable de chaque pixel
        targets.append(target.cpu())
    dsc = dice_per_class(torch.cat(preds), torch.cat(targets))
    return np.mean(losses), dsc