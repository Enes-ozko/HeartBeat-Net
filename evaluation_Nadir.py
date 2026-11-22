# -*- coding: utf-8 -*-
"""
Évaluation du balayage de learning rate et du modèle retenu (projet MTI865, ACDC).

Ce module ne contient que des fonctions : rien ne s'exécute à l'import.
Il est appelé depuis les dernières cellules de `mainSegmentationChallenge_Nadir.ipynb`.

Choix techniques :
- pas de pandas (bloqué sur certains postes Windows) : csv / json / numpy uniquement ;
- pas de `utils.inference` (elle appelle `.cuda()` sans vérification) : tout tourne sur CPU,
  les poids sont chargés avec `torch.load(..., map_location='cpu')` ;
- les images de validation sont lues directement avec PIL + `transforms.ToTensor()`,
  exactement comme `medicalDataLoader.MedicalImageDataset('val', ..., equalize=False)`
  (mêmes listes triées d'images et de masques, appariées dans l'ordre) ;
- les classes sont nommées « classe 1/2/3 » : la correspondance avec VD / myocarde / VG
  reste à vérifier visuellement.
"""

import csv
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from medpy.metric.binary import hd, assd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from UNet_Base_Nadir import UNet
from utils_Nadir import getTargetSegmentation, dice_per_class


# --------------------------------------------------------------------------------------
# Constantes d'affichage (mêmes couleurs de classes dans toutes les figures)
# --------------------------------------------------------------------------------------
N_CLASSES = 4
NOMS_CLASSES = ['fond (classe 0)', 'classe 1', 'classe 2', 'classe 3']
# Fond noir + 3 premières teintes catégorielles d'une palette validée daltonisme.
COULEURS_CLASSES = ['#000000', '#2a78d6', '#eb6834', '#1baf7a']
COULEUR_ERREUR = '#d55181'      # magenta : distinct des 3 couleurs de classes
COULEUR_MOYENNE = '#222222'     # encre foncée pour la courbe « moyenne »
CMAP_CLASSES = ListedColormap(COULEURS_CLASSES)


# --------------------------------------------------------------------------------------
# Utilitaires fichiers
# --------------------------------------------------------------------------------------
def nom_modele(lr):
    """Nom du modèle utilisé par la boucle de balayage : 'lr_<valeur>'."""
    return 'lr_{}'.format(lr)


def lire_historiques(lrs, dossier_stats='Results/Statistics'):
    """Lit les fichiers history.json écrits par la boucle de balayage.

    Renvoie un dictionnaire {lr: history}. Lève une erreur explicite si un fichier manque.
    """
    historiques = {}
    for lr in lrs:
        chemin = os.path.join(dossier_stats, nom_modele(lr), 'history.json')
        if not os.path.exists(chemin):
            raise FileNotFoundError("Historique introuvable : {} (lancer d'abord le balayage)".format(chemin))
        with open(chemin, encoding='utf-8') as f:
            historiques[lr] = json.load(f)
    return historiques


def ecrire_csv(lignes, colonnes, chemin):
    """Écrit une liste de dictionnaires dans un CSV (séparateur virgule, UTF-8)."""
    os.makedirs(os.path.dirname(chemin) or '.', exist_ok=True)
    with open(chemin, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        for ligne in lignes:
            w.writerow({c: ligne.get(c, '') for c in colonnes})


def _formater(v, nd=4):
    """Formatage d'une cellule de tableau (flottants arrondis, NaN -> 'n/d')."""
    if isinstance(v, (float, np.floating)):
        return 'n/d' if np.isnan(v) else '{:.{}f}'.format(v, nd)
    return str(v)


def tableau_markdown(lignes, colonnes, entetes=None):
    """Construit un tableau markdown (chaîne) à partir d'une liste de dictionnaires."""
    entetes = entetes or colonnes
    txt = '| ' + ' | '.join(entetes) + ' |\n'
    txt += '|' + '|'.join(['---'] * len(colonnes)) + '|\n'
    for ligne in lignes:
        txt += '| ' + ' | '.join(_formater(ligne.get(c, '')) for c in colonnes) + ' |\n'
    return txt


def sauvegarder_figure(fig, chemin_png):
    """Enregistre la figure en PNG (300 dpi, obligatoire) puis tente le PDF.

    Le PDF est dans un try/except car Windows peut bloquer `fontTools`.
    Renvoie la liste des fichiers effectivement écrits.
    """
    os.makedirs(os.path.dirname(chemin_png) or '.', exist_ok=True)
    fig.savefig(chemin_png, dpi=300, bbox_inches='tight')
    ecrits = [chemin_png]
    chemin_pdf = os.path.splitext(chemin_png)[0] + '.pdf'
    try:
        fig.savefig(chemin_pdf, bbox_inches='tight')
        ecrits.append(chemin_pdf)
    except Exception as e:  # noqa: BLE001 - on veut juste prévenir, pas planter
        print("[Avertissement] PDF non enregistré ({}) : {}".format(chemin_pdf, e))
    return ecrits


# --------------------------------------------------------------------------------------
# A. Données, modèle et métriques (calculées UNE seule fois par modèle)
# --------------------------------------------------------------------------------------
def charger_validation(root_dir='./Data/'):
    """Charge toute la validation en mémoire (74 coupes 256x256 : c'est léger).

    Renvoie :
      images : tenseur float (N, 1, H, W) dans [0, 1]
      gt     : tenseur long (N, H, W) de classes 0..3 (via getTargetSegmentation)
      noms   : liste des noms de fichiers d'images
    """
    dossier_img = os.path.join(root_dir, 'val', 'Img')
    dossier_gt = os.path.join(root_dir, 'val', 'GT')
    noms_img = sorted(os.listdir(dossier_img))
    noms_gt = sorted(os.listdir(dossier_gt))
    to_tensor = transforms.ToTensor()
    images, masques = [], []
    for n_img, n_gt in zip(noms_img, noms_gt):
        images.append(to_tensor(Image.open(os.path.join(dossier_img, n_img))))
        masques.append(to_tensor(Image.open(os.path.join(dossier_gt, n_gt)).convert('L')))
    images = torch.stack(images)                 # (N, 1, H, W)
    masques = torch.stack(masques)               # (N, 1, H, W)
    gt = getTargetSegmentation(masques)          # (N, H, W) ; squeeze() du prof
    if gt.dim() == 2:                            # cas N = 1 : on garde l'axe des coupes
        gt = gt.unsqueeze(0)
    return images, gt, noms_img


def charger_modele(chemin_poids, construire=None):
    """Crée le réseau (UNet par défaut), charge les poids sur CPU, passe en mode évaluation."""
    net = construire() if construire else UNet(N_CLASSES)
    net.load_state_dict(torch.load(chemin_poids, map_location='cpu'))
    net.eval()
    return net


def compter_parametres(net):
    """Nombre de paramètres entraînables (même calcul que dans runTraining)."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def predire_et_ce(net, images, gt, batch_size=8):
    """Un seul passage du réseau sur la validation (CPU).

    Renvoie (preds, ce) : preds = argmax des classes (N, H, W) ; ce = entropie croisée
    moyenne sur tous les pixels de la validation (calculée d'un bloc, donc très
    légèrement différente de la moyenne des CE par batch de history.json, le dernier
    batch étant plus petit).
    """
    somme_ce, n_pixels, preds = 0.0, 0, []
    with torch.no_grad():
        for k in range(0, images.shape[0], batch_size):
            sortie = net(images[k:k + batch_size])
            cible = gt[k:k + batch_size]
            somme_ce += torch.nn.functional.cross_entropy(sortie, cible, reduction='sum').item()
            n_pixels += cible.numel()
            preds.append(torch.argmax(sortie, dim=1))
    return torch.cat(preds), somme_ce / n_pixels


def distances_par_classe(preds, gt, n_classes=N_CLASSES):
    """Hausdorff (medpy hd) et ASD (medpy assd, symétrique), coupe par coupe, en PIXELS.

    Une seule boucle sur les coupes pour les deux distances. Pour chaque classe c >= 1 :
      - vérité ET prédiction non vides  -> on calcule hd et assd ;
      - vérité non vide, prédiction vide -> comptée dans `n_pred_vide` (pas de distance) ;
      - vérité vide, prédiction non vide -> comptée dans `n_gt_vide` (pas de distance) ;
      - les deux vides                   -> ignorée.
    Écart-type de population (ddof=0) ; NaN si aucune distance n'a pu être calculée.
    """
    p_np = preds.cpu().numpy()
    g_np = gt.cpu().numpy()
    nan = float('nan')
    res = {}
    for c in range(1, n_classes):
        d_hd, d_asd, n_pred_vide, n_gt_vide = [], [], 0, 0
        for k in range(p_np.shape[0]):
            p = p_np[k] == c
            g = g_np[k] == c
            if g.any() and p.any():
                d_hd.append(hd(p, g))            # voxelspacing=None -> pixels
                d_asd.append(assd(p, g))
            elif g.any():
                n_pred_vide += 1
            elif p.any():
                n_gt_vide += 1
        res[c] = {'hd_moyenne': float(np.mean(d_hd)) if d_hd else nan,
                  'hd_ecart_type': float(np.std(d_hd)) if d_hd else nan,
                  'asd_moyenne': float(np.mean(d_asd)) if d_asd else nan,
                  'n_calcule': len(d_hd),
                  'n_pred_vide': n_pred_vide,
                  'n_gt_vide': n_gt_vide}
    return res


COLONNES_FINAL = ['classe', 'dice', 'hd_moyenne_px', 'hd_ecart_type_px', 'asd_moyenne_px',
                  'n_coupes_hd', 'n_pred_vide_gt_non_vide', 'n_gt_vide_pred_non_vide']
ENTETES_FINAL = ['Classe', 'Dice', 'HD moyenne (px)', 'HD écart-type (px)', 'ASD moyenne (px)',
                 'Coupes avec HD', 'Préd. vide / vérité non vide', 'Vérité vide / préd. non vide']


def _moyenne_sans_nan(valeurs):
    v = [x for x in valeurs if not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


def calculer_metriques(preds, gt, ce, nb_parametres):
    """Toutes les métriques d'un modèle à partir de ses prédictions.

    - 'par_classe' : une ligne par classe 1..3 puis une ligne « moyenne » (moyenne des
      3 classes pour Dice / HD / ASD ; les comptes sont sommés) -> tableau détaillé ;
    - champs résumés (ce_moyenne, dice_moyen, hd_moyenne_px, asd_moyenne_px,
      nb_parametres) -> une ligne du tableau comparatif.
    Le Dice est « pooled » sur toute la validation (même définition que dice_per_class
    pendant l'entraînement).
    """
    dices = dice_per_class(preds, gt)
    dist = distances_par_classe(preds, gt)
    lignes = []
    for c in range(1, N_CLASSES):
        lignes.append({'classe': 'classe {}'.format(c),
                       'dice': float(dices[c - 1]),
                       'hd_moyenne_px': dist[c]['hd_moyenne'],
                       'hd_ecart_type_px': dist[c]['hd_ecart_type'],
                       'asd_moyenne_px': dist[c]['asd_moyenne'],
                       'n_coupes_hd': dist[c]['n_calcule'],
                       'n_pred_vide_gt_non_vide': dist[c]['n_pred_vide'],
                       'n_gt_vide_pred_non_vide': dist[c]['n_gt_vide']})
    moyenne = {'classe': 'moyenne',
               'dice': float(np.mean(dices)),
               'hd_moyenne_px': _moyenne_sans_nan([l['hd_moyenne_px'] for l in lignes]),
               'hd_ecart_type_px': '',
               'asd_moyenne_px': _moyenne_sans_nan([l['asd_moyenne_px'] for l in lignes]),
               'n_coupes_hd': sum(l['n_coupes_hd'] for l in lignes),
               'n_pred_vide_gt_non_vide': sum(l['n_pred_vide_gt_non_vide'] for l in lignes),
               'n_gt_vide_pred_non_vide': sum(l['n_gt_vide_pred_non_vide'] for l in lignes)}
    lignes.append(moyenne)
    return {'ce_moyenne': float(ce),
            'dice_moyen': moyenne['dice'],
            'hd_moyenne_px': moyenne['hd_moyenne_px'],
            'asd_moyenne_px': moyenne['asd_moyenne_px'],
            'nb_parametres': int(nb_parametres),
            'par_classe': lignes}


# --------------------------------------------------------------------------------------
# B. Courbes d'apprentissage
# --------------------------------------------------------------------------------------
def figure_courbes_apprentissage(history, chemin_png, titre=None):
    """(a) perte d'entraînement et CE de validation ; (b) Dice de validation par époque.

    Les époques sont numérotées à partir de 1 ; la meilleure époque est marquée.
    """
    n = len(history['train_loss'])
    ep = np.arange(1, n + 1)
    best = int(history['best_epoch']) + 1
    dices = np.asarray(history['val_dice'])      # (n, 3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(ep, history['train_loss'], marker='o', linewidth=2, color=COULEURS_CLASSES[1],
            label="Perte d'entraînement (CE)")
    ax.plot(ep, history['val_ce'], marker='o', linewidth=2, color=COULEURS_CLASSES[2],
            label='CE de validation')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Entropie croisée')
    ax.set_title("(a) Pertes par époque")

    ax = axes[1]
    for c in range(3):
        ax.plot(ep, dices[:, c], marker='o', linestyle='--', linewidth=1.2,
                color=COULEURS_CLASSES[c + 1], label='Classe {}'.format(c + 1))
    ax.plot(ep, history['val_dice_mean'], marker='o', linewidth=2.5, color=COULEUR_MOYENNE,
            label='Moyenne')
    ax.set_xlabel('Époque')
    ax.set_ylabel('Dice de validation')
    ax.set_ylim(0, 1)
    ax.set_title("(b) Dice de validation par époque")

    for ax in axes:
        ax.axvline(best, color='#888888', linestyle=':', linewidth=1,
                   label='Meilleure époque ({})'.format(best))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if n <= 20:
            ax.set_xticks(ep)

    if titre:
        fig.suptitle(titre)
    fig.tight_layout()
    ecrits = sauvegarder_figure(fig, chemin_png)
    return fig, ecrits


# --------------------------------------------------------------------------------------
# C. Figure qualitative
# --------------------------------------------------------------------------------------
def dice_par_coupe(preds, gt, n_classes=N_CLASSES):
    """Dice moyen de chaque coupe, sur les classes présentes (vérité OU prédiction).

    Une classe absente à la fois de la vérité et de la prédiction n'entre pas dans la
    moyenne (sinon elle compterait pour un Dice de 1). Coupe sans aucune structure
    (ni vérité, ni prédiction) -> NaN, exclue du classement.
    """
    out = np.full(preds.shape[0], np.nan)
    for k in range(preds.shape[0]):
        valeurs = []
        for c in range(1, n_classes):
            p = preds[k] == c
            g = gt[k] == c
            if p.any() or g.any():
                inter = (p & g).sum().item()
                valeurs.append(2.0 * inter / (p.sum().item() + g.sum().item()))
        if valeurs:
            out[k] = float(np.mean(valeurs))
    return out


def choisir_coupes(dices_coupes, seed=0):
    """Indices de 4 coupes distinctes : meilleure, médiane, pire, une au hasard.

    Le hasard utilise np.random.default_rng(seed) (reproductible) parmi les coupes
    valides restantes. Renvoie une liste de (indice, étiquette).
    """
    valides = np.where(~np.isnan(dices_coupes))[0]
    if valides.size < 4:
        raise ValueError("Moins de 4 coupes valides pour la figure qualitative.")
    ordre = valides[np.argsort(dices_coupes[valides], kind='stable')]   # croissant
    choix = [(int(ordre[-1]), 'meilleure'),
             (int(ordre[len(ordre) // 2]), 'médiane'),
             (int(ordre[0]), 'pire')]
    deja = {i for i, _ in choix}
    restants = [int(i) for i in valides if int(i) not in deja]
    rng = np.random.default_rng(seed)
    choix.append((int(rng.choice(restants)), 'au hasard'))
    return choix


def figure_qualitative(images, gt, preds, choix, noms, dices_coupes, chemin_png):
    """Une ligne par coupe : image IRM, vrai masque, prédiction, carte des erreurs.

    Même palette (CMAP_CLASSES, valeurs 0..3) pour le vrai masque et la prédiction.
    Carte des erreurs : image en niveaux de gris + pixels où prédiction != vérité en magenta.
    Le titre de chaque ligne est le nom du fichier.
    """
    n = len(choix)
    fig = plt.figure(figsize=(12, 3.3 * n + 0.6))
    sous_figs = fig.subfigures(n, 1)
    if n == 1:
        sous_figs = [sous_figs]
    titres_col = ['Image IRM', 'Vrai masque', 'Prédiction', 'Carte des erreurs']
    cmap_err = ListedColormap([COULEUR_ERREUR])

    for sf, (idx, etiquette) in zip(sous_figs, choix):
        img = images[idx, 0].cpu().numpy()
        g = gt[idx].cpu().numpy()
        p = preds[idx].cpu().numpy()
        err = np.ma.masked_where(p == g, np.ones_like(p))
        sf.suptitle('{}  —  coupe {} (Dice moyen de la coupe = {:.3f})'.format(
            os.path.basename(noms[idx]), etiquette, dices_coupes[idx]), fontsize=10)
        axes = sf.subplots(1, 4)
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(g, cmap=CMAP_CLASSES, vmin=0, vmax=N_CLASSES - 1, interpolation='nearest')
        axes[2].imshow(p, cmap=CMAP_CLASSES, vmin=0, vmax=N_CLASSES - 1, interpolation='nearest')
        axes[3].imshow(img, cmap='gray')
        axes[3].imshow(err, cmap=cmap_err, vmin=0, vmax=1, interpolation='nearest')
        for ax, t in zip(axes, titres_col):
            ax.set_title(t, fontsize=9)
            ax.axis('off')

    poignees = [Patch(facecolor=COULEURS_CLASSES[c], edgecolor='#888888', label=NOMS_CLASSES[c])
                for c in range(N_CLASSES)]
    poignees.append(Patch(facecolor=COULEUR_ERREUR, label='erreur (prédiction ≠ vérité)'))
    fig.legend(handles=poignees, loc='lower center', ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    ecrits = sauvegarder_figure(fig, chemin_png)
    return fig, ecrits


# --------------------------------------------------------------------------------------
# D. Sensibilité au learning rate (compare les lr : tracée après le balayage)
# --------------------------------------------------------------------------------------
def figure_balayage_lr(historiques, chemin_png):
    """(a) Meilleur Dice selon le lr ; (b) Dice moyen de validation par époque pour chaque lr.

    Couleurs de classes communes aux autres figures, époques numérotées à partir de 1.
    """
    lrs = list(historiques.keys())
    hist = historiques
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) Meilleur Dice de validation en fonction du learning rate
    ax = axes[0]
    best_mean = [hist[lr]['best_dice'] for lr in lrs]
    per_class = np.array([hist[lr]['val_dice'][hist[lr]['best_epoch']] for lr in lrs])
    for c in range(3):
        ax.plot(lrs, per_class[:, c], marker='o', linestyle='--', linewidth=1,
                color=COULEURS_CLASSES[c + 1], label='Classe {}'.format(c + 1))
    ax.plot(lrs, best_mean, marker='o', linewidth=2.5, color=COULEUR_MOYENNE, label='Moyenne')
    ax.set_xscale('log')
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Dice de validation (meilleure époque)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("(a) Meilleur Dice selon le learning rate")

    # (b) Évolution du Dice moyen de validation par époque
    ax = axes[1]
    cmap = plt.cm.viridis
    for k, lr in enumerate(lrs):
        y = hist[lr]['val_dice_mean']
        ax.plot(np.arange(1, len(y) + 1), y, color=cmap(k / max(len(lrs) - 1, 1)),
                label='lr = {}'.format(lr))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # époques entières
    ax.set_xlabel("Époque")
    ax.set_ylabel("Dice moyen de validation")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(b) Évolution du Dice de validation")

    fig.tight_layout()
    ecrits = sauvegarder_figure(fig, chemin_png)
    return fig, ecrits


# --------------------------------------------------------------------------------------
# E. Évaluation complète d'un entraînement + tableau comparatif lr x modèles
# --------------------------------------------------------------------------------------
DOSSIER_STATS = os.path.join('Results', 'Statistics')
DOSSIER_FIGURES = os.path.join('Results', 'Figures')


def evaluer_entrainement(nom, history, images, gt, noms, construire=None,
                         dossier_modeles='models', figures=True, seed=0):
    """À appeler juste après l'entraînement d'un modèle (un lr) : tout est fait en une fois.

    1. charge models/<nom>/best_model et prédit la validation une seule fois ;
    2. calcule toutes les métriques (CE, Dice, Hausdorff, ASD, nb de paramètres) et les
       écrit dans Results/Statistics/<nom>/metrics.json ;
    3. si figures=True, trace les courbes d'apprentissage et la figure qualitative de ce
       modèle dans Results/Figures/<nom>/ (figures fermées : rien ne s'affiche ici).
    Renvoie le dictionnaire des métriques.
    """
    net = charger_modele(os.path.join(dossier_modeles, nom, 'best_model'), construire)
    preds, ce = predire_et_ce(net, images, gt)
    m = calculer_metriques(preds, gt, ce, compter_parametres(net))
    m.update({'nom': nom,
              'lr': history.get('lr'),
              'meilleure_epoque': int(history['best_epoch']) + 1,
              'n_epoques': len(history['train_loss']),
              'best_dice_history': float(history['best_dice'])})

    os.makedirs(os.path.join(DOSSIER_STATS, nom), exist_ok=True)
    with open(os.path.join(DOSSIER_STATS, nom, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    if figures:
        dossier = os.path.join(DOSSIER_FIGURES, nom)
        fig, _ = figure_courbes_apprentissage(
            history, os.path.join(dossier, 'learning_curves.png'), titre=nom)
        plt.close(fig)
        dices_coupes = dice_par_coupe(preds, gt)
        choix = choisir_coupes(dices_coupes, seed=seed)
        fig, _ = figure_qualitative(images, gt, preds, choix, noms, dices_coupes,
                                    os.path.join(dossier, 'qualitative.png'))
        plt.close(fig)
    return m


def lire_metriques(nom):
    """Lit Results/Statistics/<nom>/metrics.json (None s'il n'existe pas encore)."""
    chemin = os.path.join(DOSSIER_STATS, nom, 'metrics.json')
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding='utf-8') as f:
        return json.load(f)


def copier_figures(nom, dossier_resultats='Results'):
    """Copie les figures d'un modèle (le retenu) à la racine de Results/ pour le rapport :
    fig_learning_curves.png/.pdf et fig_qualitative.png/.pdf. Renvoie les fichiers copiés."""
    copies = []
    for src, dst in [('learning_curves', 'fig_learning_curves'), ('qualitative', 'fig_qualitative')]:
        for ext in ('.png', '.pdf'):
            s = os.path.join(DOSSIER_FIGURES, nom, src + ext)
            if os.path.exists(s):
                d = os.path.join(dossier_resultats, dst + ext)
                shutil.copyfile(s, d)
                copies.append(d)
    return copies


# Un modèle = un dictionnaire :
#   'nom'       : nom affiché dans le tableau (ex. 'Baseline', 'Modèle V1') ;
#   'construire': fonction sans argument qui renvoie le réseau (non entraîné) ;
#   'dossier'   : gabarit du modelName utilisé par la boucle de balayage, avec {lr}
#                 (ex. 'lr_{lr}' pour la baseline, 'v1_lr_{lr}' pour un futur modèle).
MODELES_PAR_DEFAUT = [
    {'nom': 'Baseline', 'construire': lambda: UNet(N_CLASSES), 'dossier': 'lr_{lr}'},
]

COLONNES_COMPARAISON = ['lr', 'modele', 'ce_moyenne', 'dice_moyen', 'hd_moyenne_px',
                        'asd_moyenne_px', 'nb_parametres']
ENTETES_COMPARAISON = ['lr', 'Modèle', 'CE moyenne', 'Dice moyen', 'Hausdorff moyen (px)',
                       'ASD moyenne (px)', 'Nb paramètres']


def tableau_comparatif(lrs, modeles=None):
    """Une ligne par (learning rate, modèle), groupée par lr. AUCUN recalcul : on lit
    les metrics.json écrits par evaluer_entrainement pendant le balayage.

    Si un modèle n'a pas encore été évalué pour un lr, la ligne est créée avec 'n/d'
    (le nombre de paramètres reste calculé à partir de l'architecture).
    """
    modeles = modeles or MODELES_PAR_DEFAUT
    nan = float('nan')
    lignes = []
    for lr in lrs:
        for m in modeles:
            met = lire_metriques(m['dossier'].format(lr=lr))
            if met is None:
                print("[Avertissement] pas de metrics.json pour {} (lr = {})".format(m['nom'], lr))
                met = {'ce_moyenne': nan, 'dice_moyen': nan, 'hd_moyenne_px': nan,
                       'asd_moyenne_px': nan,
                       'nb_parametres': compter_parametres(m['construire']())}
            ligne = {c: met[c] for c in COLONNES_COMPARAISON[2:]}
            ligne.update({'lr': lr, 'modele': m['nom']})
            lignes.append(ligne)
    return lignes
