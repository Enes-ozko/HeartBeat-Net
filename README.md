# HeartBeat-Net

Segmentation IRM cardiaque (ACDC) : fond, ventricule droit (VD), myocarde (Myo), ventricule gauche (VG).

## Organisation (branche enes)

```
baseline/          Tâche 1A : baseline supervisée
  mainSegmentationChallenge_Enes.ipynb
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py, progressBar.py
semi-supervised/   Tâche 1B : semi-supervisé
  semiSupervised_Enes.ipynb   essai 1 : pseudo-labels
  meanTeacher_Enes.ipynb      méthode retenue : Mean Teacher (variables SEED, FINAL)
  predict_Enes.ipynb          évaluation, ensemble, masques PNG (variables MODELS, TEST_DIR)
  figures_Enes.ipynb          figures et tableau du rapport
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py, progressBar.py
```

| Fichier | Rôle |
|---|---|
| `UNet_Base_Enes.py` | UNet (32 → 512 filtres, padding) |
| `medicalDataLoader_Enes.py` | Chargement des données, augmentation, images non annotées, pseudo-labels |
| `utils_Enes.py` | Dice, Hausdorff, DiceLoss, post-traitement, figures |
| `progressBar.py` | Barre de progression (code du prof) |

Les `.py` sont copiés dans chaque dossier pour que chaque partie soit autonome.

## Lancer

Les données sont dans `../Data/` en local (sur Kaggle, le chemin est choisi automatiquement).

1. `mainSegmentationChallenge_Enes.ipynb` → `models/final_dice/`
2. `semiSupervised_Enes.ipynb` → `models/pseudo_labels/` (a besoin du modèle 1A)
3. `meanTeacher_Enes.ipynb` avec `SEED=0` puis `SEED=1` → `models/mean_teacher_w10_seed*/`
4. Modèles finaux : `meanTeacher_Enes.ipynb` avec `FINAL=1` et `SEED=0,1,2` (train + val) → `models/final_seed*/`
5. Test : dans `predict_Enes.ipynb`, `MODELS=models/final_seed0/last.pth,models/final_seed1/last.pth,models/final_seed2/last.pth` et `TEST_DIR=<dossier des images>` → `predictions_test.zip`

## Résultats (validation, 74 images)

| Méthode | Dice moyen, 10 dernières époques | Dice + post-traitement | Hausdorff VD / Myo / VG (px) |
|---|---|---|---|
| 1A supervisé | 0.815 | 0.840 | 13.4 / 9.8 / 4.6 |
| Pseudo-labels | 0.771 | 0.807 | 14.6 / 14.0 / 5.8 |
| Mean Teacher, graine 0 | 0.840 | 0.839 | 14.3 / 9.2 / 4.9 |
| Mean Teacher, graine 1 | 0.844 | 0.847 | 12.7 / 8.1 / 4.2 |
| Ensemble Mean Teacher | — | 0.848 | 12.9 / 8.3 / 4.3 |

Soumission : ensemble des 3 Mean Teacher entraînés sur train + val.
