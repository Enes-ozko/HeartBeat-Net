# HeartBeat-Net (branche nadir)

Segmentation IRM cardiaque (ACDC) : fond + 3 structures du cœur.

## Organisation

```
baseline/   Baseline supervisée : UNet du prof, CE seule, sans augmentation, balayage de 6 lr
  mainSegmentationChallenge_Nadir.ipynb   notebook principal (entraînement + évaluation + synthèse)
  evaluation_Nadir.py                     métriques (CE, Dice, Hausdorff, ASD), tableaux, figures
  UNet_Base_Nadir.py, utils_Nadir.py, medicalDataLoader_Nadir.py, progressBar_Nadir.py
```

Règle d'évaluation (la même que la branche `enes`) : on évalue le **dernier** modèle de
chaque entraînement et on choisit le lr sur le Dice de validation moyenné sur les
**10 dernières époques**.

## Lancer

Les données sont dans `Data/` à la racine du dépôt (`Data/train`, `Data/val`). Le notebook se
lance depuis `baseline/` : il cherche `./Data/`, sinon `../Data/`. Les sorties vont dans
`baseline/Results/` et `baseline/models/`.

Sur Kaggle (GPU, Internet activé) : cloner la branche `nadir` dans `/kaggle/temp`, aller dans
`baseline/`, créer un lien `Data` vers le dataset, puis exécuter le notebook avec `nbconvert`.
