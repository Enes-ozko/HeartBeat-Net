# HeartBeat-Net

Segmentation IRM cardiaque (ACDC) : fond, ventricule droit (VD), myocarde (Myo), ventricule gauche (VG).

## Organisation (branche enes)

```
baseline/          Tâche 1A : baseline supervisée
  mainSegmentationChallenge_Enes.ipynb
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py
semi-supervised/   Tâche 1B : semi-supervisé
  semiSupervised_Enes.ipynb   essai 1 : pseudo-labels
  meanTeacher_Enes.ipynb      méthode retenue : Mean Teacher (réglage FINAL)
  predict_Enes.ipynb          prédiction avec le modèle final, masques PNG (réglage TEST_DIR)
  figures_Enes.ipynb          figures et tableau du rapport
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py
```

| Fichier | Rôle |
|---|---|
| `UNet_Base_Enes.py` | UNet (32 → 512 filtres, padding) |
| `medicalDataLoader_Enes.py` | Chargement des données (masque → classes 0 à 3), augmentation, images non annotées, pseudo-labels |
| `utils_Enes.py` | DiceLoss, Dice, Hausdorff, post-traitement, évaluation, pseudo-labels, figures |

Les `.py` sont copiés dans chaque dossier pour que chaque partie soit autonome.

## Lancer

Sur Kaggle (GPU activé, données ACDC ajoutées) : ouvrir `run_kaggle_Enes.ipynb` et tout lancer. Il entraîne toute la tâche 1 dans l'ordre ci-dessous et met les modèles et les figures dans `/kaggle/working/resultats.zip`.

À la main :

Les données sont dans `Data/`, à côté du dossier `HeartBeat-Net` (sur Kaggle, le chemin est choisi automatiquement). Chaque notebook se lance depuis son dossier.

1. `baseline/mainSegmentationChallenge_Enes.ipynb` → `baseline/models/final_dice/`
2. `semi-supervised/semiSupervised_Enes.ipynb` → `models/pseudo_labels/` (utilise le modèle de la baseline)
3. `semi-supervised/meanTeacher_Enes.ipynb`
   - `FINAL = False` : entraînement sur train, évaluation sur val → `models/mean_teacher/` (comparaison du rapport)
   - `FINAL = True` : entraînement sur train + val → `models/mean_teacher_final/` (**modèle rendu**)
4. `semi-supervised/predict_Enes.ipynb` : mettre `TEST_DIR` sur le dossier des images de test → masques PNG (0, 85, 170, 255) et `predictions_test.zip`
5. `semi-supervised/figures_Enes.ipynb` : figures et tableau du rapport

## Résultats (validation, 74 images)

| Méthode | Dice moyen, 10 dernières époques | Dice + post-traitement | Hausdorff VD / Myo / VG (px) |
|---|---|---|---|
| 1A supervisé | 0.815 | 0.840 | 13.4 / 9.8 / 4.6 |
| Pseudo-labels | 0.771 | 0.807 | 14.6 / 14.0 / 5.8 |
| Mean Teacher, graine 0 | 0.840 | 0.839 | 14.3 / 9.2 / 4.9 |
| Mean Teacher, graine 1 | 0.844 | 0.847 | 12.7 / 8.1 / 4.2 |

Deux graines ont été testées pour vérifier que le gain de Mean Teacher est reproductible.
Modèle rendu : un seul Mean Teacher (graine 0) ré-entraîné sur train + val (`models/mean_teacher_final/last.pth`).
