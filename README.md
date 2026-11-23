# HeartBeat-Net

Segmentation IRM cardiaque (ACDC) : fond, ventricule droit (VD), myocarde (Myo), ventricule gauche (VG).

## Organisation (branche enes)

```
baseline/          Tâche 1A : baseline supervisée
  mainSegmentationChallenge_Enes.ipynb
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py
semi-supervised/   Tâche 1B : semi-supervisé
  semiSupervised_Enes.ipynb   essai 1 : pseudo-labels
  meanTeacher_Enes.ipynb      méthode retenue : Mean Teacher (modèle rendu)
  predict_Enes.ipynb          prédiction avec le modèle final, masques PNG (réglage TEST_DIR)
  figures_Enes.ipynb          figures et tableau du rapport
  UNet_Base_Enes.py, medicalDataLoader_Enes.py, utils_Enes.py
classification/    Tâche 2 : diagnostic (NOR, MINF, DCM, HCM, RV) par transfert de l'encodeur du UNet
  classification_Enes.ipynb          méthode retenue : mesures du cœur par le UNet + petit réseau
  classificationEncodeur_Enes.ipynb  essai 1 : encodeur du UNet + tête (réglage FREEZE)
  classification_Enes.py             patients, paires ED/ES, mesures du cœur, modèles
  groupes.csv                        diagnostic de chaque patient (tiré de MetaData)
  UNet_Base_Enes.py, utils_Enes.py
```

| Fichier | Rôle |
|---|---|
| `UNet_Base_Enes.py` | UNet (32 → 512 filtres, padding) |
| `medicalDataLoader_Enes.py` | Chargement des données (masque → classes 0 à 3), augmentation, images non annotées, pseudo-labels |
| `utils_Enes.py` | DiceLoss, Dice, Hausdorff, post-traitement, évaluation, pseudo-labels, figures |

Les `.py` sont copiés dans chaque dossier pour que chaque partie soit autonome.

## Lancer

Sur Kaggle (GPU, dataset acdc-heartbeat, Internet activés) : cloner la branche `enes`, puis lancer chaque notebook depuis son dossier (`%cd` puis `%run`), dans l'ordre ci-dessous.

À la main :

Les données sont dans `Data/`, à côté du dossier `HeartBeat-Net` (sur Kaggle, le chemin est choisi automatiquement). Chaque notebook se lance depuis son dossier.

1. `baseline/mainSegmentationChallenge_Enes.ipynb` → `baseline/models/final_dice/`
2. `semi-supervised/semiSupervised_Enes.ipynb` → `models/pseudo_labels/` (utilise le modèle de la baseline)
3. `semi-supervised/meanTeacher_Enes.ipynb`
   entraînement sur train (+ les images sans masque), évaluation sur val → `models/mean_teacher/last.pth` (**modèle rendu**)
4. `semi-supervised/predict_Enes.ipynb` : mettre `TEST_DIR` sur le dossier des images de test → masques PNG (0, 85, 170, 255) et `predictions_test.zip`
5. `semi-supervised/figures_Enes.ipynb` : figures et tableau du rapport

## Résultats (run final du 22/11/2026)

### Tâche 1 : segmentation (validation, 74 images)

Dernier modèle de chaque méthode, avec post-traitement. ASD = distance moyenne entre les contours, Hausdorff = la pire.

| Méthode | Dice moyen, 10 dernières époques | CE | Dice VD / Myo / VG | Hausdorff VD / Myo / VG (px) | ASD VD / Myo / VG (px) | Paramètres |
|---|---|---|---|---|---|---|
| 1A supervisé | 0.818 | 0.039 | 0.807 / 0.754 / 0.897 | 15.1 / 10.4 / 5.6 | 4.71 / 2.50 / 1.73 | 3 844 388 |
| Pseudo-labels | 0.778 | 0.061 | 0.767 / 0.696 / 0.853 | 17.7 / 15.4 / 7.5 | 5.50 / 3.81 / 2.69 | 3 844 388 |
| **Mean Teacher** | **0.854** | **0.027** | **0.841 / 0.786 / 0.926** | **12.3 / 9.8 / 4.1** | **4.06 / 2.26 / 1.24** | 3 844 388 |

Mean Teacher est le meilleur sur toutes les métriques, avec le même réseau. Les runs précédents donnaient 0.840 et 0.844 : le gain est reproductible.
Modèle rendu : ce Mean Teacher (`semi-supervised/models/mean_teacher/last.pth`).

### Tâche 2 : diagnostic (validation croisée 5 plis par patient, 66 patients)

| Méthode | Patients bien classés |
|---|---|
| Encodeur du UNet gelé + tête | 18 / 66 = 27 % |
| Encodeur du UNet fine-tuné + tête | 17 / 66 = 26 % |
| **Mesures du coeur par le UNet + ensemble de 10 réseaux** | **56 / 66 = 84.8 %** |

Modèle rendu : `classification/models/classifier.pth`.
