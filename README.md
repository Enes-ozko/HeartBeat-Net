# HeartBeat-Net

Projet MTI865 : segmentation semi-supervisée et diagnostic automatique sur IRM cardiaque (challenge ACDC).

- **Tâche 1, segmentation** : pour chaque pixel, fond, ventricule droit (VD), myocarde (Myo) ou ventricule gauche (VG).
- **Tâche 2, diagnostic** : groupe du patient parmi NOR, MINF, DCM, HCM et RV, par transfert depuis le réseau de segmentation.

## Évaluer nos modèles (pour le correcteur)

Les modèles entraînés sont inclus dans le dépôt. Les données ACDC doivent être dans un dossier `Data/` placé à côté du dossier `HeartBeat-Net` (`Data/train`, `Data/val`) ; sur Kaggle, le chemin est trouvé tout seul.

```bash
pip install -r requirements.txt
```

**Segmentation** (depuis `semi-supervised/`) :

```bash
python3 evaluer_segmentation.py models/mean_teacher/last.pth eval
python3 evaluer_segmentation.py models/mean_teacher_final/last.pth predict <dossier_des_images>
```

- `eval` : Dice, Hausdorff et ASD sur la validation (avec post-traitement).
- `predict` : un masque PNG par image (valeurs 0, 85, 170, 255, même nom que l'image) dans `predictions_test/`.

**Diagnostic** (depuis `classification/`) :

```bash
python3 evaluer_classification.py ../semi-supervised/models/mean_teacher_final/last.pth models/classifier.pth eval
python3 evaluer_classification.py ../semi-supervised/models/mean_teacher_final/last.pth models/classifier.pth predict <dossier_des_images>
```

- `eval` : patients bien classés et rappel par classe, comparés à `groupes.csv`.
- `predict` : `predictions_diagnostic.csv` (patient, groupe prédit, confiance). Les images doivent s'appeler `patientXXX_YY_ZZ.png` (patient, instant, coupe), avec les deux instants ED et ES.

Les deux scripts choisissent CUDA ou CPU tout seuls.

## Modèles inclus

| Fichier | Contenu |
|---|---|
| `baseline/models/final_dice/last.pth` | baseline+ : UNet amélioré, supervisé |
| `semi-supervised/models/pseudo_labels/last.pth` | essai semi-supervisé : pseudo-labels |
| `semi-supervised/models/mean_teacher/last.pth` | Mean Teacher entraîné sur l'entraînement seul : **modèle des résultats ci-dessous** |
| `semi-supervised/models/mean_teacher_final/last.pth` | **modèle rendu** : même recette, réentraîné sur entraînement + validation |
| `classification/models/classifier.pth` | ensemble de 10 petits réseaux de diagnostic (mesures du cœur par `mean_teacher_final`) |

Les hyperparamètres ont été choisis sur la validation avec le Mean Teacher entraîné sur l'entraînement seul. Le modèle rendu a ensuite été réentraîné avec la même recette sur l'entraînement et la validation réunis : son score sur la validation n'est donc pas un résultat. On garde toujours le **dernier** modèle (`last.pth`) : aucune époque n'est choisie sur la validation.

## Résultats (validation, 74 images)

Dernier modèle de chaque méthode, avec post-traitement. Dice moyen des 10 dernières époques pour comparer les méthodes. ASD = distance moyenne entre les contours, Hausdorff = la pire.

| Méthode | Dice moyen (10 dernières ép.) | CE | Dice VD / Myo / VG | Hausdorff (px) | ASD (px) | Paramètres |
|---|---|---|---|---|---|---|
| Baseline (UNet du prof, CE seule, Nadir) | voir `mainSegmentationChallenge_Nadir.ipynb` | | | | | |
| Baseline+ (UNet amélioré) | 0.818 | 0.039 | 0.807 / 0.754 / 0.897 | 15.1 / 10.4 / 5.6 | 4.71 / 2.50 / 1.73 | 3 844 388 |
| Pseudo-labels | 0.778 | 0.061 | 0.767 / 0.696 / 0.853 | 17.7 / 15.4 / 7.5 | 5.50 / 3.81 / 2.69 | 3 844 388 |
| **Mean Teacher** | **0.854** | **0.027** | **0.841 / 0.786 / 0.926** | **12.3 / 9.8 / 4.1** | **4.06 / 2.26 / 1.24** | 3 844 388 |

Diagnostic, validation croisée en 5 plis par patient (66 patients) :

| Méthode | Patients bien classés |
|---|---|
| Encodeur du UNet gelé + tête | 18 / 66 = 27 % |
| Encodeur du UNet réentraîné + tête | 17 / 66 = 26 % |
| **Mesures du cœur par le UNet + ensemble de 10 réseaux** | **56 / 66 = 84.8 %** |

Toutes les figures et les tableaux sont dans `resultats/` (courbes, matrices de confusion, exemples de prédictions, mesures par maladie).

## Organisation

```
baseline/           Tâche 1A
  mainSegmentationChallenge_Nadir.ipynb   baseline : UNet du prof, CE seule, balayage du learning rate
  evaluation_Nadir.py                     CE, Dice, Hausdorff, ASD, tableaux et figures de la baseline
  mainSegmentationChallenge_Enes.ipynb    baseline+ : 32 à 512 filtres, padding, CE + Dice, augmentation, cosinus
semi-supervised/    Tâche 1B
  semiSupervised_Enes.ipynb   essai : pseudo-labels
  meanTeacher_Enes.ipynb      méthode retenue : Mean Teacher (réglage FINAL : entraînement + validation)
  figures_Enes.ipynb          figures et tableau des résultats
  predict_Enes.ipynb          masques PNG avec le modèle rendu
  evaluer_segmentation.py     évaluation ou prédiction en ligne de commande
classification/     Tâche 2
  classificationEncodeur_Enes.ipynb  essai : encodeur du UNet + tête (réglage FREEZE)
  classification_Enes.ipynb          méthode retenue : mesures du cœur + ensemble de réseaux
  classification_Enes.py             patients, paires ED/ES, mesures du cœur, modèles
  groupes.csv                        diagnostic de chaque patient (tiré de MetaData)
  evaluer_classification.py          évaluation ou prédiction en ligne de commande
resultats/          figures et tableaux du run final
```

Dans chaque dossier, `UNet_Base_*.py` est le réseau, `medicalDataLoader_*.py` le chargement des données et `utils_*.py` les métriques et le post-traitement. Les fichiers `_Enes` sont copiés dans chaque dossier pour que chaque partie soit autonome.

## Réentraîner

Chaque notebook se lance depuis son dossier, dans cet ordre : `baseline/`, `semi-supervised/semiSupervised_Enes`, `semi-supervised/meanTeacher_Enes` (avec `FINAL = False` puis `True`), `semi-supervised/figures_Enes`, puis `classification/`. La classification a besoin de `mean_teacher_final/last.pth`. Le notebook de Nadir a besoin de `medpy` (`pip install medpy` sur Kaggle).
