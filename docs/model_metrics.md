# Métriques et losses — YOLO détection moelle épinière

## Architecture

Modèle : **YOLO26n** (ultralytics 8.4.23) — variante nano, la plus légère de la famille YOLO26.  
Tâche : détection d'objet 2D (1 classe : `spine`), slices axiales IRM.  
Tête de détection : **Detect** (anchor-free, héritée de YOLOv8).

---

## Losses d'entraînement

À chaque step, trois losses sont calculées sur le batch d'entraînement. Elles sont loggées sous `train/` dans W&B.

### `train/box_loss` — CIoU loss

Mesure l'erreur de localisation de la bounding box.

```
L_box = (1 - CIoU) * poids_ancre
```

**CIoU (Complete IoU)** pénalise simultanément :
- le chevauchement (IoU classique)
- la distance entre centres
- le ratio d'aspect (w/h) de la box prédite vs GT

Plus CIoU est proche de 1 → les boxes coïncident parfaitement. `box_loss → 0` signifie des prédictions bien localisées.

### `train/cls_loss` — BCE loss

Mesure l'erreur de classification (présence/absence de la classe `spine`).

```
L_cls = BCE(score_prédit, score_cible)
```

La cible n'est pas un entier 0/1 mais un score continu issu du **Task-Aligned Assigner** (TAL) : chaque ancre reçoit un score d'alignement combinant IoU et confiance, ce qui lisse la supervision.

`cls_loss → 0` signifie que le modèle distingue bien les ancres avec objet des ancres vides.

### `train/dfl_loss` — Distribution Focal Loss

Mesure l'erreur sur la **distribution de probabilité** des distances bord-à-bord (left, top, right, bottom depuis l'ancre).

YOLO ne prédit pas directement `(x,y,w,h)` mais une distribution discrète sur `reg_max=16` valeurs pour chaque bord. DFL supervise cette distribution.

```
L_dfl = CrossEntropy(distribution_prédite, bord_cible_discret)
```

`dfl_loss → 0` signifie que la distribution est bien piquée autour de la vraie distance au bord.

### Loss totale

```
L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
```

Les poids par défaut ultralytics : `λ_box = 7.5`, `λ_cls = 0.5`, `λ_dfl = 1.5`.

---

## Losses de validation

Calculées sur le set val à la fin de chaque epoch, sans gradient. Mêmes formules que train, loggées sous `val/` dans W&B.

| Métrique | Description |
|---|---|
| `val/box_loss` | CIoU loss sur les boxes val |
| `val/cls_loss` | BCE loss sur la classification val |
| `val/dfl_loss` | DFL loss sur les distributions val |

> **Signal clé :** si `val/cls_loss` augmente alors que `train/cls_loss` diminue, le modèle surcharge sur la classification (le cas du run `yolo26_10mm_SI`).

---

## Métriques de détection (calculées sur val)

Loggées sous `metrics/` dans W&B, calculées après NMS sur les prédictions val.

### `metrics/precision(B)`

```
Precision = TP / (TP + FP)
```

Parmi toutes les boxes prédites, quelle fraction est correcte (IoU ≥ 0.5 avec une GT).  
Une precision faible = beaucoup de faux positifs (le modèle détecte partout).

### `metrics/recall(B)`

```
Recall = TP / (TP + FN)
```

Parmi toutes les GT, quelle fraction est détectée.  
Un recall faible = des objets manqués.

> `(B)` = Box, par opposition aux métriques de segmentation masque qui seraient `(M)`.

### `metrics/mAP50(B)`

**mean Average Precision à IoU = 0.5.**

Pour la classe `spine` :
1. Trier toutes les prédictions val par confiance décroissante
2. Calculer cumul TP/FP à chaque seuil de confiance
3. Tracer la courbe Précision–Rappel
4. Intégrer sous la courbe (aire trapézoïdale, interpolation 101 points)

Avec 1 seule classe, `mAP50 = AP50` de la classe `spine`.

### `metrics/mAP50-95(B)`

**mean Average Precision moyennée sur IoU ∈ {0.50, 0.55, …, 0.95}** (10 seuils, pas de 0.05).

C'est la métrique standard COCO. Elle est plus stricte que mAP50 car elle récompense les boxes précisément localisées.

```
mAP50-95 = mean(AP_0.50, AP_0.55, ..., AP_0.95)
```

### Fitness (critère `best.pt`)

```
fitness = 0.1 * mAP50 + 0.9 * mAP50-95
```

`best.pt` est sauvegardé à chaque fois que `fitness` dépasse le meilleur précédent sur val.  
Le poids fort sur mAP50-95 favorise la précision de localisation fine.

---

## Courbes loggées (onglet "curves" W&B)

| Courbe | Axes | Utilité |
|---|---|---|
| Precision–Recall | Recall (x), Precision (y) | Compromis P/R à tous les seuils de conf |
| F1–Confidence | Conf (x), F1 (y) | Seuil optimal de confiance |
| Precision–Confidence | Conf (x), Precision (y) | |
| Recall–Confidence | Conf (x), Recall (y) | |

---

## Métriques `evaluate.py` (post-entraînement)

Script `scripts/evaluate.py` — inference conf=0.001 sur `processed_{res}mm_SI/`, calcul sur DataFrame pandas.

| Métrique | Définition |
|---|---|
| `iou_mean` | IoU moyen sur les slices où GT et pred existent tous les deux |
| `dice_mean` | `2·IoU / (1 + IoU)` — exact algébriquement pour les bboxes rectangulaires |
| `recall50` | TP@IoU≥0.5 / n_gt |
| `precision50` | TP@IoU≥0.5 / n_pred |
| `f1_50` | 2·P·R / (P+R) à IoU≥0.5 |
| `ap50` | AP intégré à IoU=0.5, trié par confiance |
| `ap50_95` | Moyenne des AP sur IoU ∈ [0.50, 0.95] pas 0.05 |

Décomposition disponible : global / split (train/val/test) / dataset / dataset×contraste / dataset×contraste×split.

---

## Résumé : que surveiller pendant l'entraînement

| Signal | Interprétation |
|---|---|
| `train/box_loss` ↓, `val/box_loss` ↓ | Le modèle apprend à localiser, pas d'overfitting localisation |
| `val/cls_loss` ↑ alors que `train/cls_loss` ↓ | Overfitting classification — trop de faux positifs sur val |
| `metrics/precision(B)` ↓ + `recall(B)` stable | Le modèle prédit des boxes sur des slices vides |
| `fitness` = 0.1·mAP50 + 0.9·mAP50-95 | Critère de sauvegarde `best.pt` |
