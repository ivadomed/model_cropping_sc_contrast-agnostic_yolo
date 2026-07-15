# Preprocessing expérimental : union_bbox + norm_3ch_joint

## Contexte

Les deux options ci-dessous ont été ajoutées à `configs/preprocess.yaml` pour tester une
variante du preprocessing axial 3ch qui améliore la robustesse RL/AP au détriment d'une
légère imprécision sur l'axe SI.

---

## `union_bbox` (défaut : `false`)

### Comportement actuel (false)
Chaque slice reçoit sa propre bbox GT = bounding box de la moelle sur **cette** slice.
Les slices sans SC reçoivent un label vide.

### Nouveau comportement (true)
1. Avant la boucle de slices, toutes les bboxes par-slice sont calculées.
2. La **bbox union** est construite en prenant les extrêmes :
   - `x1 = min(cx - w/2)` sur toutes les slices avec SC
   - `x2 = max(cx + w/2)` idem
   - `y1 = min(cy - h/2)` idem
   - `y2 = max(cy + h/2)` idem
3. Cette bbox globale est écrite **dans toutes les slices** (avec ou sans SC).

### Conséquences
- **Gain** : précision RL et AP améliorée (le modèle voit la totalité de l'étendue de la moelle).
- **Perte** : l'axe SI perd sa précision — `bbox_3d.txt` aura `z1=0, z2=N-1`
  (toutes les slices extraites), soit une indétermination SI de ±20 mm environ.
- **Edge case** : si seule la dernière slice contient de la SC, la bbox union = bbox
  de cette seule slice → prédiction quand même possible.

### Suffixe dossier de sortie
`_unionbbox`  →  ex. `processed/10mm_SI_1mm_axial_3ch_norm3ch_unionbbox/`

---

## `norm_3ch_joint` (défaut : `false`)

### Comportement actuel (false)
Chaque canal (R, G, B) de l'image 3ch est normalisé **indépendamment** par percentile
[0.5 %, 99.5 %] sur ses propres pixels non-nuls.

### Nouveau comportement (true)
Les pixels non-nuls des **3 canaux réunis** sont poolés, puis un unique `[lo, hi]` est
calculé (`np.percentile(all_nz, [0.5, 99.5])`). Chaque canal est ensuite normalisé avec
ce même `lo/hi`.

### Conséquences
- Les intensités relatives entre les 3 canaux (slices voisines) sont préservées.
- Une slice voisine plus sombre reste plus sombre dans l'image (pas amplifiée).
- Cohérent avec `norm_scope: volume` en esprit, mais limité aux 3 slices de l'image.

### Condition
N'a d'effet que si `three_ch: true`. Sans effet sur le mode 1ch.

### Suffixe dossier de sortie
`_norm3ch`  →  ex. `processed/10mm_SI_1mm_axial_3ch_norm3ch/`

---

## Utilisation

### Via `run_pipeline.py` (recommandé)
Éditer `configs/preprocess.yaml`, mettre les flags à `true`, puis :
```bash
python scripts/run_pipeline.py --start 2
```

### Via CLI directe
```bash
python scripts/preprocess.py --config configs/preprocess.yaml --union-bbox --norm-3ch-joint
```

### Exemple de config complète pour tester les deux
```yaml
plane: axial
axial:
  si_res: 10.0
  inplane_res: 1.0
three_ch: true
si_stride: 1
norm_scope: slice
union_bbox: true
norm_3ch_joint: true
```
Dossier de sortie automatique : `processed/10mm_SI_1mm_axial_3ch_norm3ch_unionbbox/`
