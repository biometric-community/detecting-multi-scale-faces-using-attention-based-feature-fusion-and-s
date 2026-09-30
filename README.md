# SANet — Detecting Multi-Scale Faces (TBIOM 2020)

PyTorch rebuild of **SANet** (Attention-guided Feature Fusion + Smoothed Context Enhancement) from:

`papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s/`

**Repository:** [biometric-community/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s](https://github.com/biometric-community/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s)  
**Monorepo path:** `projects/papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s` in [tbiom](https://github.com/biometric-community/tbiom) (git submodule).

## Setup

```bash
bash scripts/setup_env.sh
```

Uses the monorepo shared `.venv` (no project-local venv).

## Data

Requires real **WIDER FACE** under:

`projects/datasets/wider-face/extracted/{WIDER_train,WIDER_val,wider_face_split}`

## Train / eval / predict / report

```bash
# Full paper schedule (120k iters, batch 14) — after size gate
bash scripts/check_dataset_size.sh ../../datasets/wider-face
bash scripts/train_full.sh

bash scripts/eval.sh
bash scripts/predict.sh --input /path/to/image.jpg
bash scripts/report.sh
```

Smoke (not for REPORT claims):

```bash
bash scripts/train.sh configs/smoke.yaml
```

## Layout

- `sanet/models/` — backbone, AFFM, SCEM, SANet
- `sanet/data/` — WIDER loader
- `sanet/losses/` — Eq. (5) + OHEM
- `configs/default.yaml` — paper protocol
- `REPORT.md` — generated from real logs
- `DEVIATIONS.md`, `FIDELITY_AUDIT.md`, `SOURCE_CODE.md`

## Citation

Shi, Xu, Kakadiaris. Detecting Multi-Scale Faces Using Attention-Based Feature Fusion and Smoothed Context Enhancement. *IEEE TBIOM*, 2020. DOI: [10.1109/TBIOM.2020.2993242](https://doi.org/10.1109/TBIOM.2020.2993242)
