# Implementation plan: Detecting Multi-Scale Faces Using Attention-Based Feature Fusion and Smoothed Context Enhancement (SANet)

- **Paper:** `papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s/`
- **Project:** `projects/papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s/`
- **Stack:** Python + PyTorch
- **Upstream:** none official (ICB 2019 / TBIOM 2020 authors did not release code; unrelated crowd-counting SANet repos ignored)

## Components to build

1. **ResNet-50 backbone (Sec. III-A)** — strides 4/8/16/32/64/128 feature maps (S3FD-M); ImageNet init.
2. **AFFM (Sec. III-B, Fig. 3, Eqs. 1–4)** — project to 512-D, upsample high→low, channel/spatial attention variants, element-wise sum fusion.
3. **Smooth layers (Fig. 2)** — lateral 3×3 on deepest levels without AFFM.
4. **SCEM (Sec. III-C, Fig. 4)** — channel split, Dilated Blocks (d=3 + 3×3, d=2 + 2×2), concat multi-RF features.
5. **Detection heads** — PyramidBox max-in-out classification + box regression per detection layer.
6. **Loss (Eq. 5)** — softmax CE + Smooth-L1 on positives; OHEM 1:3; IoU assign 0.35.
7. **WIDER FACE loader + FDDB eval** — real data under `projects/datasets/`.
8. **Train / eval / predict / report** — JSON metrics + PR / mAP figures (dev-plot).

## Upstream port (if any)

- N/A — reimplement from PDF + `analysis.md`.

## Data

| Dataset | Split | Path | Loader notes |
|---------|-------|------|--------------|
| WIDER FACE | train | `projects/datasets/wider-face/extracted/WIDER_train` + `wider_face_split/wider_face_train_bbx_gt.txt` | primary train |
| WIDER FACE | val | `.../WIDER_val` + `wider_face_val_bbx_gt.txt` | easy/medium/hard mAP |
| FDDB | folds | `projects/datasets/fddb/extracted/` | discontinuous ROC (optional secondary) |

Required corpus for size gate: **WIDER FACE** (train+val annotations/images). FDDB is optional eval.

## Training protocol

- Loss: Eq. (5) multi-task CE + Smooth-L1
- Optimizer: SGD momentum 0.9, weight decay `5e-4`, LR `1e-3`, ×0.1 at 80k and 100k iters
- Iterations: **120k**; batch **14** (paper SANet); input **640×640**
- Anchors: scales `[16,32,64,128,256,512] × 2^(1/3)`, ratio 1; NMS 0.5
- Primary metric: WIDER FACE val mAP easy/medium/hard

## Fidelity plan

- Claim inventory: backbone S3FD-M, AFFM variants, SCEM Dilated Block, loss Eq.5, OHEM, schedule, anchors, max-in-out, multi-scale test
- Passes 1–5 as skill requires; hard stop after Pass 5

## Results figure inventory (paper Results → our plots)

| Paper fig | Type | Metrics / axes | Our path |
|-----------|------|----------------|----------|
| Fig. 6 | PR curves | recall → precision (easy/med/hard) | `outputs/figures/wider_pr.svg` |
| Tables I–II style | grouped bars | mAP by subset / variant | `outputs/figures/wider_map_bars.svg` |
| Fig. 5(d) style | ROC | FDDB discontinuous | `outputs/figures/fddb_disc.svg` |
| Train curve | line | loss vs iter | `outputs/figures/train_loss.svg` |

## Reporting

- Vendored `sanet/plot_style.py` (dev-plot)
- `python -m sanet.report` + `scripts/report.sh`
- `REPORT.md` with real metrics only

## Risks / unknowns

- Exact ResNet stage→detection wiring inferred from Fig. 2 (3 AFFM + 3 Smooth); logged if ambiguous.
- Max-in-out channel counts follow PyramidBox convention (3+3) — paper cites PyramidBox without listing channel count.
- AFW / PASCAL Faces / UFDD not on disk → `deviation:D*` eval skipped.
- Multi-scale test strategy follows S3FD defaults.

## Deviation seeds

- D1: No AFW/PASCAL/UFDD local data
- D2: No official code; PyTorch reimplementation
- D3: Max-in-out channel count assumed 3/3 from PyramidBox


## Size gate decision

- Measured WIDER FACE root: **3.4538 GiB** (`outputs/logs/dataset_size.json`)
- Decision: **full_train** (< 5 GiB)
- Full train command: `CUDA_VISIBLE_DEVICES=2 python -m sanet.train --config configs/default.yaml`
- Log: `outputs/logs/train_full.log` (PID tracked at launch; ~3–4 s/iter ⇒ multi-day wall clock for 120k)
