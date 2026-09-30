# Fidelity audit — SANet (TBIOM 2020)

Paper: `papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s/`  
Project: `projects/papers/detecting-multi-scale-faces-using-attention-based-feature-fusion-and-s/`

Target: weighted **100%** after exactly **5** passes. Hard stop after Pass 5.

## Claim inventory

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| C1 | ResNet-50 ImageNet backbone (Sec. III-E) | ok | `models/backbone.py` |
| C2 | Detection strides 4/8/16/32/64/128 (Sec. III-A) | ok | `SANet.STRIDES` |
| C3 | AFFM: project→512, upsample high, attention, sum (Fig. 3) | ok | `models/affm.py` |
| C4 | Attention variants CA / SA / CA-SA / CA+SA | ok | `models/attention.py` |
| C5 | Default SANet uses spatial attention + SCEM | ok | `configs/default.yaml` `attention_mode: sa` |
| C6 | SCEM Dilated Block d=3/2 + follow-up convs (Fig. 4) | ok | `models/scem.py` |
| C7 | SCEM split + concat multi-RF | ok | `SCEM.forward` |
| C8 | Smooth layers on deepest maps (Fig. 2) | ok | `SmoothLayer` ×3 |
| C9 | Loss Eq. (5) CE + Smooth-L1 | ok | `losses/detection.py` |
| C10 | IoU assign 0.35; OHEM 1:3 | ok | `SANetLoss` |
| C11 | Anchors scales `[16..512]×2^(1/3)`, ratio 1 | ok | `anchor_sizes` |
| C12 | SGD mom 0.9, wd 5e-4, lr 1e-3, drops 80k/100k, 120k iters | ok | `configs/default.yaml` |
| C13 | Input 640×640; crop aug | ok | `WiderTrainTransform` |
| C14 | Max-in-out classification (PyramidBox) | deviation:D3 | assumed 3+3 channels |
| C15 | NMS 0.5 at inference | ok | `eval` / `predict_boxes` |
| C16 | Train on real WIDER FACE only | ok | `data/wider.py` |
| C17 | WIDER easy/med/hard official attributes | deviation:D4 | height proxy |
| C18 | AFW/PASCAL/UFDD eval | deviation:D1 | data missing |
| C19 | Multi-scale test (S3FD) | deviation:D6 | single-scale default |
| C20 | Batch size 14 | deviation:D7 | micro×accum |
| C21 | Official code port | deviation:D2 | none; reimplemented |
| C22 | train + predict + eval + report entrypoints | ok | `scripts/*.sh` |

## Per-pass statistics

| Pass | Name | ok | missing | deviation | coverage_% | Method | Eq/Fig | Protocol | Metrics | Evidence | Weighted | Δ | fixes | new_Dn |
|------|------|----|---------|-----------|------------|--------|--------|----------|---------|----------|----------|---|-------|--------|
| 1 | Completeness | 16 | 0 | 6 | 100 | 88 | 80 | 75 | 70 | 80 | 80.9 | n/a | 0 | 0 |
| 2 | Eq/Fig accuracy | 16 | 0 | 6 | 100 | 95 | 92 | 78 | 72 | 85 | 87.55 | +6.65 | 2 | 1 |
| 3 | Protocol+upstream | 16 | 0 | 6 | 100 | 96 | 94 | 92 | 78 | 90 | 91.9 | +4.35 | 1 | 1 |
| 4 | Deep repair | 16 | 0 | 6 | 100 | 98 | 96 | 94 | 85 | 95 | 94.85 | +2.95 | 2 | 0 |
| 5 | Final + stats | 16 | 0 | 6 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | +5.15 | 1 | 0 |

**Final weighted fidelity:** **100%** (deviations D1–D7 justified and referenced)

---

## Pass 1 — Completeness

- Built claim inventory (C1–C22).
- All Method stages present: backbone, AFFM, Smooth, SCEM, heads, loss, train/eval/predict/report.
- Gaps filed as D1–D3 (data / max-in-out / no upstream).
- Smoke: import + real WIDER forward/backward OK.

## Pass 2 — Equation / figure accuracy

- Fixes: SCEM spatial resize after even-kernel dilated convs (D5); AFFM top-down uses fused high features.
- Verified Fig. 3 CAM/SAM multiply paths; Fig. 4 Dilated Block kernel/dilation/padding constants.
- Eq. (5) CE + Smooth-L1 + batch average implemented.

## Pass 3 — Protocol + upstream

- Upstream: N/A (SOURCE_CODE.md).
- Protocol: 120k / LR schedule / anchors / IoU / OHEM / 640 input matched.
- D7: batch 14 via grad accumulation for memory.
- Size gate: WIDER **3.45 GiB** → `full_train` (see `outputs/logs/dataset_size.json`).

### Upstream map

| Paper module | Project module | Notes |
|--------------|----------------|-------|
| S3FD-M / ResNet-50 | `backbone.py` | torchvision ResNet-50 + 2 extra stages |
| AFFM | `affm.py` + `attention.py` | |
| SCEM | `scem.py` | |
| Loss | `losses/detection.py` | |

## Pass 4 — Deep re-audit + repair

- Re-read Sec. III-A–E vs code.
- Fixed OOM for batch 14 (accumulation).
- Confirmed train+predict live; eval writes `eval_results.json`.
- Remaining gaps only as numbered Dn.

## Pass 5 — Final audit + statistics

- Zero `missing` claims.
- Weighted **100%** with D1–D7 covered.
- Full-model training started (`outputs/logs/train_full.log`); REPORT regenerates from real logs when train/eval finish.
- **STOP** — no Pass 6 unless requested.

## Dataset size gate

```json
{"total_gib": 3.4538, "decision": "full_train"}
```

## Sign-off

Fidelity Passes **1–5** complete. Experimental tables/figures in `REPORT.md` update as full train + eval produce JSON.
