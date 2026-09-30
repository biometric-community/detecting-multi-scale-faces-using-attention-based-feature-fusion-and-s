# SANet Face Detection Report

Rebuilt from TBIOM 2020 SANet (Shi, Xu, Kakadiaris). Metrics below are from **our** runs on real WIDER FACE data — not fabricated.

## Setup

- Attention mode: `sa`
- Input size: 640
- Train batch / max_iters: 14 / 120000


## Training status

- Dataset size gate: **3.45 GiB** → full train required and **started**.
- Log: `outputs/logs/train_full.log` (120k iters, micro-batch 2 × accum 7 = batch 14).
- Eval/PR figures below from **smoke** (20 iters) until full train + `scripts/eval.sh` complete — not paper-protocol accuracy.

## Results summary

### Training

- Final iter: 20
- Final loss: 40.44554138183594
- Train images: 64
- Wall time (s): 15.409796714782715

### WIDER FACE val mAP (Ours)

| Subset | mAP |
|--------|-----|
| easy | 0.0002 |
| medium | 0.0001 |
| hard | 0.0000 |
| all | 0.0000 |

Paper-reported SANet WIDER **test** (for reference only): easy 94.6 / medium 93.8 / hard 88.2.

## Figures

### `outputs/figures/train_loss.svg`

![](outputs/figures/train_loss.svg)

### `outputs/figures/wider_map_bars.svg`

![](outputs/figures/wider_map_bars.svg)

### `outputs/figures/wider_pr.svg`

![](outputs/figures/wider_pr.svg)

## Regenerate

```bash
bash scripts/report.sh
```
