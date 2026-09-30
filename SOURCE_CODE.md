# Upstream / open-source notes

## Official code

No official GitHub release found for Shi, Xu, Kakadiaris SANet (ICB 2019 / TBIOM 2020).

## Related repos (not used)

| Repo | Why ignored |
|------|-------------|
| `BIGKnight/SANet_implementation` | Crowd-counting SANet (different paper/task), TensorFlow |

## This project

Faithful PyTorch reimplementation from the TBIOM PDF + `papers/.../analysis.md`, using:

- ResNet-50 ImageNet init (Sec. III-E)
- AFFM / SCEM as in Figs. 3–4
- S3FD-M multi-scale detection layout (Sec. III-A)
- PyramidBox max-in-out (cited; channel count assumed 3+3 → **D3**)
