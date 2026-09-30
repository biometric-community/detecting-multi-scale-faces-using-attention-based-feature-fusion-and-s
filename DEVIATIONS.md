# Deviations from the paper

| ID | Description | Severity | Mitigation |
|----|-------------|----------|------------|
| D1 | AFW, PASCAL Faces, UFDD not present under `projects/datasets/` | eval coverage | Skip those benchmarks; WIDER (+ optional FDDB) only |
| D2 | No official code; PyTorch reimplementation from PDF | medium | Cite sections/figures; fidelity audit |
| D3 | Max-in-out channel counts assumed **3 face + 3 bg** (PyramidBox convention); paper does not state integers | low | Documented; configurable in model |
| D4 | WIDER easy/medium/hard split approximated by face height on the 640 canvas when attribute fields are not carried | medium | Prefer official WIDER eval toolkit when available |
| D5 | Dilated 2×2 layers may change HxW; we bilinear-resize back to the input map size | low | Keeps SCEM spatially aligned |
| D6 | Multi-scale testing follows a single 640 scale by default in `eval.py` (S3FD multi-scale can be enabled later) | medium | Logged; single-scale still valid for protocol smoke |
| D7 | Paper batch size 14 realized as `micro_batch_size=2` × `grad_accum=7` (+ AMP) to fit GPU memory | low | Effective samples/step match paper batch |

Intentional non-goals: matching published mAP numbers exactly (structural fidelity is the default skill target).
