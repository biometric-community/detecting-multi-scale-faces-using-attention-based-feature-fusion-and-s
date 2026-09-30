"""WIDER FACE-style AP evaluation (IoU 0.5)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def average_precision(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Compute VOC-style AP and a PR curve over a list of images."""
    scores, matches = [], []
    n_gt = 0
    for pb, ps, gb in zip(pred_boxes, pred_scores, gt_boxes):
        n_gt += len(gb)
        if len(pb) == 0:
            continue
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        matched_gt = np.zeros(len(gb), dtype=bool)
        ious = _iou_matrix(pb, gb) if len(gb) else np.zeros((len(pb), 0))
        for i in range(len(pb)):
            scores.append(float(ps[i]))
            if len(gb) == 0:
                matches.append(0)
                continue
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thresh and not matched_gt[j]:
                matches.append(1)
                matched_gt[j] = True
            else:
                matches.append(0)
    if n_gt == 0:
        return {"ap": 0.0, "recall": [], "precision": [], "n_gt": 0}
    if not scores:
        return {"ap": 0.0, "recall": [0.0], "precision": [0.0], "n_gt": n_gt}
    order = np.argsort(-np.asarray(scores))
    matches = np.asarray(matches, dtype=np.float64)[order]
    tp = np.cumsum(matches)
    fp = np.cumsum(1.0 - matches)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-6)
    # VOC 11-point-ish interpolated AP (continuous)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {
        "ap": ap,
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "n_gt": int(n_gt),
    }


def split_wider_difficulty(gt_boxes: np.ndarray) -> dict[str, np.ndarray]:
    """Approximate WIDER easy/medium/hard by face height (pixels on eval canvas).

    Official WIDER uses annotation attributes; height proxy is a documented deviation
    when attribute fields are not carried through the loader.
    """
    if gt_boxes.size == 0:
        empty = gt_boxes.reshape(0, 4)
        return {"easy": empty, "medium": empty, "hard": empty}
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    return {
        "easy": gt_boxes[h >= 50],
        "medium": gt_boxes[h >= 30],
        "hard": gt_boxes,  # all faces
    }
