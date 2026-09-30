"""Multi-task detection loss (Eq. 5) with OHEM 1:3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sanet.models.sanet import SANet, box_iou, max_in_out_scores


class SANetLoss(nn.Module):
    """Softmax CE + Smooth-L1 on positives; OHEM neg:pos = 3:1 (Sec. III-D/E)."""

    def __init__(
        self,
        strides: tuple[int, ...] = SANet.STRIDES,
        anchor_sizes: list[float] | None = None,
        iou_thresh: float = 0.35,
        neg_pos_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.anchor_sizes = anchor_sizes or [s * (2 ** (1 / 3)) for s in SANet.BASE_SCALES]
        self.iou_thresh = iou_thresh
        self.neg_pos_ratio = neg_pos_ratio

    def _make_anchors(self, fh: int, fw: int, stride: int, size: float, device) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(fh, device=device),
            torch.arange(fw, device=device),
            indexing="ij",
        )
        cx = (xs.float() + 0.5) * stride
        cy = (ys.float() + 0.5) * stride
        x1 = cx - size / 2
        y1 = cy - size / 2
        x2 = cx + size / 2
        y2 = cy + size / 2
        return torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

    def _encode(self, anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # anchors/boxes xyxy → deltas
        acx = (anchors[:, 0] + anchors[:, 2]) / 2
        acy = (anchors[:, 1] + anchors[:, 3]) / 2
        aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        gcx = (boxes[:, 0] + boxes[:, 2]) / 2
        gcy = (boxes[:, 1] + boxes[:, 3]) / 2
        gw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        gh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        dx = (gcx - acx) / aw
        dy = (gcy - acy) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def forward(self, outputs: dict, targets: list[dict], max_in: int = 3, max_out: int = 3):
        cls_list, reg_list = outputs["cls"], outputs["reg"]
        device = cls_list[0].device
        total_cls = cls_list[0].new_tensor(0.0)
        total_reg = cls_list[0].new_tensor(0.0)
        n_matched = 0
        batch = cls_list[0].shape[0]

        for b in range(batch):
            gt = targets[b]["boxes"].to(device)
            for i, (cls_map, reg_map) in enumerate(zip(cls_list, reg_list)):
                stride = self.strides[i]
                size = self.anchor_sizes[i]
                _, _, fh, fw = cls_map.shape
                anchors = self._make_anchors(fh, fw, stride, size, device)
                logits2 = max_in_out_scores(cls_map[b : b + 1], max_in, max_out)[0]  # 2,H,W
                logits2 = logits2.permute(1, 2, 0).reshape(-1, 2)  # A,2
                reg = reg_map[b].permute(1, 2, 0).reshape(-1, 4)

                if gt.numel() == 0:
                    # all background — OHEM on hard negatives
                    loss_c = F.cross_entropy(logits2, torch.zeros(logits2.shape[0], device=device, dtype=torch.long), reduction="none")
                    k = min(logits2.shape[0], 32)
                    total_cls = total_cls + loss_c.topk(k).values.mean()
                    continue

                ious = box_iou(anchors, gt)  # A,G
                max_iou, gt_idx = ious.max(dim=1)
                pos = max_iou >= self.iou_thresh
                # ensure each gt has at least one match (best anchor)
                best_anchor = ious.argmax(dim=0)
                pos[best_anchor] = True
                labels = torch.zeros(anchors.shape[0], device=device, dtype=torch.long)
                labels[pos] = 1

                # classification with OHEM
                loss_c_all = F.cross_entropy(logits2, labels, reduction="none")
                n_pos = int(pos.sum().item())
                if n_pos == 0:
                    k = min(anchors.shape[0], 32)
                    total_cls = total_cls + loss_c_all.topk(k).values.mean()
                    continue
                n_neg = min(int(n_pos * self.neg_pos_ratio), int((~pos).sum().item()))
                neg_loss = loss_c_all[~pos]
                if n_neg > 0 and neg_loss.numel() > 0:
                    hard_neg = torch.topk(neg_loss, n_neg).values.sum()
                else:
                    hard_neg = loss_c_all.new_tensor(0.0)
                pos_loss = loss_c_all[pos].sum()
                total_cls = total_cls + (pos_loss + hard_neg) / max(n_pos, 1)

                # regression on positives
                matched = gt[gt_idx[pos]]
                enc = self._encode(anchors[pos], matched)
                total_reg = total_reg + F.smooth_l1_loss(reg[pos], enc, reduction="sum") / max(n_pos, 1)
                n_matched += n_pos

        # Eq. (5): average over mini-batch size M
        m = float(batch)
        loss = (total_cls + total_reg) / m
        return {
            "loss": loss,
            "loss_cls": total_cls / m,
            "loss_reg": total_reg / m,
            "n_matched": n_matched,
        }
