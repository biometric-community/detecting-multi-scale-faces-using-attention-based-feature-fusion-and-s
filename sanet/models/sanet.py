"""SANet detector (Fig. 2): backbone → AFFM/Smooth → SCEM → heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .affm import AFFM, SmoothLayer
from .backbone import ResNet50Backbone
from .scem import SCEM


class DetectHead(nn.Module):
    """Per-layer classification (max-in-out) + box regression."""

    def __init__(
        self,
        in_channels: int = 512,
        num_anchors: int = 1,
        max_in: int = 3,
        max_out: int = 3,
    ) -> None:
        super().__init__()
        # PyramidBox max-in-out: max_in face channels + max_out background channels
        self.max_in = max_in
        self.max_out = max_out
        cls_ch = num_anchors * (max_in + max_out)
        self.cls = nn.Conv2d(in_channels, cls_ch, kernel_size=3, padding=1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg


def max_in_out_scores(cls: torch.Tensor, max_in: int = 3, max_out: int = 3) -> torch.Tensor:
    """Reduce max-in-out channels to 2-class logits [bg, face] per anchor.

    cls: (N, A*(max_in+max_out), H, W) with A=1 typically.
    returns: (N, 2, H, W) as [background, face]
    """
    n, c, h, w = cls.shape
    per = max_in + max_out
    a = c // per
    cls = cls.view(n, a, per, h, w)
    face = cls[:, :, :max_in].max(dim=2).values  # (N,A,H,W)
    bg = cls[:, :, max_in:].max(dim=2).values
    # collapse anchors (A=1) → (N,2,H,W)
    face = face.max(dim=1).values
    bg = bg.max(dim=1).values
    return torch.stack([bg, face], dim=1)


class SANet(nn.Module):
    """Full SANet face detector (TBIOM 2020)."""

    STRIDES = (4, 8, 16, 32, 64, 128)
    BASE_SCALES = (16, 32, 64, 128, 256, 512)

    def __init__(
        self,
        attention_mode: str = "sa",
        pretrained_backbone: bool = True,
        feat_channels: int = 512,
        max_in: int = 3,
        max_out: int = 3,
    ) -> None:
        super().__init__()
        self.attention_mode = attention_mode
        self.feat_channels = feat_channels
        self.max_in = max_in
        self.max_out = max_out
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        # channels from backbone
        chs = [256, 512, 1024, 2048, 512, 256]
        # Deepest three: Smooth; shallow three: AFFM top-down (Fig. 2)
        self.smooth5 = SmoothLayer(chs[5], feat_channels)
        self.smooth4 = SmoothLayer(chs[4], feat_channels)
        self.smooth3 = SmoothLayer(chs[3], feat_channels)
        self.affm2 = AFFM(chs[2], feat_channels, feat_channels, attention_mode)
        self.affm1 = AFFM(chs[1], feat_channels, feat_channels, attention_mode)
        self.affm0 = AFFM(chs[0], feat_channels, feat_channels, attention_mode)
        self.scems = nn.ModuleList([SCEM(feat_channels) for _ in range(6)])
        self.heads = nn.ModuleList(
            [DetectHead(feat_channels, 1, max_in, max_out) for _ in range(6)]
        )
        self._init_heads()
        scale_factor = 2 ** (1.0 / 3.0)
        self.anchor_sizes = [s * scale_factor for s in self.BASE_SCALES]

    def _init_heads(self) -> None:
        for m in self.heads.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def fuse(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        c0, c1, c2, c3, c4, c5 = feats
        p5 = self.smooth5(c5)
        p4 = self.smooth4(c4)
        p3 = self.smooth3(c3)
        # top-down AFFM: high semantic into lower resolution maps
        p2 = self.affm2(c2, p3)
        p1 = self.affm1(c1, p2)
        p0 = self.affm0(c0, p1)
        return [p0, p1, p2, p3, p4, p5]

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        feats = self.backbone(x)
        fused = self.fuse(feats)
        cls_list, reg_list, det_feats = [], [], []
        for i, f in enumerate(fused):
            d = self.scems[i](f)
            # keep spatial size stable if DilatedBlock pads oddly
            if d.shape[-2:] != f.shape[-2:]:
                d = F.interpolate(d, size=f.shape[-2:], mode="bilinear", align_corners=False)
            det_feats.append(d)
            cls, reg = self.heads[i](d)
            cls_list.append(cls)
            reg_list.append(reg)
        return {"cls": cls_list, "reg": reg_list, "feats": det_feats}

    @torch.no_grad()
    def predict_boxes(
        self,
        outputs: dict[str, list[torch.Tensor]],
        image_size: tuple[int, int],
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        top_k: int = 750,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode detections for a batch. image_size = (H, W) of network input."""
        cls_list, reg_list = outputs["cls"], outputs["reg"]
        device = cls_list[0].device
        batch = cls_list[0].shape[0]
        h_img, w_img = image_size
        results = []
        for b in range(batch):
            boxes_all, scores_all = [], []
            for i, (cls_map, reg_map) in enumerate(zip(cls_list, reg_list)):
                stride = self.STRIDES[i]
                size = self.anchor_sizes[i]
                logits = max_in_out_scores(cls_map[b : b + 1], self.max_in, self.max_out)[0]
                prob = torch.softmax(logits, dim=0)[1]  # face
                reg = reg_map[b]
                fh, fw = prob.shape
                ys, xs = torch.meshgrid(
                    torch.arange(fh, device=device),
                    torch.arange(fw, device=device),
                    indexing="ij",
                )
                cx = (xs.float() + 0.5) * stride
                cy = (ys.float() + 0.5) * stride
                # decode deltas (Girshick / Faster R-CNN style)
                dx, dy, dw, dh = reg[0], reg[1], reg[2], reg[3]
                pred_cx = dx * size + cx
                pred_cy = dy * size + cy
                pred_w = torch.exp(dw.clamp(max=4)) * size
                pred_h = torch.exp(dh.clamp(max=4)) * size
                x1 = pred_cx - pred_w / 2
                y1 = pred_cy - pred_h / 2
                x2 = pred_cx + pred_w / 2
                y2 = pred_cy + pred_h / 2
                mask = prob > score_thresh
                if mask.any():
                    boxes_all.append(torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=1))
                    scores_all.append(prob[mask])
            if not boxes_all:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                    }
                )
                continue
            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)
            boxes[:, 0::2].clamp_(0, w_img - 1)
            boxes[:, 1::2].clamp_(0, h_img - 1)
            keep = nms(boxes, scores, nms_thresh)[:top_k]
            results.append({"boxes": boxes[keep], "scores": scores[keep]})
        return results


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    try:
        from torchvision.ops import nms as tv_nms

        return tv_nms(boxes, scores, iou_thresh)
    except Exception:
        # fallback greedy
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            iou = box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            order = rest[iou <= iou_thresh]
        return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b - inter + 1e-6)
