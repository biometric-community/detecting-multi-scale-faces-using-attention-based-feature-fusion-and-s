"""Image transforms for SANet training (640×640, crop augmentation)."""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WiderTrainTransform:
    """Random crop / resize to square input with box filtering (paper Sec. III-A)."""

    def __init__(self, size: int = 640, min_face: float = 8.0) -> None:
        self.size = size
        self.min_face = min_face

    def __call__(self, img: Image.Image, target: dict[str, Any]):
        boxes = target["boxes"].clone()
        w, h = img.size
        # random square crop biased to contain faces when possible
        if random.random() < 0.5 and boxes.numel() > 0:
            scale = random.uniform(0.3, 1.0)
            crop = int(min(w, h) * scale)
            crop = max(crop, 16)
            # pick a face center
            bi = random.randrange(boxes.shape[0])
            cx = float((boxes[bi, 0] + boxes[bi, 2]) / 2)
            cy = float((boxes[bi, 1] + boxes[bi, 3]) / 2)
            x0 = int(max(0, min(cx - crop / 2, w - crop)))
            y0 = int(max(0, min(cy - crop / 2, h - crop)))
        else:
            crop = min(w, h)
            x0 = random.randint(0, max(0, w - crop))
            y0 = random.randint(0, max(0, h - crop))
        img = img.crop((x0, y0, x0 + crop, y0 + crop))
        if boxes.numel() > 0:
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0
            boxes[:, [0, 2]].clamp_(0, crop)
            boxes[:, [1, 3]].clamp_(0, crop)
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw >= self.min_face) & (bh >= self.min_face)
            boxes = boxes[keep]
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if boxes.numel() > 0:
            s = self.size / float(crop)
            boxes = boxes * s
        if random.random() < 0.5:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = self.size - x2
                boxes[:, 2] = self.size - x1
        # color jitter light
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = dict(target)
        target["boxes"] = boxes
        target["size"] = torch.tensor([self.size, self.size], dtype=torch.int64)
        return tensor, target


class WiderEvalTransform:
    def __init__(self, size: int = 640) -> None:
        self.size = size

    def __call__(self, img: Image.Image, target: dict[str, Any]):
        w, h = img.size
        img_r = img.resize((self.size, self.size), Image.BILINEAR)
        tensor = TF.to_tensor(img_r)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= self.size / w
            boxes[:, [1, 3]] *= self.size / h
        target = dict(target)
        target["boxes"] = boxes
        target["size"] = torch.tensor([self.size, self.size], dtype=torch.int64)
        target["scale_w"] = torch.tensor(w / self.size, dtype=torch.float32)
        target["scale_h"] = torch.tensor(h / self.size, dtype=torch.float32)
        return tensor, target
