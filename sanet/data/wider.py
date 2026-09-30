"""WIDER FACE dataset loader (real data only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def parse_wider_gt(gt_file: Path) -> list[dict[str, Any]]:
    """Parse WIDER FACE bbxtxt annotation file."""
    records: list[dict[str, Any]] = []
    lines = gt_file.read_text(errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        rel = lines[i].strip()
        i += 1
        if not rel:
            continue
        n = int(lines[i].strip())
        i += 1
        boxes = []
        # WIDER uses n=0 with a dummy line in some releases
        read_n = max(n, 1) if n == 0 else n
        for _ in range(read_n if n > 0 else 1):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            i += 1
            if n == 0:
                break
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            # blur / invalid flags: parts[4] blur, parts[7] invalid in standard format
            invalid = int(parts[7]) if len(parts) > 7 else 0
            if invalid == 1 or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
        records.append({"path": rel, "boxes": boxes})
    return records


class WiderFaceDataset(Dataset):
    """WIDER FACE images + boxes in xyxy absolute pixels."""

    def __init__(
        self,
        image_root: str | Path,
        gt_file: str | Path,
        transforms=None,
        max_samples: int | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transforms = transforms
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"WIDER image root missing: {self.image_root}. "
                "Obtain WIDER FACE under projects/datasets/wider-face/."
            )
        gt_path = Path(gt_file)
        if not gt_path.is_file():
            raise FileNotFoundError(f"WIDER GT missing: {gt_path}")
        self.records = parse_wider_gt(gt_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        path = self.image_root / "images" / rec["path"]
        if not path.is_file():
            # some dumps store images directly under event folders
            alt = self.image_root / rec["path"]
            path = alt if alt.is_file() else path
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "image_id": idx,
            "path": str(rec["path"]),
            "orig_size": torch.tensor([img.height, img.width], dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return {"image": img, "target": target}


def wider_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    return {"images": images, "targets": targets}


def resolve_wider_paths(root: str | Path) -> dict[str, Path]:
    """Resolve WIDER extracted layout under projects/datasets/wider-face."""
    root = Path(root)
    # accept either extracted/ or the dataset root containing extracted/
    if (root / "extracted").is_dir():
        base = root / "extracted"
    else:
        base = root
    split = base / "wider_face_split"
    return {
        "train_images": base / "WIDER_train",
        "val_images": base / "WIDER_val",
        "test_images": base / "WIDER_test",
        "train_gt": split / "wider_face_train_bbx_gt.txt",
        "val_gt": split / "wider_face_val_bbx_gt.txt",
        "base": base,
    }
