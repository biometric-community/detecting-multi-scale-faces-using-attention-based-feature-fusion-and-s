"""Run SANet inference on images and save detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from sanet.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from sanet.engine.config import load_config
from sanet.models import SANet


@torch.no_grad()
def predict_image(model: SANet, path: Path, size: int, device, score_thresh: float, nms_thresh: float):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img_r = img.resize((size, size), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    out = model(tensor)
    preds = model.predict_boxes(out, (size, size), score_thresh, nms_thresh)[0]
    boxes = preds["boxes"].cpu()
    scores = preds["scores"].cpu()
    # map to original resolution
    boxes[:, [0, 2]] *= w / size
    boxes[:, [1, 3]] *= h / size
    return img, boxes, scores


def main() -> None:
    ap = argparse.ArgumentParser(description="SANet predict")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--input", type=str, required=True, help="Image file or directory")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = SANet(attention_mode=cfg["model"].get("attention_mode", "sa"), pretrained_backbone=False).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else project_root / cfg["paths"]["checkpoint_dir"] / "latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    inp = Path(args.input)
    paths = sorted(inp.glob("*")) if inp.is_dir() else [inp]
    paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not paths:
        raise FileNotFoundError(f"No images under {inp}")
    out_dir = Path(args.out_dir) if args.out_dir else project_root / cfg["paths"]["pred_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["model"]["input_size"]
    score_thresh = cfg["eval"].get("score_thresh", 0.05)
    nms_thresh = cfg["eval"].get("nms_thresh", 0.5)

    manifest = []
    for p in paths:
        img, boxes, scores = predict_image(model, p, size, device, score_thresh, nms_thresh)
        draw = ImageDraw.Draw(img)
        dets = []
        for box, sc in zip(boxes.tolist(), scores.tolist()):
            draw.rectangle(box, outline=(0, 51, 204), width=3)
            dets.append({"box": box, "score": sc})
        out_img = out_dir / f"{p.stem}_det.jpg"
        img.save(out_img)
        out_json = out_dir / f"{p.stem}_det.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"image": str(p), "detections": dets}, f, indent=2)
        manifest.append({"image": str(p), "out_image": str(out_img), "n": len(dets)})
        print(f"{p.name}: {len(dets)} faces -> {out_img}")
    with open(out_dir / "predict_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
