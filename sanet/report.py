"""Generate REPORT.md + SVG figures from real train/eval logs (dev-plot)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sanet.plot_style import (
    BAR_COLOR,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    FIGSIZE,
    LABEL_SIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
)


def _load_json(path: Path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_train_loss(history, figdir: Path) -> Path | None:
    if not history:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    iters = [h["iter"] for h in history]
    loss = [h["loss"] for h in history]
    ax.plot(iters, loss, color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN, label="train loss")
    apply_style(ax)
    ax.set_title("SANet training loss", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = figdir / "train_loss.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_wider_map_bars(eval_payload, figdir: Path, paper_maps: dict | None = None) -> Path | None:
    if not eval_payload:
        return None
    maps = eval_payload.get("map", {})
    keys = [k for k in ("easy", "medium", "hard") if k in maps]
    if not keys:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = list(range(len(keys)))
    vals = [maps[k] * 100 for k in keys]
    ax.bar([i - 0.15 for i in x], vals, width=0.3, color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, label="Ours")
    if paper_maps:
        pvals = [paper_maps.get(k, 0) for k in keys]
        ax.bar([i + 0.15 for i in x], pvals, width=0.3, color=MULTI_SERIES_COLORS[1], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, label="Paper SANet (reported)")
    apply_style(ax, grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("mAP (%)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("WIDER FACE validation mAP", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = figdir / "wider_map_bars.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_pr(eval_payload, figdir: Path) -> Path | None:
    if not eval_payload or "pr" not in eval_payload:
        return None
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, name in enumerate(("easy", "medium", "hard")):
        pr = eval_payload["pr"].get(name)
        if not pr or not pr.get("recall"):
            continue
        ax.plot(
            pr["recall"],
            pr["precision"],
            color=MULTI_SERIES_COLORS[i % len(MULTI_SERIES_COLORS)],
            linewidth=LINEWIDTH_MAIN,
            label=f"{name} (AP={eval_payload['map'].get(name, 0)*100:.1f})",
        )
    apply_style(ax)
    ax.set_xlabel("Recall", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title("WIDER FACE PR curves (Ours)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out = figdir / "wider_pr.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def write_report(project_root: Path, cfg: dict, figures: list[Path], train_s, eval_s) -> Path:
    title = cfg.get("report", {}).get("title") or cfg.get("paper", {}).get("title") or "SANet Report"
    lines = [
        f"# {title}",
        "",
        "Rebuilt from TBIOM 2020 SANet (Shi, Xu, Kakadiaris). Metrics below are from **our** runs on real WIDER FACE data — not fabricated.",
        "",
        "## Setup",
        "",
        f"- Attention mode: `{cfg['model'].get('attention_mode', 'sa')}`",
        f"- Input size: {cfg['model']['input_size']}",
        f"- Train batch / max_iters: {cfg['train']['batch_size']} / {cfg['train']['max_iters']}",
        "",
        "## Results summary",
        "",
    ]
    if train_s:
        lines += [
            "### Training",
            "",
            f"- Final iter: {train_s.get('final_iter')}",
            f"- Final loss: {train_s.get('final_loss')}",
            f"- Train images: {train_s.get('num_train_images')}",
            f"- Wall time (s): {train_s.get('seconds')}",
            "",
        ]
    else:
        lines += ["### Training", "", "BLOCKED or not run yet — no `train_summary.json`.", ""]
    if eval_s and "map" in eval_s:
        m = eval_s["map"]
        lines += [
            "### WIDER FACE val mAP (Ours)",
            "",
            "| Subset | mAP |",
            "|--------|-----|",
            f"| easy | {m.get('easy', float('nan')):.4f} |",
            f"| medium | {m.get('medium', float('nan')):.4f} |",
            f"| hard | {m.get('hard', float('nan')):.4f} |",
            f"| all | {m.get('all', float('nan')):.4f} |",
            "",
            "Paper-reported SANet WIDER **test** (for reference only): easy 94.6 / medium 93.8 / hard 88.2.",
            "",
        ]
    else:
        lines += ["### WIDER FACE val mAP", "", "BLOCKED — run `scripts/eval.sh` after training.", ""]

    lines += ["## Figures", ""]
    for fig in figures:
        rel = fig.relative_to(project_root).as_posix()
        lines += [f"### `{rel}`", "", f"![]({rel})", ""]
    lines += [
        "## Regenerate",
        "",
        "```bash",
        "bash scripts/report.sh",
        "```",
        "",
    ]
    out = project_root / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SANet report")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--logs", type=str, default=None)
    ap.add_argument("--figdir", type=str, default=None)
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    from sanet.engine.config import load_config

    cfg = load_config(project_root / args.config)
    log_dir = Path(args.logs) if args.logs else project_root / cfg["paths"]["log_dir"]
    figdir = Path(args.figdir) if args.figdir else project_root / cfg["paths"]["figure_dir"]
    figdir.mkdir(parents=True, exist_ok=True)

    history = _load_json(log_dir / "train_history.json") or []
    train_s = _load_json(log_dir / "train_summary.json")
    eval_s = _load_json(log_dir / "eval_results.json")

    figures = []
    for fn in (
        plot_train_loss(history, figdir),
        plot_wider_map_bars(eval_s, figdir, paper_maps={"easy": 94.6, "medium": 93.8, "hard": 88.2}),
        plot_pr(eval_s, figdir),
    ):
        if fn is not None:
            figures.append(fn)

    report = write_report(project_root, cfg, figures, train_s, eval_s)
    print("Wrote", report)
    for f in figures:
        print("Figure", f)


if __name__ == "__main__":
    main()
