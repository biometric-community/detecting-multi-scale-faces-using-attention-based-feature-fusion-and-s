"""Shared config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_data_root(cfg: dict[str, Any], project_root: Path) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        cand = (project_root / root).resolve()
        if cand.exists():
            return cand
        # monorepo fallback
        repo = project_root
        while repo != repo.parent:
            if (repo / "docs" / "PAPERS.md").exists():
                break
            repo = repo.parent
        cand2 = (repo / "projects" / "datasets" / "wider-face").resolve()
        if cand2.exists():
            return cand2
        return cand
    return root
