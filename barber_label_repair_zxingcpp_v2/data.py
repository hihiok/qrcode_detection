from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import numpy as np


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Quad:
    class_token: str
    token_points: tuple[tuple[str, str], ...]
    points: np.ndarray

    def reordered_line(self, semantic_to_manual: tuple[int, ...]) -> str:
        vals = [self.class_token]
        for i in semantic_to_manual:
            vals.extend(self.token_points[i])
        return " ".join(vals)


@dataclass(frozen=True)
class ImageRecord:
    split: str
    stem: str
    image_path: Path
    label_path: Path

    @property
    def image_id(self) -> str:
        return f"{self.split}/{self.stem}"


def parse_label(path: Path, width: int, height: int) -> list[Quad]:
    quads: list[Quad] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        t = text.split()
        if len(t) != 9:
            raise ValueError(f"{path}:{line_no}: expected 9 tokens, got {len(t)}")
        pairs = tuple((t[i], t[i + 1]) for i in range(1, 9, 2))
        try:
            p = np.array([[float(x) * width, float(y) * height] for x, y in pairs])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric coordinate") from exc
        if not np.isfinite(p).all() or len(np.unique(np.round(p, 6), axis=0)) != 4:
            raise ValueError(f"{path}:{line_no}: invalid/duplicate quad vertices")
        quads.append(Quad(t[0], pairs, p))
    if not quads:
        raise ValueError(f"{path}: no valid QR instances")
    return quads


def discover_dataset(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing label for {image_path}")
            records.append(ImageRecord(split, image_path.stem, image_path, label_path))
    if not records:
        raise FileNotFoundError(f"no split/images + split/labels dataset under {root}")
    return records


def discover_v3_labels(v3_work: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    candidates = [p for p in v3_work.rglob("*.txt") if "proposed" in "/".join(p.parts).lower()]
    for p in candidates:
        split = next((s for s in ("train", "val", "test") if s in p.parts), None)
        if split is None:
            continue
        key = (split, p.stem)
        if key in out and out[key].read_bytes() != p.read_bytes():
            raise RuntimeError(f"ambiguous V3 label copies for {split}/{p.stem}")
        out[key] = p
    if not out:
        raise FileNotFoundError(f"no proposed V3 labels found under {v3_work}")
    return out


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(p): file_hash(p) for p in sorted(set(paths))}


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_jsonl(handle, obj: object) -> None:
    handle.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
