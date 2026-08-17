from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import (discover_dataset, discover_v3_labels, open_rgb,
                   parse_label, write_json)
from .review import make_sheet
from .vgg_geometry import _quad_set_error, load_geometry


P_COLORS = ("#ff3030", "#28b463", "#2471a3", "#f4d03f")
V_COLOR = "#ff4fd8"


def _match_instances(gold, geometry):
    if len(gold) != len(geometry):
        return None, float("inf")
    best = None
    for order in itertools.permutations(range(len(geometry))):
        errors = [_quad_set_error(gold[i].points, geometry[j].points)
                  for i, j in enumerate(order)]
        key = (max((value[1] for value in errors), default=0.0),
               sum(value[0] for value in errors), order)
        if best is None or key < best:
            best = key
    assert best is not None
    return best[2], float(best[0])


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, label: str,
                color: str, offset: tuple[int, int]) -> None:
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=color, outline="black", width=1)
    draw.text((x + offset[0], y + offset[1]), label, fill=color,
              stroke_width=2, stroke_fill="black", font=ImageFont.load_default())


def annotate_gold_geometry(image: Image.Image, gold, geometry,
                           assignment, max_error: float, scale: int = 3) -> Image.Image:
    canvas = image.resize((image.width * scale, image.height * scale),
                          Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if assignment is not None:
        for gold_index, geometry_index in enumerate(assignment):
            quad = geometry[geometry_index]
            points = [(float(x) * scale, float(y) * scale) for x, y in quad.points]
            draw.line(points + [points[0]], fill=V_COLOR, width=2)
            for vertex, (x, y) in enumerate(points):
                _draw_point(draw, x, y, f"V{vertex}#{geometry_index}", V_COLOR,
                            (-42, -16))
    for gold_index, quad in enumerate(gold):
        points = [(float(x) * scale, float(y) * scale) for x, y in quad.points]
        draw.line(points + [points[0]], fill="white", width=2)
        for semantic, (x, y) in enumerate(points):
            _draw_point(draw, x, y, f"P{semantic}#{gold_index}",
                        P_COLORS[semantic], (7, 2))
    text = (f"V3 proposed gold P0-P3; VGG raw V0-V3; max-set-error={max_error:.6f}px"
            if np.isfinite(max_error) else
            "V3 proposed gold P0-P3; VGG geometry missing/count mismatch")
    draw.text((5, 5), text, fill="white", stroke_width=2,
              stroke_fill="black", font=font)
    return canvas


def build_gold_geometry_pack(dataset: Path, v3_work: Path, work: Path,
                             output: Path, image_ids: list[str] | None = None,
                             tolerance_px: float = .55) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review output: {output}")
    records = {record.image_id: record for record in discover_dataset(dataset)}
    v3 = discover_v3_labels(v3_work)
    geometry = load_geometry(work)
    selected = sorted(image_ids or
                      [f"{split}/{stem}" for split, stem in v3])
    per_image = output / "per_image"
    per_image.mkdir(parents=True)
    items = []
    rows = []
    for image_id in selected:
        if image_id not in records:
            raise KeyError(f"unknown dataset image_id: {image_id}")
        rec = records[image_id]
        gold_path = v3.get((rec.split, rec.stem))
        if gold_path is None:
            raise KeyError(f"image is not V3 proposed gold: {image_id}")
        image = open_rgb(rec.image_path)
        gold = parse_label(gold_path, *image.size)
        geometry_image = geometry.get(image_id)
        quads = list(geometry_image.quads) if geometry_image is not None else []
        assignment, max_error = _match_instances(gold, quads)
        annotated = annotate_gold_geometry(
            image, gold, quads, assignment, max_error)
        filename = image_id.replace("/", "__") + ".jpg"
        annotated.save(per_image / filename, quality=92)
        items.append((image_id, annotated.copy()))
        rows.append({
            "image_id": image_id,
            "gold_source": "v3_work/proposed",
            "old_dataset_txt_used_as_gold": False,
            "gold_instances": len(gold),
            "geometry_instances": len(quads),
            "assignment_gold_to_geometry": list(assignment)
                if assignment is not None else None,
            "maximum_vertex_set_error_px": max_error,
            "passed": bool(assignment is not None and max_error <= tolerance_px),
        })
    make_sheet(items, output / "gold_geometry_sheet.jpg", columns=4,
               cell_width=740, cell_height=1020)
    report = {
        "passed": all(row["passed"] for row in rows),
        "images": len(rows),
        "gold_source": "v3_work/proposed",
        "old_dataset_txt_used_as_gold": False,
        "tolerance_px": tolerance_px,
        "maximum_vertex_set_error_px": max(
            (row["maximum_vertex_set_error_px"] for row in rows), default=0.0),
        "records": rows,
    }
    write_json(output / "gold_geometry_review_report.json", report)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--v3-work", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", action="append", dest="image_ids")
    parser.add_argument("--tolerance-px", type=float, default=.55)
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    report = build_gold_geometry_pack(
        args.dataset.resolve(), args.v3_work.resolve(), args.work.resolve(),
        args.output.resolve(), args.image_ids, args.tolerance_px)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
