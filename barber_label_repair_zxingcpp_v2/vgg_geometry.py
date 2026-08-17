from __future__ import annotations

import collections
import copy
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .data import ImageRecord, Quad, append_jsonl, open_rgb, parse_label, write_json


QR_TYPES = {"qr", "qrcode", "qr code"}
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class VGGGeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class VGGGeometryImage:
    image_id: str
    source_rel: str
    source_image: Path
    source_output_ssim: float
    runner_up_ssim: float | None
    transform: dict
    quads: tuple[Quad, ...]
    objects: tuple[dict, ...]
    invalid_objects: tuple[dict, ...]

    @property
    def complete(self) -> bool:
        return bool(self.quads) and not self.invalid_objects


def _normalize_type(value: object) -> str:
    text = re.sub(r"[_\-]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", text)


def _attr_get_casefold(mapping: object, names: Iterable[str]):
    if not isinstance(mapping, dict):
        return None
    folded = {str(k).strip().lower(): v for k, v in mapping.items()}
    for name in names:
        if name.lower() in folded:
            return folded[name.lower()]
    return None


def _iter_vgg_records(data: object):
    if isinstance(data, dict) and isinstance(data.get("_via_img_metadata"), dict):
        rows = data["_via_img_metadata"]
    elif isinstance(data, dict):
        rows = data
    elif isinstance(data, list):
        rows = {str(i): row for i, row in enumerate(data)}
    else:
        return
    for key, record in rows.items():
        if isinstance(record, dict) and ("filename" in record or "regions" in record):
            yield str(key), record


def _resolve_source_image(filename: object, json_path: Path, root: Path,
                          images_by_basename: dict[str, list[Path]]) -> Path:
    normalized = str(filename).replace("\\", "/")
    for candidate in (json_path.parent / normalized, root / normalized):
        if candidate.is_file():
            return candidate.resolve()
    matches = images_by_basename.get(Path(normalized).name.lower(), [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise VGGGeometryError(f"image not found for VGG record: {normalized}")
    target_parts = [x.lower() for x in Path(normalized).parts]
    ranked = []
    for candidate in matches:
        parts = [x.lower() for x in candidate.relative_to(root).parts]
        common = 0
        for left, right in zip(reversed(parts), reversed(target_parts)):
            if left != right:
                break
            common += 1
        ranked.append((common, str(candidate), candidate))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise VGGGeometryError(f"ambiguous source basename: {normalized}")
    return ranked[0][2]


def _numeric_polygon(xs: object, ys: object) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(xs, list) or not isinstance(ys, list):
        return None, "all_points_x/all_points_y are not lists"
    if len(xs) != len(ys):
        return None, f"x/y vertex count differs: {len(xs)} vs {len(ys)}"
    try:
        points = np.asarray(list(zip(xs, ys)), dtype=np.float64)
    except (TypeError, ValueError):
        return None, "polygon has non-numeric coordinate"
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        return None, "polygon has invalid coordinate shape/value"
    if len(points) == 5 and np.allclose(points[0], points[-1], atol=0, rtol=0):
        points = points[:-1]
    if len(points) != 4:
        return points, f"QR polygon has {len(points)} vertices after safe normalization"
    if len(np.unique(np.round(points, 8), axis=0)) != 4:
        return points, "QR polygon has duplicate vertices"
    return points, None


def _object_id(source_rel: str, region_key: str, points: np.ndarray | None) -> str:
    if points is None:
        token = "missing"
    else:
        token = ",".join(f"{float(x):.5f}" for x in points.reshape(-1))
    digest = hashlib.sha1(token.encode("ascii")).hexdigest()[:10]
    return f"{source_rel}#region_{region_key}_{digest}"


def load_barber_annotations(root: Path) -> tuple[list[dict], list[dict]]:
    """Read VGG QR geometry without retaining or emitting barcode payloads."""
    root = root.resolve()
    image_paths = sorted({p.resolve() for p in root.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})
    by_basename: dict[str, list[Path]] = collections.defaultdict(list)
    for path in image_paths:
        by_basename[path.name.lower()].append(path)
    records: list[dict] = []
    parse_errors: list[dict] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for record_key, record in _iter_vgg_records(data):
            filename = record.get("filename")
            if not filename:
                continue
            try:
                source_image = _resolve_source_image(
                    filename, json_path, root, by_basename)
            except VGGGeometryError as exc:
                parse_errors.append({"json": str(json_path), "record": record_key,
                                     "error": str(exc)})
                continue
            regions = record.get("regions", [])
            if isinstance(regions, dict):
                region_items = sorted(regions.items(), key=lambda row: str(row[0]))
            elif isinstance(regions, list):
                region_items = list(enumerate(regions))
            else:
                region_items = []
            objects: list[dict] = []
            invalid_objects: list[dict] = []
            saw_qr = False
            source_rel = source_image.relative_to(root).as_posix()
            for region_key, region in region_items:
                if not isinstance(region, dict):
                    continue
                attrs = region.get("region_attributes", {})
                raw_type = _attr_get_casefold(attrs, ("type", "barcode_type", "class"))
                if _normalize_type(raw_type) not in QR_TYPES:
                    continue
                saw_qr = True
                shape = region.get("shape_attributes", {})
                xs = _attr_get_casefold(shape, ("all_points_x",))
                ys = _attr_get_casefold(shape, ("all_points_y",))
                points, error = _numeric_polygon(xs, ys)
                region_key_text = str(region_key)
                instance_id = _object_id(source_rel, region_key_text, points)
                item = {"instance_id": instance_id, "region_key": region_key_text,
                        "polygon": points}
                if error is None:
                    objects.append(item)
                else:
                    invalid = {**item, "error": error,
                               "raw_vertex_count": int(len(points)) if points is not None else None}
                    invalid_objects.append(invalid)
                    parse_errors.append({"json": str(json_path), "record": record_key,
                                         "region": region_key_text, "source_rel": source_rel,
                                         "error": error,
                                         "raw_vertex_count": invalid["raw_vertex_count"]})
            if saw_qr:
                records.append({"source_image": source_image, "source_rel": source_rel,
                                "source_basename": source_image.name,
                                "json_path": json_path, "record_key": record_key,
                                "objects": objects, "invalid_objects": invalid_objects})
    if not records:
        raise VGGGeometryError(f"no BarBeR QR VGG records found under {root}")

    # BarBeR may repeat the same annotation in aggregate and per-folder JSON.
    merged: dict[str, dict] = {}
    for record in records:
        key = str(record["source_image"])
        target = merged.get(key)
        if target is None:
            target = copy.deepcopy(record)
            target["json_paths"] = [str(record["json_path"])]
            merged[key] = target
        else:
            target["json_paths"].append(str(record["json_path"]))
            for field in ("objects", "invalid_objects"):
                def fingerprint(item):
                    points = item.get("polygon")
                    coords = (tuple(round(float(x), 5) for x in points.reshape(-1))
                              if isinstance(points, np.ndarray) else None)
                    return coords, item.get("error")
                known = {fingerprint(item) for item in target[field]}
                for item in record[field]:
                    token = fingerprint(item)
                    if token not in known:
                        target[field].append(copy.deepcopy(item))
                        known.add(token)
    return sorted(merged.values(), key=lambda row: row["source_rel"]), parse_errors


def build_record_indices(records: Iterable[dict]) -> dict[str, dict[str, list[dict]]]:
    indices = {"relative": collections.defaultdict(list),
               "basename": collections.defaultdict(list),
               "stem": collections.defaultdict(list)}
    for record in records:
        indices["relative"][record["source_rel"].lower()].append(record)
        indices["basename"][record["source_basename"].lower()].append(record)
        indices["stem"][Path(record["source_basename"]).stem.lower()].append(record)
    return indices


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise VGGGeometryError(f"invalid JSONL {path}:{line_no}") from exc
    return rows


def processed_meta(dataset: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for split in SPLITS:
        for row in _read_jsonl(dataset / split / "annotations.jsonl"):
            image_value = str(row.get("image", ""))
            if not image_value:
                continue
            key = (split, Path(image_value).name.lower())
            if key in out:
                raise VGGGeometryError(f"duplicate processed metadata: {key}")
            out[key] = row
    return out


def _source_hints(meta: dict, output_path: Path) -> list[str]:
    hints = []
    for key in ("source_image", "source_path", "original_image", "source_id"):
        value = meta.get(key) if isinstance(meta, dict) else None
        if value:
            hints.append(str(value).split("#region_")[0].replace("\\", "/"))
    hints.extend((output_path.name, output_path.stem))
    return hints


def rotate_points_clockwise(points: np.ndarray, width: int, height: int,
                            angle: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    if angle == 0:
        return p.copy()
    if angle == 90:
        return np.stack([height - 1.0 - y, x], axis=1)
    if angle == 180:
        return np.stack([width - 1.0 - x, height - 1.0 - y], axis=1)
    if angle == 270:
        return np.stack([y, width - 1.0 - x], axis=1)
    raise ValueError(angle)


def render_to_processed(source: Image.Image, out_width: int, out_height: int,
                        pad_value: int) -> tuple[Image.Image, dict]:
    source = source.convert("RGB")
    width, height = source.size
    rotated = width > height
    image = (source.transpose(Image.Transpose.ROTATE_270) if rotated else source.copy())
    rotated_width, rotated_height = image.size
    scale = min(out_width / float(rotated_width), out_height / float(rotated_height))
    new_width = max(1, int(round(rotated_width * scale)))
    new_height = max(1, int(round(rotated_height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    pad_left = (out_width - new_width) // 2
    pad_top = (out_height - new_height) // 2
    canvas = Image.new("RGB", (out_width, out_height), (pad_value,) * 3)
    canvas.paste(resized, (pad_left, pad_top))
    transform = {"rotated_landscape_cw": rotated,
                 "source_width": width, "source_height": height,
                 "rotated_width": rotated_width, "rotated_height": rotated_height,
                 "scale": scale, "resized_width": new_width,
                 "resized_height": new_height, "pad_left": pad_left,
                 "pad_top": pad_top, "output_width": out_width,
                 "output_height": out_height}
    return canvas, transform


def transform_points_to_processed(points: np.ndarray, transform: dict) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    if transform["rotated_landscape_cw"]:
        out = rotate_points_clockwise(out, int(transform["source_width"]),
                                      int(transform["source_height"]), 90)
    out[:, 0] = out[:, 0] * float(transform["scale"]) + int(transform["pad_left"])
    out[:, 1] = out[:, 1] * float(transform["scale"]) + int(transform["pad_top"])
    return out


def _gray_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def gray_ssim(left: Image.Image, right: Image.Image) -> float:
    a, b = _gray_array(left), _gray_array(right)
    if a.shape != b.shape:
        return -1.0
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    cov = float(np.mean((a - mean_a) * (b - mean_b)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mean_a * mean_b + c1) * (2 * cov + c2) /
            ((mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)))


def resolve_record(output_path: Path, meta: dict, indices: dict,
                   out_width: int, out_height: int, pad_value: int,
                   min_ssim: float, min_margin: float = .015):
    candidates, seen = [], set()
    for hint in _source_hints(meta, output_path):
        normalized = hint.lower().lstrip("./")
        values = []
        values.extend(indices["relative"].get(normalized, []))
        values.extend(indices["basename"].get(Path(normalized).name, []))
        values.extend(indices["stem"].get(Path(normalized).stem, []))
        for record in values:
            token = record["source_rel"]
            if token not in seen:
                candidates.append(record)
                seen.add(token)
    if not candidates:
        raise VGGGeometryError("no source VGG record mapping")
    target = open_rgb(output_path)
    scored = []
    for record in candidates:
        source = open_rgb(record["source_image"])
        rendered, transform = render_to_processed(source, out_width, out_height, pad_value)
        if rendered.size != target.size:
            continue
        scored.append((gray_ssim(rendered, target), record, transform))
    if not scored:
        raise VGGGeometryError("source candidates cannot be rendered")
    scored.sort(key=lambda row: (-row[0], row[1]["source_rel"]))
    best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None
    if best[0] < min_ssim:
        raise VGGGeometryError(f"source/output transform mismatch: SSIM {best[0]:.5f}")
    if runner_up is not None and best[0] - runner_up < min_margin:
        raise VGGGeometryError(
            f"ambiguous source mapping: SSIM {best[0]:.5f} vs {runner_up:.5f}")
    return best[0], runner_up, best[1], best[2]


def _quad_from_points(points: np.ndarray, width: int, height: int) -> Quad:
    tokens = tuple((f"{float(x) / width:.8f}", f"{float(y) / height:.8f}")
                   for x, y in points)
    return Quad("0", tokens, np.asarray(points, dtype=np.float64))


def _manifest_row(rec: ImageRecord, score: float, runner_up: float | None,
                  record: dict, transform: dict) -> dict:
    objects = []
    for obj in sorted(record["objects"], key=lambda item: item["instance_id"]):
        source_points = np.asarray(obj["polygon"], dtype=np.float64)
        processed_points = transform_points_to_processed(source_points, transform)
        quad = _quad_from_points(processed_points, int(transform["output_width"]),
                                 int(transform["output_height"]))
        objects.append({"instance_id": obj["instance_id"],
                        "region_key": obj["region_key"],
                        "source_points": source_points.tolist(),
                        "processed_points": processed_points.tolist(),
                        "token_points": [list(pair) for pair in quad.token_points]})
    invalid = []
    for obj in sorted(record["invalid_objects"], key=lambda item: item["instance_id"]):
        raw = obj.get("polygon")
        processed = (transform_points_to_processed(raw, transform).tolist()
                     if isinstance(raw, np.ndarray) and len(raw) else None)
        invalid.append({"instance_id": obj["instance_id"],
                        "region_key": obj["region_key"], "error": obj["error"],
                        "raw_vertex_count": obj["raw_vertex_count"],
                        "source_points": raw.tolist() if isinstance(raw, np.ndarray) else None,
                        "processed_points": processed})
    return {"image_id": rec.image_id, "split": rec.split, "stem": rec.stem,
            "processed_image": str(rec.image_path),
            "source_image": str(record["source_image"]),
            "source_rel": record["source_rel"], "source_output_ssim": score,
            "runner_up_ssim": runner_up, "transform": transform,
            "objects": objects, "invalid_objects": invalid,
            "complete": bool(objects) and not invalid}


def _quad_set_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    best = None
    for order in itertools.permutations(range(4)):
        distances = np.linalg.norm(left - right[list(order)], axis=1)
        key = (float(distances.max()), float(distances.mean()))
        if best is None or key < best:
            best = key
    assert best is not None
    return best[1], best[0]


def cross_validate_gold(rows: list[dict], v3: dict[tuple[str, str], Path],
                        records: list[ImageRecord], tolerance_px: float = .55,
                        expected_gold_images: int = 799,
                        expected_gold_instances: int = 915) -> dict:
    by_id = {row["image_id"]: row for row in rows}
    record_by_key = {(rec.split, rec.stem): rec for rec in records}
    errors = []
    matched_instances = 0
    maximum_error = 0.0
    for key, gold_path in sorted(v3.items()):
        rec = record_by_key[key]
        row = by_id.get(rec.image_id)
        if row is None:
            errors.append({"image_id": rec.image_id, "error": "missing VGG geometry"})
            continue
        if row["invalid_objects"]:
            errors.append({"image_id": rec.image_id,
                           "error": "gold image has invalid VGG object"})
            continue
        gold = parse_label(gold_path, *open_rgb(rec.image_path).size)
        manual = [np.asarray(obj["processed_points"], dtype=np.float64)
                  for obj in row["objects"]]
        if len(gold) != len(manual):
            errors.append({"image_id": rec.image_id, "error": "instance count mismatch",
                           "gold": len(gold), "vgg": len(manual)})
            continue
        used = set()
        for index, gold_quad in enumerate(gold):
            candidates = []
            for manual_index, manual_quad in enumerate(manual):
                mean_error, max_error = _quad_set_error(gold_quad.points, manual_quad)
                candidates.append((max_error, mean_error, manual_index))
            candidates.sort()
            max_error, mean_error, manual_index = candidates[0]
            if manual_index in used or max_error > tolerance_px:
                errors.append({"image_id": rec.image_id, "gold_instance": index,
                               "error": "V3 gold does not equal transformed VGG vertices",
                               "best_mean_px": mean_error, "best_max_px": max_error})
                continue
            used.add(manual_index)
            matched_instances += 1
            maximum_error = max(maximum_error, max_error)
    report = {"passed": (not errors and len(v3) == expected_gold_images and
                         matched_instances == expected_gold_instances),
              "gold_images": len(v3), "expected_gold_images": expected_gold_images,
              "matched_instances": matched_instances,
              "expected_gold_instances": expected_gold_instances,
              "tolerance_px": tolerance_px,
              "maximum_vertex_error_px": maximum_error,
              "error_count": len(errors), "errors": errors[:100]}
    return report


def build_geometry(dataset: Path, barber_root: Path, work: Path,
                   records: list[ImageRecord], v3: dict[tuple[str, str], Path],
                   min_ssim: float = .90, pad_value: int = 127) -> tuple[dict, dict]:
    annotations, parse_errors = load_barber_annotations(barber_root)
    indices = build_record_indices(annotations)
    metadata = processed_meta(dataset)
    rows, failures = [], []
    manifest_path = work / "vgg_geometry_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            try:
                image = open_rgb(rec.image_path)
                score, runner_up, record, transform = resolve_record(
                    rec.image_path, metadata.get((rec.split, rec.image_path.name.lower()), {}),
                    indices, image.width, image.height, pad_value, min_ssim)
                row = _manifest_row(rec, score, runner_up, record, transform)
                rows.append(row)
                append_jsonl(handle, row)
            except Exception as exc:
                failures.append({"image_id": rec.image_id, "error": str(exc)})
    with (work / "vgg_geometry_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            append_jsonl(handle, row)
    with (work / "vgg_parse_errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in parse_errors:
            append_jsonl(handle, row)
    cross_validation = cross_validate_gold(rows, v3, records)
    write_json(work / "vgg_gold_cross_validation_report.json", cross_validation)
    report = {"dataset_images": len(records), "resolved_images": len(rows),
              "unresolved_images": len(failures),
              "complete_geometry_images": sum(bool(row["complete"]) for row in rows),
              "valid_instances": sum(len(row["objects"]) for row in rows),
              "invalid_vgg_instances": sum(len(row["invalid_objects"]) for row in rows),
              "parse_error_records": len(parse_errors), "min_source_ssim": min_ssim,
              "gold_cross_validation_passed": cross_validation["passed"]}
    write_json(work / "vgg_geometry_report.json", report)
    if not cross_validation["passed"]:
        raise VGGGeometryError(
            f"VGG->240x320 gold cross-validation failed: {cross_validation['error_count']} errors")
    return {row["image_id"]: geometry_from_row(row) for row in rows}, report


def geometry_from_row(row: dict) -> VGGGeometryImage:
    quads = tuple(Quad("0", tuple(tuple(pair) for pair in obj["token_points"]),
                       np.asarray(obj["processed_points"], dtype=np.float64))
                  for obj in row["objects"])
    return VGGGeometryImage(
        image_id=row["image_id"], source_rel=row["source_rel"],
        source_image=Path(row["source_image"]),
        source_output_ssim=float(row["source_output_ssim"]),
        runner_up_ssim=(float(row["runner_up_ssim"])
                        if row.get("runner_up_ssim") is not None else None),
        transform=row["transform"], quads=quads,
        objects=tuple(row["objects"]), invalid_objects=tuple(row["invalid_objects"]))


def load_geometry(work: Path) -> dict[str, VGGGeometryImage]:
    path = work / "vgg_geometry_manifest.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"missing/empty VGG geometry manifest: {path}")
    return {row["image_id"]: geometry_from_row(row) for row in rows}
