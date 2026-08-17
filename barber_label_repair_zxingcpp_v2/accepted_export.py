from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data import (ImageRecord, append_jsonl, discover_dataset, file_hash,
                   open_rgb, parse_label, snapshot, write_json)
from .geometry import quad_geometry_issue
from .review import COLORS, make_sheet, read_jsonl


SPLITS = ("train", "val", "test")


def _accepted_labels(work: Path) -> dict[tuple[str, str], Path]:
    root = work / "combined_proposed"
    accepted: dict[tuple[str, str], Path] = {}
    for split in SPLITS:
        label_dir = root / split / "labels"
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.txt")):
            key = (split, path.stem)
            if key in accepted:
                raise RuntimeError(f"duplicate accepted label: {split}/{path.stem}")
            accepted[key] = path
    if not accepted:
        raise FileNotFoundError(f"no accepted labels below {root}")
    return accepted


def _class_id(token: str):
    try:
        return int(token)
    except ValueError:
        return token


def _annotate(image: Image.Image, quads, scale: int = 2) -> Image.Image:
    canvas = image.resize((image.width * scale, image.height * scale),
                          Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for instance_index, quad in enumerate(quads):
        points = [(float(x) * scale, float(y) * scale) for x, y in quad.points]
        draw.line(points + [points[0]], fill="white", width=max(2, scale))
        for semantic_index, (x, y) in enumerate(points):
            color = COLORS[semantic_index]
            radius = 5 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=color, outline="black", width=1)
            draw.text((x + radius + 1, y - radius),
                      f"#{instance_index}:P{semantic_index}", fill=color,
                      stroke_width=2, stroke_fill="black", font=font)
    return canvas


def _validate_reports(work: Path, expected_images: int, expected_instances: int,
                      expected_dropped_images: int, expected_gold_instances: int,
                      expected_accepted_za_instances: int,
                      expected_dropped_instances: int,
                      expected_dropped_non_za_instances: int,
                      expected_dropped_embedded_za_instances: int) -> dict:
    recovery = json.loads((work / "recovery_report.json").read_text(encoding="utf-8"))
    validation = json.loads((work / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise RuntimeError("validation_report passed != true")
    if validation.get("old_txt_geometry_used") is not False:
        raise RuntimeError("validation_report old_txt_geometry_used != false")
    if validation.get("vgg_gold_cross_validation_passed") is not True:
        raise RuntimeError("VGG gold cross-validation did not pass")
    if recovery.get("combined_proposed_images") != expected_images:
        raise RuntimeError("unexpected combined_proposed image count")
    if recovery.get("combined_proposed_instances") != expected_instances:
        raise RuntimeError("unexpected combined_proposed instance count")
    if recovery.get("old_txt_geometry_used") is not False:
        raise RuntimeError("recovery_report old_txt_geometry_used != false")
    dropped = recovery.get("ZB_images", 0) + recovery.get("ZM_images", 0)
    if dropped != expected_dropped_images:
        raise RuntimeError(f"unexpected dropped image count: {dropped}")
    accepted_za_instances = (recovery.get("combined_proposed_instances", 0) -
                             expected_gold_instances)
    dropped_non_za_instances = (recovery.get("ZB_instances", 0) +
                                recovery.get("ZM_instances", 0))
    dropped_embedded_za_instances = (recovery.get("ZA_instances", 0) -
                                     accepted_za_instances)
    dropped_instances = (recovery.get("input_failed_instances", 0) -
                         accepted_za_instances)
    actual = (accepted_za_instances, dropped_instances, dropped_non_za_instances,
              dropped_embedded_za_instances)
    expected = (expected_accepted_za_instances, expected_dropped_instances,
                expected_dropped_non_za_instances,
                expected_dropped_embedded_za_instances)
    if actual != expected:
        raise RuntimeError(f"unexpected accepted/dropped instance accounting: {actual}")
    return recovery


def _source_for(record: ImageRecord, recovered: set[tuple[str, str]]) -> str:
    return "ZXing_ZA" if (record.split, record.stem) in recovered else "V3_gold"


def _ordered_keys(accepted: dict[tuple[str, str], Path]) -> list[tuple[str, str]]:
    return [
        (split, stem) for split in SPLITS
        for stem in sorted(value for current_split, value in accepted
                           if current_split == split)
    ]


def export_accepted(dataset: Path, work: Path, output: Path,
                    expected_images: int = 1027, expected_instances: int = 1144,
                    expected_dropped_images: int = 192,
                    expected_v3_images: int = 799, expected_za_images: int = 228,
                    expected_gold_instances: int = 915,
                    expected_accepted_za_instances: int = 229,
                    expected_dropped_instances: int = 246,
                    expected_dropped_non_za_instances: int = 237,
                    expected_dropped_embedded_za_instances: int = 9,
                    page_size: int = 20, columns: int = 4,
                    drop_out_of_bounds: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if page_size < 1 or columns < 1:
        raise ValueError("page_size and columns must be positive")

    recovery = _validate_reports(
        work, expected_images, expected_instances, expected_dropped_images,
        expected_gold_instances, expected_accepted_za_instances,
        expected_dropped_instances, expected_dropped_non_za_instances,
        expected_dropped_embedded_za_instances)
    records = {(record.split, record.stem): record
               for record in discover_dataset(dataset)}
    accepted = _accepted_labels(work)
    if len(accepted) != expected_images:
        raise RuntimeError(f"accepted label count {len(accepted)} != {expected_images}")

    recovered_root = work / "recovered_labels"
    recovered = {(split, path.stem) for split in SPLITS
                 for path in (recovered_root / split / "labels").glob("*.txt")}
    failure_ids = {row["image_id"] for row in read_jsonl(work / "recovery_failures.jsonl")}
    accepted_ids = {f"{split}/{stem}" for split, stem in accepted}
    overlap = sorted(accepted_ids & failure_ids)
    if overlap:
        raise RuntimeError(f"dropped image entered accepted set: {overlap[:5]}")
    if len(failure_ids) != expected_dropped_images:
        raise RuntimeError(
            f"failure report images {len(failure_ids)} != {expected_dropped_images}")

    ordered_keys = _ordered_keys(accepted)
    candidate_quads = {}
    candidate_source_counts = {"V3_gold": 0, "ZXing_ZA": 0}
    candidate_source_instances = {"V3_gold": 0, "ZXing_ZA": 0}
    boundary_rejections = []
    for key in ordered_keys:
        record = records.get(key)
        if record is None:
            raise FileNotFoundError(f"accepted label has no source image: {key}")
        image = open_rgb(record.image_path)
        if image.size != (240, 320):
            raise ValueError(f"unexpected image size {record.image_id}: {image.size}")
        quads = parse_label(accepted[key], *image.size)
        source = _source_for(record, recovered)
        candidate_source_counts[source] += 1
        candidate_source_instances[source] += len(quads)
        outside = []
        for instance_index, quad in enumerate(quads):
            issue = quad_geometry_issue(quad.points)
            if issue is not None:
                raise ValueError(f"invalid accepted quad {record.image_id}: {issue}")
            for point_index, (x, y) in enumerate(quad.points):
                if x < 0 or x >= image.width or y < 0 or y >= image.height:
                    outside.append({
                        "instance": instance_index, "point": point_index,
                        "x": float(x), "y": float(y),
                    })
        if outside:
            boundary_rejections.append({
                "image_id": record.image_id, "source": source,
                "instances": len(quads), "image_width": image.width,
                "image_height": image.height, "outside_points": outside,
                "policy": "drop_entire_image_no_coordinate_clipping",
            })
        else:
            candidate_quads[key] = quads

    if candidate_source_counts != {"V3_gold": expected_v3_images,
                                   "ZXing_ZA": expected_za_images}:
        raise RuntimeError(f"unexpected candidate source counts: {candidate_source_counts}")
    if candidate_source_instances != {
            "V3_gold": expected_gold_instances,
            "ZXing_ZA": expected_accepted_za_instances}:
        raise RuntimeError(
            f"unexpected candidate source instances: {candidate_source_instances}")
    if boundary_rejections and not drop_out_of_bounds:
        first = boundary_rejections[0]["image_id"]
        raise RuntimeError(
            f"{len(boundary_rejections)} accepted images have out-of-bounds points; "
            f"first={first}; rerun only with explicit --drop-out-of-bounds")

    boundary_rejected_instances = sum(row["instances"] for row in boundary_rejections)
    exported_images = expected_images - len(boundary_rejections)
    exported_instances = expected_instances - boundary_rejected_instances
    rejected_source_counts = {"V3_gold": 0, "ZXing_ZA": 0}
    rejected_source_instances = {"V3_gold": 0, "ZXing_ZA": 0}
    for row in boundary_rejections:
        rejected_source_counts[row["source"]] += 1
        rejected_source_instances[row["source"]] += row["instances"]
    expected_output_source_counts = {
        key: candidate_source_counts[key] - rejected_source_counts[key]
        for key in candidate_source_counts
    }
    expected_output_source_instances = {
        key: candidate_source_instances[key] - rejected_source_instances[key]
        for key in candidate_source_instances
    }

    immutable_paths = []
    for key, label in accepted.items():
        record = records.get(key)
        if record is None:
            raise FileNotFoundError(f"accepted label has no source image: {key}")
        immutable_paths.extend((record.image_path, record.label_path, label))
    before = snapshot(immutable_paths)

    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale building directory exists: {building}")
    source_counts = {"V3_gold": 0, "ZXing_ZA": 0}
    source_instance_counts = {"V3_gold": 0, "ZXing_ZA": 0}
    split_counts = {split: {"images": 0, "instances": 0} for split in SPLITS}
    total_instances = 0
    page_items: list[tuple[str, Image.Image]] = []
    preview_pages = []
    preview_records = []

    def flush_page() -> None:
        if not page_items:
            return
        page_number = len(preview_pages) + 1
        relative = Path("preview") / "pages" / f"accepted_page_{page_number:03d}.jpg"
        make_sheet(page_items, building / relative, columns=columns)
        preview_pages.append(str(relative))
        page_items.clear()

    try:
        building.mkdir(parents=True)
        annotation_handles = {}
        try:
            for split in SPLITS:
                path = building / split / "annotations.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                annotation_handles[split] = path.open("w", encoding="utf-8")
            manifest = (building / "accepted_manifest.jsonl").open("w", encoding="utf-8")
            try:
                for key in ordered_keys:
                    if key not in candidate_quads:
                        continue
                    record = records[key]
                    accepted_label = accepted[key]
                    image = open_rgb(record.image_path)
                    quads = candidate_quads[key]

                    source = _source_for(record, recovered)
                    source_counts[source] += 1
                    source_instance_counts[source] += len(quads)
                    split_counts[record.split]["images"] += 1
                    split_counts[record.split]["instances"] += len(quads)
                    total_instances += len(quads)

                    image_dst = building / record.split / "images" / record.image_path.name
                    label_dst = building / record.split / "labels" / f"{record.stem}.txt"
                    image_dst.parent.mkdir(parents=True, exist_ok=True)
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(record.image_path, image_dst)
                    shutil.copy2(accepted_label, label_dst)

                    annotation = {
                        "image": f"images/{record.image_path.name}",
                        "width": image.width,
                        "height": image.height,
                        "image_id": record.image_id,
                        "label_source": source,
                        "instances": [
                            {"class_id": _class_id(quad.class_token),
                             "corners": [[float(x), float(y)]
                                         for x, y in quad.points]}
                            for quad in quads
                        ],
                    }
                    append_jsonl(annotation_handles[record.split], annotation)
                    append_jsonl(manifest, {
                        "image_id": record.image_id, "source": source,
                        "instances": len(quads),
                        "image_sha256": file_hash(image_dst),
                        "label_sha256": file_hash(label_dst),
                    })

                    annotated = _annotate(image, quads)
                    preview_relative = (Path("preview") / "per_image" / record.split /
                                        f"{record.stem}.jpg")
                    preview_path = building / preview_relative
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    annotated.save(preview_path, quality=92)
                    page_items.append((record.image_id, annotated))
                    preview_records.append({
                        "image_id": record.image_id,
                        "preview": str(preview_relative),
                        "instances": len(quads), "source": source,
                    })
                    if len(page_items) == page_size:
                        flush_page()
                flush_page()
            finally:
                manifest.close()
        finally:
            for handle in annotation_handles.values():
                handle.close()

        if total_instances != exported_instances:
            raise RuntimeError(
                f"accepted instance count {total_instances} != {exported_instances}")
        if source_counts != expected_output_source_counts:
            raise RuntimeError(f"unexpected source counts: {source_counts}")
        if source_instance_counts != expected_output_source_instances:
            raise RuntimeError(
                f"unexpected source instance counts: {source_instance_counts}")
        if sum(x["images"] for x in split_counts.values()) != exported_images:
            raise RuntimeError("split image counts do not sum to expected total")

        with (building / "boundary_rejections.jsonl").open(
                "w", encoding="utf-8") as handle:
            for row in boundary_rejections:
                append_jsonl(handle, row)

        preview_report = {
            "passed": True, "images": exported_images,
            "instances": total_instances,
            "per_image_previews": len(preview_records),
            "page_size": page_size, "pages": len(preview_pages),
            "page_files": preview_pages,
            "records_manifest": "preview/preview_manifest.jsonl",
        }
        with (building / "preview" / "preview_manifest.jsonl").open(
                "w", encoding="utf-8") as handle:
            for row in preview_records:
                append_jsonl(handle, row)
        write_json(building / "preview" / "preview_report.json", preview_report)

        report = {
            "passed": True,
            "candidate_images": expected_images,
            "candidate_instances": expected_instances,
            "candidate_sources": candidate_source_counts,
            "candidate_source_instances": candidate_source_instances,
            "images": exported_images,
            "instances": total_instances, "sources": source_counts,
            "source_instances": source_instance_counts,
            "splits": split_counts,
            "audit_dropped_images": expected_dropped_images,
            "audit_dropped_instances": expected_dropped_instances,
            "additional_boundary_dropped_images": len(boundary_rejections),
            "additional_boundary_dropped_instances": boundary_rejected_instances,
            "dropped_images": expected_dropped_images + len(boundary_rejections),
            "dropped_instances": (expected_dropped_instances +
                                  boundary_rejected_instances),
            "dropped_non_za_instances": expected_dropped_non_za_instances,
            "dropped_embedded_za_instances": expected_dropped_embedded_za_instances,
            "boundary_rejections": "boundary_rejections.jsonl",
            "audit_instance_grades": {
                "ZA": recovery.get("ZA_instances", 0),
                "ZB": recovery.get("ZB_instances", 0),
                "ZM": recovery.get("ZM_instances", 0),
            },
            "accepted_labels_source": str(work / "combined_proposed"),
            "old_dataset_txt_used_as_geometry": False,
            "original_inputs_unchanged": True,
            "preview": preview_report,
        }
        write_json(building / "accepted_export_report.json", report)
        if before != snapshot(immutable_paths):
            raise RuntimeError("source images/labels changed during export")
        building.replace(output)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=1027)
    parser.add_argument("--expected-instances", type=int, default=1144)
    parser.add_argument("--expected-dropped-images", type=int, default=192)
    parser.add_argument("--expected-v3-images", type=int, default=799)
    parser.add_argument("--expected-za-images", type=int, default=228)
    parser.add_argument("--expected-gold-instances", type=int, default=915)
    parser.add_argument("--expected-accepted-za-instances", type=int, default=229)
    parser.add_argument("--expected-dropped-instances", type=int, default=246)
    parser.add_argument("--expected-dropped-non-za-instances", type=int, default=237)
    parser.add_argument("--expected-dropped-embedded-za-instances", type=int, default=9)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--drop-out-of-bounds", action="store_true")
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    result = export_accepted(
        args.dataset.resolve(), args.work.resolve(), args.output.resolve(),
        expected_images=args.expected_images,
        expected_instances=args.expected_instances,
        expected_dropped_images=args.expected_dropped_images,
        expected_v3_images=args.expected_v3_images,
        expected_za_images=args.expected_za_images,
        expected_gold_instances=args.expected_gold_instances,
        expected_accepted_za_instances=args.expected_accepted_za_instances,
        expected_dropped_instances=args.expected_dropped_instances,
        expected_dropped_non_za_instances=args.expected_dropped_non_za_instances,
        expected_dropped_embedded_za_instances=(
            args.expected_dropped_embedded_za_instances),
        page_size=args.page_size, columns=args.columns,
        drop_out_of_bounds=args.drop_out_of_bounds)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
