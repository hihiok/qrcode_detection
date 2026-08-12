from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data import (append_jsonl, discover_dataset, file_hash, open_rgb,
                   parse_label, snapshot, write_json)
from .geometry import quad_geometry_issue
from .pipeline import Config, image_records, immutable_paths


COLORS = ("#ff3030", "#28b463", "#2471a3", "#f4d03f")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def candidate_text(instance: dict) -> str:
    order = instance.get("semantic_to_manual")
    if order is None:
        return "no candidate"
    return "P0/P1/P2/P3 <- V" + "/V".join(str(x) for x in order)


def annotate_record(image: Image.Image, quads, report: dict, scale: int = 2) -> Image.Image:
    canvas = image.resize((image.width * scale, image.height * scale),
                          Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, quad in enumerate(quads):
        inst = report["instances"][index] if index < len(report["instances"]) else {}
        order = inst.get("semantic_to_manual")
        semantic_at_manual = ({int(manual): semantic
                               for semantic, manual in enumerate(order)}
                              if order is not None else {})
        points = [(float(x) * scale, float(y) * scale) for x, y in quad.points]
        draw.line(points + [points[0]], fill="white", width=max(2, scale))
        for manual_index, point in enumerate(points):
            semantic = semantic_at_manual.get(manual_index)
            label = (f"P{semantic}/V{manual_index}" if semantic is not None
                     else f"V{manual_index}")
            color = COLORS[semantic] if semantic is not None else "white"
            x, y = point
            radius = 5 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=color, outline="black", width=1)
            draw.text((x + radius + 1, y - radius), label, fill=color,
                      stroke_width=2, stroke_fill="black", font=font)
        header = f"#{index} {inst.get('grade', 'ZM')} {candidate_text(inst)}"
        draw.text((4, 4 + index * 14), header, fill="white",
                  stroke_width=2, stroke_fill="black", font=font)
    return canvas


def make_sheet(items: list[tuple[str, Image.Image]], output: Path,
               columns: int = 4, cell_width: int = 520, cell_height: int = 700) -> None:
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (title, image) in enumerate(items):
        col, row = index % columns, index // columns
        x0, y0 = col * cell_width, row * cell_height
        image.thumbnail((cell_width - 12, cell_height - 36), Image.Resampling.LANCZOS)
        sheet.paste(image, (x0 + 6, y0 + 28))
        draw.text((x0 + 6, y0 + 6), title, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def build_review_pack(dataset: Path, work: Path, page_size: int = 20) -> dict:
    records = {r.image_id: r for r in discover_dataset(dataset)}
    failures = read_jsonl(work / "recovery_failures.jsonl")
    preview_root = work / "manual_review_pack" / "previews"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_root.mkdir(parents=True, exist_ok=True)
    template_path = (work / "manual_review_pack" /
                     "manual_review_decisions_TEMPLATE.jsonl")
    grade_items: dict[str, list[tuple[str, Image.Image]]] = {"ZB": [], "ZM": []}
    with template_path.open("w", encoding="utf-8") as template:
        for report in failures:
            image_id = report["image_id"]
            rec = records[image_id]
            image = open_rgb(rec.image_path)
            try:
                quads = parse_label(rec.label_path, *image.size)
                annotated = annotate_record(image, quads, report)
            except ValueError:
                annotated = image.resize((image.width * 2, image.height * 2))
            grade = report.get("grade", "ZM")
            grade_items.setdefault(grade, []).append((image_id, annotated))
            append_jsonl(template, {
                "image_id": image_id,
                "audit_grade": grade,
                "reviewer": "",
                "action": "pending",
                "orders": [inst.get("semantic_to_manual")
                           for inst in report.get("instances", [])],
                "instance_grades": [inst.get("grade")
                                    for inst in report.get("instances", [])],
                "note": "Use approve_suggestion, manual_order, or corrected_label."
            })
    pages = []
    for grade, items in grade_items.items():
        for start in range(0, len(items), page_size):
            page = start // page_size + 1
            out = preview_root / f"{grade.lower()}_page_{page:03d}.jpg"
            make_sheet(items[start:start + page_size], out)
            pages.append(str(out))
    report = {"review_images": len(failures),
              "ZB_images": len(grade_items.get("ZB", [])),
              "ZM_images": len(grade_items.get("ZM", [])),
              "decision_template": str(template_path), "preview_pages": pages}
    write_json(work / "manual_review_pack" / "review_pack_report.json", report)
    return report


def decision_label(rec, image, decision: dict, failure: dict, work: Path) -> tuple[Path | None, str | None]:
    reviewer = str(decision.get("reviewer", "")).strip()
    if not reviewer:
        return None, "reviewer is empty"
    action = decision.get("action")
    if action == "corrected_label":
        path = work / "manual_corrected_labels" / rec.split / "labels" / f"{rec.stem}.txt"
        if not path.is_file():
            return None, f"missing corrected label: {path}"
        return path, None
    try:
        quads = parse_label(rec.label_path, *image.size)
    except ValueError as exc:
        return None, f"original label cannot be reordered: {exc}"
    if action == "approve_suggestion":
        orders = [x.get("semantic_to_manual") for x in failure.get("instances", [])]
    elif action == "manual_order":
        orders = decision.get("orders")
    else:
        return None, f"unsupported/pending action: {action}"
    if not isinstance(orders, list) or len(orders) != len(quads):
        return None, "orders count does not equal QR instance count"
    normalized = []
    for index, order in enumerate(orders):
        if not isinstance(order, list) or sorted(order) != [0, 1, 2, 3]:
            return None, f"instance {index} order is not a permutation of 0,1,2,3"
        if quad_geometry_issue(quads[index].points) is not None:
            return None, f"instance {index} needs corrected_label, not reordering"
        normalized.append(tuple(int(x) for x in order))
    out = work / "manual_review_pack" / "approved_labels" / rec.split / "labels" / f"{rec.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(q.reordered_line(order)
                              for q, order in zip(quads, normalized)) + "\n",
                   encoding="utf-8")
    return out, None


def finalize(dataset: Path, v3_work: Path, work: Path, decisions_path: Path,
             output: Path) -> dict:
    cfg = Config(dataset.resolve(), v3_work.resolve(), work.resolve())
    records, v3 = image_records(cfg)
    baseline = json.loads((work / "immutable_before.json").read_text(encoding="utf-8"))
    if baseline != snapshot(immutable_paths(records, v3)):
        raise RuntimeError("immutable inputs changed; finalization blocked")
    failures = {x["image_id"]: x for x in read_jsonl(work / "recovery_failures.jsonl")}
    decisions = {x["image_id"]: x for x in read_jsonl(decisions_path)}
    if len(decisions) != len(read_jsonl(decisions_path)):
        raise ValueError("duplicate image_id in decisions file")
    selected: list[tuple[object, Path, str]] = []
    pending = []
    for rec in records:
        key = (rec.split, rec.stem)
        if key in v3:
            selected.append((rec, v3[key], "V3_gold"))
            continue
        recovered = work / "recovered_labels" / rec.split / "labels" / f"{rec.stem}.txt"
        if recovered.is_file():
            selected.append((rec, recovered, "ZXing_ZA"))
            continue
        decision = decisions.get(rec.image_id)
        failure = failures.get(rec.image_id)
        if decision is None or failure is None:
            pending.append({"image_id": rec.image_id, "reason": "missing decision/report"})
            continue
        image = open_rgb(rec.image_path)
        label, error = decision_label(rec, image, decision, failure, work)
        if error:
            pending.append({"image_id": rec.image_id, "reason": error})
        else:
            selected.append((rec, label, "human_reviewed"))
    write_json(work / "manual_review_pack" / "finalization_pending.json", pending)
    if pending:
        raise RuntimeError(f"finalization blocked: {len(pending)} images remain unresolved")
    if len(selected) != len(records):
        raise RuntimeError("finalization selection count invariant failed")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final dataset: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale building directory exists: {building}")
    source_counts: dict[str, int] = {}
    total_instances = 0
    try:
        building.mkdir(parents=True)
        with (building / "finalization_manifest.jsonl").open("w", encoding="utf-8") as mf:
            for rec, label_src, source in selected:
                image = open_rgb(rec.image_path)
                quads = parse_label(label_src, *image.size)
                issues = [quad_geometry_issue(q.points) for q in quads]
                if any(issue is not None for issue in issues):
                    raise ValueError(f"invalid final quad for {rec.image_id}: {issues}")
                for quad in quads:
                    if (quad.points[:, 0].min() < 0 or
                            quad.points[:, 0].max() > image.width or
                            quad.points[:, 1].min() < 0 or
                            quad.points[:, 1].max() > image.height):
                        raise ValueError(f"out-of-image vertex for {rec.image_id}")
                total_instances += len(quads)
                source_counts[source] = source_counts.get(source, 0) + 1
                image_dst = building / rec.split / "images" / rec.image_path.name
                label_dst = building / rec.split / "labels" / f"{rec.stem}.txt"
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rec.image_path, image_dst)
                shutil.copy2(label_src, label_dst)
                append_jsonl(mf, {"image_id": rec.image_id, "source": source,
                                  "instances": len(quads),
                                  "image_sha256": file_hash(image_dst),
                                  "label_sha256": file_hash(label_dst)})
        report = {"passed": True, "images": len(selected),
                  "instances": total_instances, "sources": source_counts,
                  "original_inputs_unchanged": True}
        write_json(building / "finalization_report.json", report)
        building.replace(output)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    if baseline != snapshot(immutable_paths(records, v3)):
        raise RuntimeError("immutable inputs changed during finalization")
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--dataset", type=Path, required=True)
    pack.add_argument("--work", type=Path, required=True)
    pack.add_argument("--page-size", type=int, default=20)
    finish = sub.add_parser("finalize")
    finish.add_argument("--dataset", type=Path, required=True)
    finish.add_argument("--v3-work", type=Path, required=True)
    finish.add_argument("--work", type=Path, required=True)
    finish.add_argument("--decisions", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "pack":
        result = build_review_pack(args.dataset.resolve(), args.work.resolve(),
                                   args.page_size)
    else:
        result = finalize(args.dataset.resolve(), args.v3_work.resolve(),
                          args.work.resolve(), args.decisions.resolve(),
                          args.output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
