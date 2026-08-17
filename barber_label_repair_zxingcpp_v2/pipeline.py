from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import (append_jsonl, discover_dataset, discover_v3_labels, open_rgb,
                   parse_label, snapshot, write_json)
from .detector import ZXingSafeDetector, preprocess
from .geometry import (View, apply_homography, best_vertex_assignment, crop_view,
                       quad_geometry_issue, rotate_image_and_h,
                       semantic_to_manual, warp_view)
from .vgg_geometry import (VGGGeometryImage, build_geometry, load_geometry)


CAL_PRE = ("rgb", "gray", "unsharp", "otsu")
AUDIT_TIERS = (
    {"crops": (.20,), "warps": ((512, .10),),
     "pre": ("rgb", "gray", "unsharp"), "bins": ("LocalAverage",)},
    {"crops": (.10, .35, .50),
     "warps": ((256, .10), (384, .10), (768, .10), (1024, .10)),
     "pre": ("gamma_0.7", "gamma_1.4", "otsu", "invert"),
     "bins": ("LocalAverage", "GlobalHistogram")},
    {"crops": (.10, .20, .35, .50),
     "warps": tuple((s, q) for s in (256, 384, 512, 768, 1024)
                    for q in (.05, .10, .15, .20)),
     "pre": ("rgb", "gray", "gamma_0.7", "gamma_1.4", "unsharp",
             "otsu", "adaptive", "invert"),
     "bins": ("LocalAverage", "GlobalHistogram", "FixedThreshold")},
)


@dataclass
class Config:
    dataset: Path
    v3_work: Path
    work: Path
    barber_root: Path = Path(".")
    expect_total: int = 1219
    expect_gold_images: int = 799
    expect_gold_instances: int = 915
    expect_failed: int = 420
    allow_count_mismatch: bool = False


def geometry_views(image, quad, spec, errors: list[dict] | None = None) -> Iterator[View]:
    for margin in spec["crops"]:
        try:
            yield crop_view(image, quad.points, margin)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if errors is not None:
                errors.append({"geometry_id": f"crop_m{margin:.2f}_t512",
                               "error": type(exc).__name__, "message": str(exc)})
    for size, quiet in spec["warps"]:
        try:
            yield warp_view(image, quad.points, size, quiet)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if errors is not None:
                errors.append({"geometry_id": f"warp_s{size}_q{quiet:.2f}",
                               "error": type(exc).__name__, "message": str(exc)})


def image_records(cfg: Config):
    records = discover_dataset(cfg.dataset)
    v3 = discover_v3_labels(cfg.v3_work)
    by_key = {(r.split, r.stem): r for r in records}
    missing = sorted(set(v3) - set(by_key))
    if missing:
        raise RuntimeError(f"{len(missing)} V3 labels have no image; first={missing[0]}")
    gold_instances = 0
    for key, path in v3.items():
        rec = by_key[key]
        im = open_rgb(rec.image_path)
        gold_instances += len(parse_label(path, *im.size))
    counts = (len(records), len(v3), gold_instances, len(records) - len(v3))
    expected = (cfg.expect_total, cfg.expect_gold_images, cfg.expect_gold_instances,
                cfg.expect_failed)
    if counts != expected and not cfg.allow_count_mismatch:
        raise RuntimeError(f"dataset/V3 count mismatch actual={counts} expected={expected}")
    return records, v3


def immutable_paths(records, v3):
    return ([r.image_path for r in records] + [r.label_path for r in records] +
            list(v3.values()))


def _geometry_context(cfg: Config, records, v3, rebuild: bool):
    report_path = cfg.work / "vgg_gold_cross_validation_report.json"
    manifest_path = cfg.work / "vgg_geometry_manifest.jsonl"
    if rebuild or not report_path.is_file() or not manifest_path.is_file():
        return build_geometry(cfg.dataset, cfg.barber_root, cfg.work, records, v3)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError("cached VGG gold cross-validation passed != true")
    geometry_report = json.loads(
        (cfg.work / "vgg_geometry_report.json").read_text(encoding="utf-8"))
    return load_geometry(cfg.work), geometry_report


def doctor(cfg: Config, detector: ZXingSafeDetector, rebuild_geometry: bool = False):
    records, v3 = image_records(cfg)
    detector.negative_control()
    baseline_path = cfg.work / "immutable_before.json"
    current = snapshot(immutable_paths(records, v3))
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline != current:
            changed = sorted(k for k in set(baseline) | set(current)
                             if baseline.get(k) != current.get(k))
            raise RuntimeError(
                f"immutable inputs changed since first doctor run: {changed[:5]}")
    else:
        write_json(baseline_path, current)
    geometry, geometry_report = _geometry_context(
        cfg, records, v3, rebuild_geometry)
    with (cfg.work / "failed_420_manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            if (r.split, r.stem) not in v3:
                item = geometry.get(r.image_id)
                append_jsonl(f, {"image_id": r.image_id, "image_path": str(r.image_path),
                                 "label_path": str(r.label_path),
                                 "geometry_source": "barber_vgg_manual_polygon",
                                 "vgg_geometry_resolved": item is not None,
                                 "valid_vgg_instances": len(item.quads) if item else 0,
                                 "invalid_vgg_instances": (len(item.invalid_objects)
                                                           if item else None)})
    write_json(cfg.work / "doctor_report.json", {
        "passed": True, "dataset_images": len(records), "V3_gold_images": len(v3),
        "failed_images": len(records) - len(v3), "vgg_geometry": geometry_report,
        "old_txt_geometry_used": False})
    return records, v3, geometry


def calibration_views(image, quad):
    return geometry_views(image, quad, {
        "crops": (.20, .35), "warps": ((384, .10), (512, .10))})


def scan_quad(detector, image, quad, views, preprocessors, binarizers,
              allow_errors, image_id, instance_id, field_to_semantic=None):
    seen = set()
    for view in views:
        for rot in range(4):
            rotated, h = rotate_image_and_h(view.image, view.original_to_view, rot)
            inv = np.linalg.inv(h)
            for pre in preprocessors:
                prepared = preprocess(rotated, pre)
                for bin_name in binarizers:
                    # return_errors=True already returns valid decodes as well as error
                    # candidates. Never scan the same view twice merely to separate them.
                    return_error_modes = (True,) if allow_errors else (False,)
                    for return_errors in return_error_modes:
                        for result in detector.read(prepared, bin_name, return_errors):
                            if not result.valid and not allow_errors:
                                continue
                            mapped = apply_homography(inv, result.position)
                            assignment, mean_err, max_err = best_vertex_assignment(
                                mapped, quad.points)
                            accepted = mean_err <= .12 and max_err <= .25
                            quant = tuple(np.round(mapped / 2.0).astype(int).ravel())
                            fingerprint = (instance_id, rot, view.geometry_family, pre,
                                           bin_name, quant, result.payload_sha256,
                                           result.error_type)
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            row = {
                                "image_id": image_id, "instance_id": instance_id,
                                "input_rotation": rot * 90,
                                "geometry_family": view.geometry_family,
                                "geometry_id": view.geometry_id,
                                "preprocessing_family": pre, "binarizer": bin_name,
                                "return_errors": return_errors, "valid": result.valid,
                                "error_type": result.error_type,
                                "zxing_position": result.position.tolist(),
                                "mapped_position": mapped.tolist(),
                                "field_to_manual": list(assignment),
                                "mean_normalized_vertex_error": mean_err,
                                "max_normalized_vertex_error": max_err,
                                "accepted_geometry": accepted,
                                "orientation": result.orientation,
                                "payload_sha256": result.payload_sha256,
                                "mirrored": result.mirrored}
                            if field_to_semantic is not None and accepted:
                                row["semantic_to_manual"] = list(semantic_to_manual(
                                    field_to_semantic, assignment))
                            yield row


def calibrate(cfg: Config, detector: ZXingSafeDetector, records, v3):
    mappings_valid = collections.Counter()
    mappings_error = collections.Counter()
    per_rot = collections.defaultdict(lambda: {"observations": 0, "matched": 0})
    detected_instances, matched_instances = set(), set()
    valid_obs = error_obs = matched_obs = attempted = 0
    instance_rot_maps = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    by_key = {(r.split, r.stem): r for r in records}
    with (cfg.work / "calibration_evidence.jsonl").open("w", encoding="utf-8") as ef:
        for key, gold_path in sorted(v3.items()):
            rec = by_key[key]
            image = open_rgb(rec.image_path)
            for i, quad in enumerate(parse_label(gold_path, *image.size)):
                attempted += 1
                iid = f"{rec.image_id}#{i}"
                rows = scan_quad(detector, image, quad, calibration_views(image, quad),
                                 CAL_PRE, ("LocalAverage", "GlobalHistogram"), True,
                                 rec.image_id, iid)
                for row in rows:
                    detected_instances.add(iid)
                    rot = str(row["input_rotation"])
                    per_rot[rot]["observations"] += 1
                    if row["valid"]: valid_obs += 1
                    else: error_obs += 1
                    if row["accepted_geometry"]:
                        matched_instances.add(iid)
                        matched_obs += 1
                        mapping = tuple(row["field_to_manual"])
                        row["field_to_semantic"] = list(mapping)
                        instance_rot_maps[iid][rot][mapping] += 1
                        (mappings_valid if row["valid"] else mappings_error)[mapping] += 1
                        per_rot[rot]["matched"] += 1
                    append_jsonl(ef, row)
    modal, modal_count = mappings_valid.most_common(1)[0] if mappings_valid else (None, 0)
    valid_total = sum(mappings_valid.values())
    modal_ratio = modal_count / valid_total if valid_total else 0.0
    consistent = 0
    for rot_counts in instance_rot_maps.values():
        if all(str(r) in rot_counts for r in (0, 90, 180, 270)):
            modes = [rot_counts[str(r)].most_common(1)[0][0] for r in (0, 90, 180, 270)]
            if len(set(modes)) == 1 and modes[0] == modal:
                consistent += 1
    error_modal, error_count = (mappings_error.most_common(1)[0]
                                if mappings_error else (None, 0))
    error_total = sum(mappings_error.values())
    error_ratio = error_count / error_total if error_total else 0.0
    allow_error_results = bool(error_total >= 100 and error_modal == modal and
                               error_ratio >= .99)
    reasons = []
    if len(matched_instances) < 50: reasons.append("matched unique instances < 50")
    if matched_obs < 100: reasons.append("matched rotation observations < 100")
    if modal is None or modal_ratio < .99:
        reasons.append("valid-result modal mapping ratio < 0.99")
    if consistent < 50: reasons.append("four-rotation consistent instances < 50")
    report = {
        "total_gold_images": len(v3), "total_gold_instances": cfg.expect_gold_instances,
        "attempted_instances": attempted, "detected_instances": len(detected_instances),
        "matched_instances": len(matched_instances),
        "total_rotation_observations": valid_obs + error_obs,
        "valid_decode_observations": valid_obs,
        "error_result_observations": error_obs,
        "matched_observations": matched_obs,
        "modal_mapping": list(modal) if modal else None,
        "modal_mapping_ratio": modal_ratio,
        "rotation_consistent_instances": consistent,
        "error_modal_mapping": list(error_modal) if error_modal else None,
        "error_modal_mapping_ratio": error_ratio,
        "allow_error_results": allow_error_results,
        "per_rotation_statistics": dict(per_rot),
        "passed": not reasons, "failure_reasons": reasons}
    write_json(cfg.work / "calibration_report.json", report)
    write_json(cfg.work / "calibration_mapping.json", {
        "field_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "field_to_semantic": list(modal) if modal else None,
        "allow_error_results": allow_error_results})
    if reasons:
        raise RuntimeError("calibration failed: " + "; ".join(reasons))
    return tuple(modal), allow_error_results


def grade(rows):
    accepted = [r for r in rows if r.get("accepted_geometry") and
                r.get("semantic_to_manual") is not None]
    votes = collections.Counter(tuple(r["semantic_to_manual"]) for r in accepted)
    if not votes:
        return "ZM", None, {"reason": "no matched evidence", "votes": 0}
    top, n1 = votes.most_common(1)[0]
    second = votes.most_common(2)[1][1] if len(votes) > 1 else 0
    top_rows = [r for r in accepted if tuple(r["semantic_to_manual"]) == top]
    ratio = n1 / len(accepted)
    dominance = float("inf") if second == 0 else n1 / second
    rotations = {r["input_rotation"] for r in top_rows}
    pre = {r["preprocessing_family"] for r in top_rows}
    geo = {r["geometry_family"] for r in top_rows}
    success = sum(bool(r["valid"]) for r in top_rows)
    payloads = {r["payload_sha256"] for r in top_rows
                if r["valid"] and r["payload_sha256"] is not None}
    mirrored = {r["mirrored"] for r in top_rows if r["mirrored"] is not None}
    conflict = len(payloads) > 1 or len(mirrored) > 1
    facts = {"votes": len(accepted), "top_votes": n1, "second_votes": second,
             "top_ratio": ratio, "dominance": dominance,
             "consistent_rotations": sorted(rotations),
             "preprocessing_families": sorted(pre),
             "geometry_families": sorted(geo),
             "successful_decodes": success, "payload_conflict": len(payloads) > 1,
             "mirror_conflict": len(mirrored) > 1}
    if (len(rotations) == 4 and len(top_rows) >= 8 and len(pre) >= 2 and
            len(geo) >= 2 and dominance >= 4 and ratio >= .90 and
            not conflict and success >= 1):
        return "ZA", top, facts
    if (len(rotations) >= 3 and len(top_rows) >= 4 and dominance >= 4 and
            ratio >= .80 and not conflict):
        return "ZB", top, facts
    return "ZM", top, facts


def physical_evidence_keys(rows, ordering):
    if ordering is None:
        return set()
    keys = set()
    for row in rows:
        if (row.get("accepted_geometry") and
                tuple(row.get("semantic_to_manual", ())) == tuple(ordering)):
            quant = tuple(np.round(np.asarray(row["mapped_position"]) / 2.0)
                          .astype(int).ravel())
            keys.add((row["input_rotation"], row["preprocessing_family"],
                      row["binarizer"], quant, row["payload_sha256"],
                      row["error_type"]))
    return keys


def copy_review(cfg, rec, geometry: VGGGeometryImage | None, grade_name, report):
    root = cfg.work / ("review_grade_zb" if grade_name == "ZB" else "review_grade_zm")
    target = root / rec.split
    target.mkdir(parents=True, exist_ok=True)
    if geometry is not None and geometry.quads:
        (target / f"{rec.stem}.txt").write_text(
            "\n".join(quad.reordered_line((0, 1, 2, 3))
                      for quad in geometry.quads) + "\n", encoding="utf-8")
    write_json(target / f"{rec.stem}.json", report)


def reset_audit_outputs(cfg: Config) -> None:
    """Remove only outputs owned by audit so a retry cannot reuse partial results."""
    for name in ("recovered_labels", "review_grade_zb", "review_grade_zm",
                 "combined_proposed"):
        path = cfg.work / name
        if path.exists():
            shutil.rmtree(path)


def audit(cfg, detector, records, v3, geometry_by_id,
          field_to_semantic, allow_errors):
    reset_audit_outputs(cfg)
    failed = [r for r in records if (r.split, r.stem) not in v3]
    predictions, failures, image_reports = [], [], []
    za_i = zb_i = zm_i = za_images = recovered_output_instances = 0
    with (cfg.work / "orientation_evidence.jsonl").open("w", encoding="utf-8") as ef:
        for rec in failed:
            image = open_rgb(rec.image_path)
            instance_reports = []
            geometry = geometry_by_id.get(rec.image_id)
            quads = list(geometry.quads) if geometry is not None else []
            if geometry is None:
                instance_reports.append({
                    "instance_id": f"{rec.image_id}#vgg_geometry_unresolved", "grade": "ZM",
                    "geometry_index": None,
                    "semantic_to_manual": None,
                    "facts": {"reason": "vgg_geometry_unresolved"},
                    "_evidence_keys": set()})
            elif geometry.invalid_objects:
                for invalid in geometry.invalid_objects:
                    instance_reports.append({
                        "instance_id": invalid["instance_id"], "grade": "ZM",
                        "geometry_index": None,
                        "semantic_to_manual": None,
                        "facts": {"reason": "invalid_vgg_polygon",
                                  "detail": invalid["error"],
                                  "raw_vertex_count": invalid["raw_vertex_count"],
                                  "processed_points": invalid.get("processed_points")},
                        "_evidence_keys": set()})
            for i, quad in enumerate(quads):
                iid = geometry.objects[i]["instance_id"]
                rows = []
                issue = quad_geometry_issue(quad.points)
                if issue is not None:
                    g, ordering, facts = "ZM", None, {
                        "reason": "invalid_vgg_quad_after_transform",
                        "geometry_issue": issue, "vgg_points": quad.points.tolist()}
                else:
                    view_errors: list[dict] = []
                    for tier_no, spec in enumerate(AUDIT_TIERS, 1):
                        try:
                            new_rows = list(scan_quad(
                                detector, image, quad,
                                geometry_views(image, quad, spec, view_errors),
                                spec["pre"], spec["bins"], allow_errors,
                                rec.image_id, iid, field_to_semantic))
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            view_errors.append({
                                "geometry_id": f"tier_{tier_no}",
                                "error": type(exc).__name__, "message": str(exc)})
                            new_rows = []
                        for row in new_rows:
                            row["tier"] = tier_no
                            append_jsonl(ef, row)
                        rows.extend(new_rows)
                        g, ordering, facts = grade(rows)
                        if g == "ZA": break
                    g, ordering, facts = grade(rows)
                    if view_errors:
                        facts["skipped_geometry_views"] = view_errors
                instance_reports.append({
                    "instance_id": iid, "grade": g,
                    "geometry_index": i,
                    "semantic_to_manual": list(ordering) if ordering else None,
                    "facts": facts,
                    "_evidence_keys": physical_evidence_keys(rows, ordering)})
            # One physical ZXing observation may support at most one manual
            # polygon. Any cross-instance reuse makes both instances manual-only.
            owners = collections.defaultdict(list)
            for index, inst in enumerate(instance_reports):
                for key in inst["_evidence_keys"]:
                    owners[key].append(index)
            conflicted = {index for indices in owners.values() if len(indices) > 1
                          for index in indices}
            for index, inst in enumerate(instance_reports):
                inst.pop("_evidence_keys", None)
                if index in conflicted:
                    inst["grade"] = "ZM"
                    inst["semantic_to_manual"] = None
                    inst["facts"]["multi_instance_evidence_conflict"] = True
                if inst["grade"] == "ZA": za_i += 1
                elif inst["grade"] == "ZB": zb_i += 1
                else: zm_i += 1
            image_grade = "ZA" if all(x["grade"] == "ZA" for x in instance_reports) else (
                "ZB" if all(x["grade"] in ("ZA", "ZB") for x in instance_reports)
                else "ZM")
            report = {"image_id": rec.image_id, "grade": image_grade,
                      "geometry_source": "barber_vgg_manual_polygon",
                      "source_rel": geometry.source_rel if geometry else None,
                      "expected_instance_count": len(instance_reports),
                      "instances": instance_reports}
            image_reports.append(report)
            if image_grade == "ZA":
                za_images += 1
                recovered_output_instances += len(quads)
                out = cfg.work / "recovered_labels" / rec.split / "labels" / f"{rec.stem}.txt"
                out.parent.mkdir(parents=True, exist_ok=True)
                ordered_reports = sorted(
                    (x for x in instance_reports if x["geometry_index"] is not None),
                    key=lambda x: x["geometry_index"])
                lines = [q.reordered_line(tuple(x["semantic_to_manual"]))
                         for q, x in zip(quads, ordered_reports)]
                out.write_text("\n".join(lines) + "\n", encoding="utf-8")
                predictions.append(report)
            else:
                copy_review(cfg, rec, geometry, image_grade, report)
                failures.append(report)
    for name, values in (("recovered_predictions.jsonl", predictions),
                         ("recovery_failures.jsonl", failures)):
        with (cfg.work / name).open("w", encoding="utf-8") as f:
            for value in values: append_jsonl(f, value)
    combined = cfg.work / "combined_proposed"
    if combined.exists(): shutil.rmtree(combined)
    for (split, stem), src in v3.items():
        dst = combined / split / "labels" / f"{stem}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rec in failed:
        src = cfg.work / "recovered_labels" / rec.split / "labels" / f"{rec.stem}.txt"
        if src.is_file():
            dst = combined / rec.split / "labels" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    geometry_failure_path = cfg.work / "vgg_geometry_failures.jsonl"
    geometry_failures = []
    if geometry_failure_path.is_file():
        for line in geometry_failure_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                geometry_failures.append(json.loads(line))
    write_json(cfg.work / "transform_mismatch_report.json", {
        "count": sum("transform mismatch" in row.get("error", "")
                     for row in geometry_failures),
        "records": [row for row in geometry_failures
                    if "transform mismatch" in row.get("error", "")]})
    parse_path = cfg.work / "vgg_parse_errors.jsonl"
    parse_rows = []
    if parse_path.is_file():
        for line in parse_path.read_text(encoding="utf-8").splitlines():
            try: parse_rows.append(json.loads(line))
            except json.JSONDecodeError: parse_rows.append({"raw_parse_error_record": line})
    write_json(cfg.work / "parse_error_report.json", {
        "count": len(parse_rows), "status": "manual_review", "records": parse_rows})
    multi = [r for r in image_reports if r["expected_instance_count"] > 1]
    write_json(cfg.work / "multi_qr_completion_report.json", {
        "multi_qr_failed_images": len(multi),
        "fully_recovered": sum(r["grade"] == "ZA" for r in multi),
        "images": multi})
    report = {"input_failed_images": len(failed),
              "input_failed_instances": za_i + zb_i + zm_i,
              "ZA_images": za_images,
              "ZA_instances": za_i, "ZB_instances": zb_i, "ZM_instances": zm_i,
              "ZB_images": sum(r["grade"] == "ZB" for r in image_reports),
              "ZM_images": sum(r["grade"] == "ZM" for r in image_reports),
              "combined_proposed_images": len(v3) + za_images,
              "combined_proposed_instances": (cfg.expect_gold_instances +
                                                recovered_output_instances),
              "geometry_source": "barber_vgg_manual_polygon",
              "old_txt_geometry_used": False,
              "unresolved_geometry_images": sum(
                  geometry_by_id.get(r.image_id) is None for r in failed),
              "invalid_vgg_instances": sum(
                  len(geometry_by_id[r.image_id].invalid_objects)
                  for r in failed if r.image_id in geometry_by_id),
              "auditable_vgg_instances": sum(
                  len(geometry_by_id[r.image_id].quads)
                  for r in failed if r.image_id in geometry_by_id)}
    write_json(cfg.work / "recovery_report.json", report)
    return report


def validate(cfg, records, v3, geometry_by_id, recovery):
    before = json.loads((cfg.work / "immutable_before.json").read_text(encoding="utf-8"))
    after = snapshot(immutable_paths(records, v3))
    if before != after:
        changed = sorted(k for k in set(before) | set(after)
                         if before.get(k) != after.get(k))
        raise RuntimeError(f"immutable inputs changed: {changed[:5]}")
    combined = list((cfg.work / "combined_proposed").rglob("*.txt"))
    if len(combined) != recovery["combined_proposed_images"]:
        raise RuntimeError("combined_proposed count invariant failed")
    by_id = {r.image_id: r for r in records}
    pred_path = cfg.work / "recovered_predictions.jsonl"
    for pred in pred_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(pred)
        rec = by_id[row["image_id"]]
        image = open_rgb(rec.image_path)
        geometry = geometry_by_id.get(rec.image_id)
        if geometry is None:
            raise RuntimeError(f"recovered image lacks VGG geometry: {rec.image_id}")
        original = list(geometry.quads)
        recovered = parse_label(
            cfg.work / "recovered_labels" / rec.split / "labels" / f"{rec.stem}.txt",
            *image.size)
        if len(original) != len(recovered):
            raise RuntimeError(f"instance count changed: {rec.image_id}")
        for oq, rq in zip(original, recovered):
            if sorted(oq.token_points) != sorted(rq.token_points):
                raise RuntimeError(f"non-manual vertex in output: {rec.image_id}")
    write_json(cfg.work / "validation_report.json", {
        "passed": True, "original_images_labels_v3_unchanged": True,
        "polygon_vertex_token_equality": True, "combined_count_valid": True,
        "geometry_source": "barber_vgg_manual_polygon",
        "old_txt_geometry_used": False,
        "vgg_gold_cross_validation_passed": True})


def reuse_calibration(source_work: Path, destination_work: Path):
    report = json.loads((source_work / "calibration_report.json").read_text())
    mapping = json.loads((source_work / "calibration_mapping.json").read_text())
    reasons = []
    if not report.get("passed"): reasons.append("source calibration passed != true")
    if report.get("matched_instances", 0) < 50: reasons.append("matched_instances < 50")
    if report.get("matched_observations", 0) < 100: reasons.append("matched_observations < 100")
    if report.get("modal_mapping_ratio", 0) < .99: reasons.append("modal_mapping_ratio < .99")
    if report.get("rotation_consistent_instances", 0) < 50:
        reasons.append("rotation_consistent_instances < 50")
    order = mapping.get("field_to_semantic")
    if not isinstance(order, list) or sorted(order) != [0, 1, 2, 3]:
        reasons.append("field_to_semantic is not a permutation")
    if reasons:
        raise RuntimeError("cannot reuse calibration: " + "; ".join(reasons))
    shutil.copy2(source_work / "calibration_report.json",
                 destination_work / "calibration_report.json")
    shutil.copy2(source_work / "calibration_mapping.json",
                 destination_work / "calibration_mapping.json")
    write_json(destination_work / "calibration_reuse_report.json", {
        "passed": True, "source_work": str(source_work),
        "matched_instances": report["matched_instances"],
        "matched_observations": report["matched_observations"],
        "modal_mapping_ratio": report["modal_mapping_ratio"],
        "rotation_consistent_instances": report["rotation_consistent_instances"]})


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("doctor", "calibrate", "audit", "all"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--v3-work", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--barber-root", type=Path, required=True)
    p.add_argument("--reuse-calibration-work", type=Path)
    p.add_argument("--allow-count-mismatch", action="store_true")
    return p


def main(argv=None):
    a = make_parser().parse_args(argv)
    cfg = Config(a.dataset.resolve(), a.v3_work.resolve(), a.work.resolve(),
                 a.barber_root.resolve(),
                 allow_count_mismatch=a.allow_count_mismatch)
    cfg.work.mkdir(parents=True, exist_ok=True)
    detector = ZXingSafeDetector()
    records, v3, geometry = doctor(
        cfg, detector, rebuild_geometry=a.command in ("doctor", "all"))
    if a.command == "doctor":
        print("DOCTOR PASSED")
        return 0
    if a.command in ("calibrate", "all"):
        mapping, allow_errors = calibrate(cfg, detector, records, v3)
        if a.command == "calibrate":
            print("CALIBRATION PASSED")
            return 0
    else:
        if (not (cfg.work / "calibration_report.json").is_file() and
                a.reuse_calibration_work is not None):
            reuse_calibration(a.reuse_calibration_work.resolve(), cfg.work)
        report = json.loads((cfg.work / "calibration_report.json").read_text())
        mapping_obj = json.loads((cfg.work / "calibration_mapping.json").read_text())
        if not report.get("passed"):
            raise RuntimeError("audit blocked: calibration passed != true")
        mapping = tuple(mapping_obj["field_to_semantic"])
        allow_errors = bool(mapping_obj["allow_error_results"])
    recovery = audit(cfg, detector, records, v3, geometry, mapping, allow_errors)
    validate(cfg, records, v3, geometry, recovery)
    print(json.dumps(recovery, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
