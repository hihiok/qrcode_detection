from pathlib import Path

import numpy as np
from PIL import Image

from barber_label_repair_zxingcpp_v2.data import ImageRecord, Quad
from barber_label_repair_zxingcpp_v2.gold_review import build_gold_geometry_pack
from barber_label_repair_zxingcpp_v2.vgg_geometry import VGGGeometryImage


def _quad(points):
    array = np.asarray(points, dtype=np.float64)
    tokens = tuple((f"{x / 240:.8f}", f"{y / 320:.8f}") for x, y in array)
    return Quad("0", tokens, array)


def test_gold_pack_uses_v3_proposed_and_nested_geometry(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    image_path = dataset / "train" / "images" / "sample.jpg"
    old_label = dataset / "train" / "labels" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    old_label.parent.mkdir(parents=True)
    Image.new("RGB", (240, 320), "gray").save(image_path)

    # The old dataset TXT deliberately has two wrong instances.  It must never
    # be used as gold by this review path.
    old_label.write_text(
        "0 .01 .01 .02 .01 .02 .02 .01 .02\n" * 2, encoding="utf-8")
    v3_label = tmp_path / "v3" / "proposed" / "train" / "labels" / "sample.txt"
    v3_label.parent.mkdir(parents=True)
    points = [[30, 60], [150, 60], [150, 180], [30, 180]]
    v3_label.write_text("0 " + " ".join(
        value for x, y in points
        for value in (f"{x / 240:.8f}", f"{y / 320:.8f}")) + "\n",
        encoding="utf-8")
    geometry = VGGGeometryImage(
        image_id="train/sample", source_rel="set/source.jpg",
        source_image=Path("source.jpg"), source_output_ssim=1.0,
        runner_up_ssim=None, transform={}, quads=(_quad(points),),
        objects=({},), invalid_objects=())
    monkeypatch.setattr(
        "barber_label_repair_zxingcpp_v2.gold_review.load_geometry",
        lambda _: {"train/sample": geometry})

    output = tmp_path / "review"
    report = build_gold_geometry_pack(
        dataset, tmp_path / "v3", tmp_path / "work", output,
        ["train/sample"])
    assert report["passed"] is True
    assert report["old_dataset_txt_used_as_gold"] is False
    assert report["records"][0]["gold_instances"] == 1
    assert report["records"][0]["geometry_instances"] == 1
    assert report["records"][0]["maximum_vertex_set_error_px"] < 1e-4
    assert (output / "per_image" / "train__sample.jpg").is_file()


def test_gold_pack_matches_multiple_qr_instances_one_to_one(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    image_path = dataset / "val" / "images" / "multi.jpg"
    old_label = dataset / "val" / "labels" / "multi.txt"
    image_path.parent.mkdir(parents=True)
    old_label.parent.mkdir(parents=True)
    Image.new("RGB", (240, 320), "gray").save(image_path)
    old_label.write_text(
        "0 .01 .01 .02 .01 .02 .02 .01 .02\n", encoding="utf-8")

    left = [[20, 50], [90, 50], [90, 120], [20, 120]]
    right = [[140, 180], [220, 180], [220, 270], [140, 270]]
    v3_label = tmp_path / "v3" / "proposed" / "val" / "labels" / "multi.txt"
    v3_label.parent.mkdir(parents=True)
    lines = []
    for points in (left, right):
        lines.append("0 " + " ".join(
            value for x, y in points
            for value in (f"{x / 240:.8f}", f"{y / 320:.8f}")))
    v3_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Geometry order is intentionally opposite to gold order.
    geometry = VGGGeometryImage(
        image_id="val/multi", source_rel="set/source.jpg",
        source_image=Path("source.jpg"), source_output_ssim=1.0,
        runner_up_ssim=None, transform={},
        quads=(_quad(right), _quad(left)), objects=({}, {}), invalid_objects=())
    monkeypatch.setattr(
        "barber_label_repair_zxingcpp_v2.gold_review.load_geometry",
        lambda _: {"val/multi": geometry})

    report = build_gold_geometry_pack(
        dataset, tmp_path / "v3", tmp_path / "work", tmp_path / "review",
        ["val/multi"])
    row = report["records"][0]
    assert report["passed"] is True
    assert row["gold_instances"] == 2
    assert row["geometry_instances"] == 2
    assert row["assignment_gold_to_geometry"] == [1, 0]
    assert row["maximum_vertex_set_error_px"] < 1e-4
