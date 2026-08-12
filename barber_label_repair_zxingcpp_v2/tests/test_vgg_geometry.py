import json
from pathlib import Path

import numpy as np
from PIL import Image

from barber_label_repair_zxingcpp_v2.data import ImageRecord
from barber_label_repair_zxingcpp_v2.vgg_geometry import (
    _manifest_row, build_record_indices, cross_validate_gold,
    load_barber_annotations, render_to_processed, resolve_record,
    transform_points_to_processed,
)


def make_fixture(tmp_path: Path):
    barber = tmp_path / "BarBeR"
    source_path = barber / "set" / "source.jpg"
    source_path.parent.mkdir(parents=True)
    source = Image.new("RGB", (400, 200), (80, 100, 120))
    source.save(source_path, quality=100)
    points = [[50, 25], [350, 25], [350, 175], [50, 175], [50, 25]]
    vgg = {"one": {"filename": "source.jpg", "regions": [{
        "shape_attributes": {"name": "polygon", "all_points_x": [p[0] for p in points],
                             "all_points_y": [p[1] for p in points]},
        "region_attributes": {"type": "QR", "encoded string": "must-not-leak"},
    }]}}
    (source_path.parent / "annotations.json").write_text(json.dumps(vgg), encoding="utf-8")

    dataset = tmp_path / "dataset"
    image_path = dataset / "train" / "images" / "barber_fixture.png"
    label_path = dataset / "train" / "labels" / "barber_fixture.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    rendered, transform = render_to_processed(source, 240, 320, 127)
    rendered.save(image_path)
    (dataset / "train" / "annotations.jsonl").write_text(json.dumps({
        "image": image_path.name, "source_image": "set/source.jpg",
    }) + "\n", encoding="utf-8")
    return barber, dataset, source_path, image_path, label_path, np.asarray(points[:-1]), transform


def test_vgg_geometry_rebuild_and_gold_cross_validation(tmp_path):
    barber, dataset, _, image_path, label_path, source_points, expected_transform = make_fixture(tmp_path)
    annotations, errors = load_barber_annotations(barber)
    assert len(annotations) == 1
    assert annotations[0]["invalid_objects"] == []
    assert "must-not-leak" not in repr(annotations)
    assert errors == []

    score, runner_up, record, transform = resolve_record(
        image_path, {"source_image": "set/source.jpg"},
        build_record_indices(annotations), 240, 320, 127, .90)
    assert score > .99
    assert runner_up is None
    assert transform == expected_transform
    transformed = transform_points_to_processed(source_points, transform)
    assert np.allclose(transformed[0], [179.2, 40.0])

    rec = ImageRecord("train", image_path.stem, image_path, label_path)
    row = _manifest_row(rec, score, runner_up, record, transform)
    # Gold may have a semantic start/order different from raw VGG order.
    ordered = transformed[[1, 2, 3, 0]]
    label_path.write_text("0 " + " ".join(
        value for x, y in ordered
        for value in (f"{x / 240:.8f}", f"{y / 320:.8f}")) + "\n", encoding="utf-8")
    report = cross_validate_gold(
        [row], {("train", image_path.stem): label_path}, [rec],
        expected_gold_images=1, expected_gold_instances=1)
    assert report["passed"]
    assert report["maximum_vertex_error_px"] < 1e-4
