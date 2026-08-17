import json

from PIL import Image
import pytest

from barber_label_repair_zxingcpp_v2.accepted_export import export_accepted
from barber_label_repair_zxingcpp_v2.data import file_hash, write_json


def _make_image(dataset, stem, color):
    image = dataset / "train" / "images" / f"{stem}.png"
    label = dataset / "train" / "labels" / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (240, 320), color).save(image)
    # Deliberately use a different order from the accepted label.  The export
    # must never copy this legacy dataset TXT into canonical annotations.
    label.write_text("0 .1 .1 .1 .8 .8 .8 .8 .1\n", encoding="utf-8")
    return image, label


def _fixture(tmp_path):
    dataset = tmp_path / "dataset"
    a_image, a_old = _make_image(dataset, "a", "black")
    b_image, b_old = _make_image(dataset, "b", "gray")
    _make_image(dataset, "drop", "white")
    work = tmp_path / "work"
    accepted_a = work / "combined_proposed" / "train" / "labels" / "a.txt"
    accepted_b = work / "combined_proposed" / "train" / "labels" / "b.txt"
    accepted_a.parent.mkdir(parents=True)
    accepted_a.write_text("0 .8 .1 .8 .8 .1 .8 .1 .1\n", encoding="utf-8")
    accepted_b.write_text("0 .2 .2 .7 .2 .7 .7 .2 .7\n", encoding="utf-8")
    recovered = work / "recovered_labels" / "train" / "labels" / "b.txt"
    recovered.parent.mkdir(parents=True)
    recovered.write_text(accepted_b.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(work / "recovery_report.json", {
        "combined_proposed_images": 2, "combined_proposed_instances": 2,
        # One additional ZA instance shares the dropped ZB image.  This mirrors
        # the real audit: instance-grade ZA is not the same as accepted ZA.
        "input_failed_instances": 3,
        "ZA_images": 1, "ZA_instances": 2,
        "ZB_images": 1, "ZB_instances": 1,
        "ZM_images": 0, "ZM_instances": 0,
        "old_txt_geometry_used": False,
    })
    write_json(work / "validation_report.json", {
        "passed": True, "old_txt_geometry_used": False,
        "vgg_gold_cross_validation_passed": True,
    })
    (work / "recovery_failures.jsonl").write_text(
        json.dumps({"image_id": "train/drop", "grade": "ZB"}) + "\n",
        encoding="utf-8")
    return dataset, work, a_image, a_old, b_image, b_old, accepted_a


def test_export_accepted_is_self_contained_and_previews_every_image(tmp_path):
    dataset, work, a_image, a_old, b_image, b_old, accepted_a = _fixture(tmp_path)
    immutable_before = {path: file_hash(path) for path in
                        (a_image, a_old, b_image, b_old, accepted_a)}
    output = tmp_path / "accepted"
    report = export_accepted(
        dataset, work, output, expected_images=2, expected_instances=2,
        expected_dropped_images=1, expected_v3_images=1,
        expected_za_images=1, expected_gold_instances=1,
        expected_accepted_za_instances=1, expected_dropped_instances=2,
        expected_dropped_non_za_instances=1,
        expected_dropped_embedded_za_instances=1, page_size=2)

    assert report["passed"] is True
    assert report["images"] == 2
    assert report["instances"] == 2
    assert report["sources"] == {"V3_gold": 1, "ZXing_ZA": 1}
    assert report["source_instances"] == {"V3_gold": 1, "ZXing_ZA": 1}
    assert report["dropped_instances"] == 2
    assert report["dropped_non_za_instances"] == 1
    assert report["dropped_embedded_za_instances"] == 1
    assert report["old_dataset_txt_used_as_geometry"] is False
    assert len(list((output / "train" / "images").iterdir())) == 2
    assert len(list((output / "train" / "labels").glob("*.txt"))) == 2
    assert not (output / "train" / "images" / "drop.png").exists()
    rows = [json.loads(line) for line in
            (output / "train" / "annotations.jsonl").read_text().splitlines()]
    assert [row["image_id"] for row in rows] == ["train/a", "train/b"]
    # P0 is taken from combined_proposed (.8,.1), not legacy TXT (.1,.1).
    assert rows[0]["instances"][0]["corners"][0] == [192.0, 32.0]
    assert len(list((output / "preview" / "per_image").rglob("*.jpg"))) == 2
    assert len(list((output / "preview" / "pages").glob("*.jpg"))) == 1
    preview = json.loads((output / "preview" / "preview_report.json").read_text())
    assert preview["images"] == 2
    assert preview["per_image_previews"] == 2
    assert preview["pages"] == 1
    assert {path: file_hash(path) for path in immutable_before} == immutable_before


def test_export_accepted_refuses_to_overwrite(tmp_path):
    dataset, work, *_ = _fixture(tmp_path)
    output = tmp_path / "accepted"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_accepted(
            dataset, work, output, expected_images=2, expected_instances=2,
            expected_dropped_images=1, expected_v3_images=1,
            expected_za_images=1, expected_gold_instances=1,
            expected_accepted_za_instances=1, expected_dropped_instances=2,
            expected_dropped_non_za_instances=1,
            expected_dropped_embedded_za_instances=1)


def test_export_accepted_drops_whole_out_of_bounds_image_without_clipping(tmp_path):
    dataset, work, *_ = _fixture(tmp_path)
    _make_image(dataset, "edge", "blue")
    edge = work / "combined_proposed" / "train" / "labels" / "edge.txt"
    edge.write_text("0 .8 .1 1.01 .1 1.01 .8 .8 .8\n", encoding="utf-8")
    write_json(work / "recovery_report.json", {
        "combined_proposed_images": 3, "combined_proposed_instances": 3,
        "input_failed_instances": 3,
        "ZA_images": 1, "ZA_instances": 2,
        "ZB_images": 1, "ZB_instances": 1,
        "ZM_images": 0, "ZM_instances": 0,
        "old_txt_geometry_used": False,
    })
    output = tmp_path / "accepted"
    kwargs = dict(
        expected_images=3, expected_instances=3, expected_dropped_images=1,
        expected_v3_images=2, expected_za_images=1,
        expected_gold_instances=2, expected_accepted_za_instances=1,
        expected_dropped_instances=2, expected_dropped_non_za_instances=1,
        expected_dropped_embedded_za_instances=1, page_size=2)
    with pytest.raises(RuntimeError, match="rerun only with explicit"):
        export_accepted(dataset, work, output, **kwargs)
    assert not output.exists()

    report = export_accepted(
        dataset, work, output, drop_out_of_bounds=True, **kwargs)
    assert report["candidate_images"] == 3
    assert report["candidate_instances"] == 3
    assert report["candidate_sources"] == {"V3_gold": 2, "ZXing_ZA": 1}
    assert report["candidate_source_instances"] == {"V3_gold": 2, "ZXing_ZA": 1}
    assert report["images"] == 2
    assert report["instances"] == 2
    assert report["additional_boundary_dropped_images"] == 1
    assert report["additional_boundary_dropped_instances"] == 1
    assert report["dropped_images"] == 2
    assert report["dropped_instances"] == 3
    assert not (output / "train" / "images" / "edge.png").exists()
    rejected = json.loads((output / "boundary_rejections.jsonl").read_text())
    assert rejected["image_id"] == "train/edge"
    assert rejected["outside_points"][0]["x"] == pytest.approx(242.4)
    assert rejected["policy"] == "drop_entire_image_no_coordinate_clipping"
