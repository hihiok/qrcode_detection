import json

from PIL import Image

from barber_label_repair_zxingcpp_v2.data import ImageRecord, snapshot, write_json
from barber_label_repair_zxingcpp_v2 import review
from barber_label_repair_zxingcpp_v2.review import build_review_pack, finalize


def test_review_pack_renders_and_writes_pending_template(tmp_path):
    image_path = tmp_path / "dataset" / "train" / "images" / "x.png"
    label_path = tmp_path / "dataset" / "train" / "labels" / "x.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (240, 320), "black").save(image_path)
    label_path.write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    failure = {
        "image_id": "train/x", "grade": "ZB",
        "instances": [{"instance_id": "train/x#0", "grade": "ZB",
                       "semantic_to_manual": [1, 2, 3, 0], "facts": {}}],
    }
    (work / "recovery_failures.jsonl").write_text(
        json.dumps(failure) + "\n", encoding="utf-8")
    report = build_review_pack(tmp_path / "dataset", work, page_size=20)
    assert report["review_images"] == 1
    assert report["ZB_images"] == 1
    assert (work / "manual_review_pack" / "previews" / "zb_page_001.jpg").is_file()
    decision = json.loads((work / "manual_review_pack" /
                           "manual_review_decisions_TEMPLATE.jsonl").read_text())
    assert decision["action"] == "pending"
    assert decision["orders"] == [[1, 2, 3, 0]]


def test_finalize_requires_human_decision_and_creates_new_dataset(tmp_path,
                                                                  monkeypatch):
    dataset = tmp_path / "dataset"
    image_path = dataset / "train" / "images" / "x.png"
    label_path = dataset / "train" / "labels" / "x.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (240, 320), "black").save(image_path)
    line = "0 .1 .1 .8 .1 .8 .8 .1 .8"
    label_path.write_text(line + "\n", encoding="utf-8")
    record = ImageRecord("train", "x", image_path, label_path)
    monkeypatch.setattr(review, "image_records", lambda cfg: ([record], {}))
    work = tmp_path / "work"
    work.mkdir()
    write_json(work / "immutable_before.json", snapshot([image_path, label_path]))
    failure = {"image_id": "train/x", "grade": "ZB",
               "instances": [{"semantic_to_manual": [0, 1, 2, 3]}]}
    (work / "recovery_failures.jsonl").write_text(
        json.dumps(failure) + "\n", encoding="utf-8")
    decisions = work / "decisions.jsonl"
    decisions.write_text(json.dumps({
        "image_id": "train/x", "reviewer": "human",
        "action": "manual_order", "orders": [[0, 1, 2, 3]],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "complete"
    report = finalize(dataset, tmp_path / "v3", work, decisions, output)
    assert report["images"] == 1
    assert report["sources"] == {"human_reviewed": 1}
    assert (output / "train" / "labels" / "x.txt").read_text().strip() == line
    assert len((output / "finalization_manifest.jsonl").read_text().splitlines()) == 1
