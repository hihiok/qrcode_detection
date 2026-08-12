import hashlib

from PIL import Image
import zxingcpp

from barber_label_repair_zxingcpp_v2.data import ImageRecord, Quad
from barber_label_repair_zxingcpp_v2.detector import ZXingSafeDetector
from barber_label_repair_zxingcpp_v2.pipeline import (
    Config, audit, calibration_views, scan_quad,
)


def test_real_zxing_detection_and_negative_control():
    detector = ZXingSafeDetector()
    detector.negative_control()
    barcode = zxingcpp.create_barcode(
        "v2-integration-control", zxingcpp.BarcodeFormat.QRCode)
    image = Image.fromarray(barcode.to_image(scale=6, add_quiet_zones=True))
    results = detector.read(image, "LocalAverage", False)
    assert len(results) == 1
    assert results[0].valid
    assert results[0].position.shape == (4, 2)
    expected = hashlib.sha256(b"v2-integration-control").hexdigest()
    assert results[0].payload_sha256 == expected


def test_real_detection_survives_four_rotation_inverse_mapping():
    detector = ZXingSafeDetector()
    barcode = zxingcpp.create_barcode(
        "v2-four-rotation-control", zxingcpp.BarcodeFormat.QRCode)
    qr = Image.fromarray(barcode.to_image(scale=6, add_quiet_zones=True)).convert("RGB")
    image = Image.new("RGB", (240, 320), "white")
    image.paste(qr, (30, 60))
    initial = detector.read(image, "LocalAverage", False)
    assert len(initial) == 1
    points = initial[0].position
    tokens = tuple((str(x / image.width), str(y / image.height))
                   for x, y in points)
    quad = Quad("0", tokens, points)
    rows = list(scan_quad(
        detector, image, quad, calibration_views(image, quad),
        ("rgb", "gray"), ("LocalAverage",), True,
        "synthetic", "synthetic#0"))
    accepted = [r for r in rows if r["accepted_geometry"]]
    assert {r["input_rotation"] for r in accepted} == {0, 90, 180, 270}
    assert {tuple(r["field_to_manual"]) for r in accepted} == {(0, 1, 2, 3)}


def test_end_to_end_single_image_audit_reuses_manual_tokens(tmp_path):
    detector = ZXingSafeDetector()
    barcode = zxingcpp.create_barcode(
        "v2-audit-control", zxingcpp.BarcodeFormat.QRCode)
    qr = Image.fromarray(barcode.to_image(scale=6, add_quiet_zones=True)).convert("RGB")
    image = Image.new("RGB", (240, 320), "white")
    image.paste(qr, (30, 60))
    position = detector.read(image, "LocalAverage", False)[0].position
    image_path = tmp_path / "train" / "images" / "fixture.png"
    label_path = tmp_path / "train" / "labels" / "fixture.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image.save(image_path)
    tokens = tuple((f"{x / image.width:.9f}", f"{y / image.height:.9f}")
                   for x, y in position)
    original_line = "0 " + " ".join(v for pair in tokens for v in pair)
    label_path.write_text(original_line + "\n", encoding="utf-8")
    record = ImageRecord("train", "fixture", image_path, label_path)
    work = tmp_path / "work"
    work.mkdir()
    report = audit(
        Config(tmp_path, tmp_path / "v3", work, allow_count_mismatch=True),
        detector, [record], {}, (0, 1, 2, 3), False)
    assert report["ZA_images"] == 1
    recovered = work / "recovered_labels" / "train" / "labels" / "fixture.txt"
    assert recovered.read_text(encoding="utf-8").strip() == original_line
