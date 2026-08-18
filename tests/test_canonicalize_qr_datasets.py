#!/usr/bin/env python3
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canonicalize_qr_datasets import convert_dataset, sha256_file
from qr_schema import CORNER_ORDER, read_jsonl


def test_conversion_is_non_destructive_and_checks_label_files():
    temporary = tempfile.mkdtemp(prefix="qr-canonical-test-")
    try:
        source = os.path.join(temporary, "source")
        output = os.path.join(temporary, "output")
        os.makedirs(output)
        quad = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
        source_hashes = {}
        for split in ("train", "val", "test"):
            split_root = os.path.join(source, split)
            os.makedirs(os.path.join(split_root, "images"))
            os.makedirs(os.path.join(split_root, "labels"))
            image_name = "%s.jpg" % split
            label_name = "%s.txt" % split
            with open(os.path.join(split_root, "images", image_name), "wb") as handle:
                handle.write(("fake-%s" % split).encode("ascii"))
            coords = []
            for x, y in quad:
                coords.extend([x / 240.0, y / 320.0])
            with open(os.path.join(split_root, "labels", label_name), "w") as handle:
                handle.write("0 " + " ".join("%.8f" % value for value in coords) + "\n")
            row = {
                "image": "images/%s" % image_name,
                "label_file": "labels/%s" % label_name,
                "width": 240, "height": 320, "num_qrcodes": 1,
                "objects": [{"label": "qrcode", "corners": quad,
                             "corner_order": CORNER_ORDER}],
                "group_key": "group-%s" % split,
            }
            annotation_path = os.path.join(split_root, "annotations.jsonl")
            with open(annotation_path, "w") as handle:
                handle.write(json.dumps(row) + "\n")
            source_hashes[split] = sha256_file(annotation_path)
        report = convert_dataset("sample", source, output, 5e-4, True, {}, [])
        assert report["splits"]["train"]["instances"] == 1
        assert report["splits"]["train"]["label_files_checked"] == 1
        for split in ("train", "val", "test"):
            annotation_path = os.path.join(source, split, "annotations.jsonl")
            assert sha256_file(annotation_path) == source_hashes[split]
            result = read_jsonl(os.path.join(
                output, "sample", split, "annotations.jsonl"))[0]
            assert result["schema_version"] == "qr_ordered_corners_v1"
            assert result["instances"][0]["corners"] == quad
            assert os.path.islink(os.path.join(output, "sample", split, "images"))
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    test_conversion_is_non_destructive_and_checks_label_files()
    print("PASS")
