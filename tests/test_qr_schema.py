#!/usr/bin/env python3
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_schema import (CORNER_ORDER, SCHEMA_VERSION, canonicalize_row,
                       validate_canonical_row)


QUAD = [[10, 10], [30, 10], [30, 30], [10, 30]]


def assert_raises(function, expected_text):
    try:
        function()
    except ValueError as exc:
        assert expected_text in str(exc), str(exc)
    else:
        raise AssertionError("Expected ValueError containing %s" % expected_text)


def test_all_three_source_schemas_become_identical():
    common = {"image": "images/a.jpg", "width": 240, "height": 320}
    barber = dict(common, instances=[{"class_id": 0, "corners": QUAD}])
    public = dict(common, num_qrcodes=1, objects=[{
        "label": "qrcode", "corners": QUAD, "corner_order": CORNER_ORDER}])
    synth = dict(common, label="qrcode", corners=QUAD, corner_order=CORNER_ORDER)
    converted = [canonicalize_row(row, "source", "train", index + 1)
                 for index, row in enumerate((barber, public, synth))]
    for row in converted:
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["num_qrcodes"] == 1
        assert row["instances"][0]["corners"] == [[float(v) for v in point]
                                                    for point in QUAD]
        assert row["instances"][0]["corner_order"] == CORNER_ORDER


def test_objects_preserve_multiple_qrs():
    row = {"image": "images/multi.jpg", "width": 240, "height": 320,
           "num_qrcodes": 2, "objects": [
               {"label": "qrcode", "corners": QUAD, "corner_order": CORNER_ORDER},
               {"label": "qrcode",
                "corners": [[50, 50], [70, 50], [70, 70], [50, 70]],
                "corner_order": CORNER_ORDER}]}
    converted = canonicalize_row(row, "boofcv", "train", 1)
    assert len(converted["instances"]) == 2


def test_explicit_negative_is_valid():
    source = {"image": "images/negative.jpg", "width": 240, "height": 320,
              "instances": []}
    row = canonicalize_row(source, "negative", "train", 1)
    assert row["num_qrcodes"] == 0
    validate_canonical_row(row)


def test_unknown_or_ambiguous_schema_fails_closed():
    base = {"image": "images/a.jpg", "width": 240, "height": 320}
    assert_raises(lambda: canonicalize_row(base, "x", "train", 1),
                  "exactly one")
    ambiguous = dict(base, corners=QUAD, instances=[])
    assert_raises(lambda: canonicalize_row(ambiguous, "x", "train", 1),
                  "exactly one")


def test_declared_count_and_order_are_strict():
    row = {"image": "images/a.jpg", "width": 240, "height": 320,
           "num_qrcodes": 2, "objects": [
               {"label": "qrcode", "corners": QUAD, "corner_order": CORNER_ORDER}]}
    assert_raises(lambda: canonicalize_row(row, "x", "train", 1),
                  "num_qrcodes")
    wrong = copy.deepcopy(row)
    wrong["num_qrcodes"] = 1
    wrong["objects"][0]["corner_order"] = list(reversed(CORNER_ORDER))
    assert_raises(lambda: canonicalize_row(wrong, "x", "train", 1),
                  "corner_order")


def test_canonical_requires_schema_and_explicit_count():
    row = canonicalize_row(
        {"image": "images/a.jpg", "width": 240, "height": 320,
         "instances": [{"corners": QUAD}]}, "barber", "train", 1)
    invalid = copy.deepcopy(row)
    del invalid["num_qrcodes"]
    assert_raises(lambda: validate_canonical_row(invalid), "num_qrcodes")
    invalid = copy.deepcopy(row)
    invalid["schema_version"] = "legacy"
    assert_raises(lambda: validate_canonical_row(invalid), "schema_version")


if __name__ == "__main__":
    test_all_three_source_schemas_become_identical()
    test_objects_preserve_multiple_qrs()
    test_explicit_negative_is_valid()
    test_unknown_or_ambiguous_schema_fails_closed()
    test_declared_count_and_order_are_strict()
    test_canonical_requires_schema_and_explicit_count()
    print("PASS")
