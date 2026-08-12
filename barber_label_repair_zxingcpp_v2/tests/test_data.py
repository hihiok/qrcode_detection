from pathlib import Path

from barber_label_repair_zxingcpp_v2.data import parse_label


def test_label_reorder_reuses_exact_coordinate_tokens(tmp_path: Path):
    label = tmp_path / "x.txt"
    label.write_text(
        "0 0.100000 0.200000 0.300000 0.400000 "
        "0.500000 0.600000 0.700000 0.800000\n",
        encoding="utf-8")
    quad = parse_label(label, 240, 320)[0]
    assert quad.reordered_line((2, 3, 0, 1)) == (
        "0 0.500000 0.600000 0.700000 0.800000 "
        "0.100000 0.200000 0.300000 0.400000")
