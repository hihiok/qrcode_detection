#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_padding import compute_center_padding, padded_size


def check_padding(width, height):
    padding = compute_center_padding(width, height)
    padded_width, padded_height = padded_size(width, height, padding)
    left, top, right, bottom = padding
    assert padded_width >= width
    assert padded_height >= height
    assert padded_width * 4 == padded_height * 3
    assert padded_width % 2 == 0
    assert padded_height % 2 == 0
    assert abs(left - right) <= 1
    assert abs(top - bottom) <= 1
    return padding, (padded_width, padded_height)


def test_portrait_padding():
    padding, size = check_padding(720, 1280)
    assert padding == (120, 0, 120, 0)
    assert size == (960, 1280)


def test_landscape_padding_is_exact_and_codec_safe():
    padding, size = check_padding(1280, 720)
    assert padding == (2, 496, 2, 496)
    assert size == (1284, 1712)


def test_existing_three_by_four_is_unchanged():
    padding, size = check_padding(720, 960)
    assert padding == (0, 0, 0, 0)
    assert size == (720, 960)


def test_odd_source_dimensions_are_centered():
    check_padding(721, 961)


def test_invalid_dimensions_fail():
    try:
        compute_center_padding(0, 720)
    except ValueError:
        pass
    else:
        raise AssertionError("zero width must fail")


if __name__ == "__main__":
    test_portrait_padding()
    test_landscape_padding_is_exact_and_codec_safe()
    test_existing_three_by_four_is_unchanged()
    test_odd_source_dimensions_are_centered()
    test_invalid_dimensions_fail()
    print("PASS")
