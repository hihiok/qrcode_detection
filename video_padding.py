#!/usr/bin/env python3
"""Dependency-free helpers for exact, centered video aspect-ratio padding."""
from __future__ import print_function


def compute_center_padding(width, height, aspect_width=3, aspect_height=4,
                           require_even_output=True):
    """Return left, top, right, bottom padding without resizing the image.

    The padded dimensions are an exact multiple of ``aspect_width:aspect_height``.
    Even output dimensions are used by default for broad MP4 codec compatibility.
    """
    width = int(width)
    height = int(height)
    aspect_width = int(aspect_width)
    aspect_height = int(aspect_height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("aspect dimensions must be positive")

    units_w = (width + aspect_width - 1) // aspect_width
    units_h = (height + aspect_height - 1) // aspect_height
    units = max(units_w, units_h)
    if require_even_output and units % 2:
        units += 1

    padded_width = aspect_width * units
    padded_height = aspect_height * units
    horizontal = padded_width - width
    vertical = padded_height - height
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    return left, top, right, bottom


def padded_size(width, height, padding):
    left, top, right, bottom = padding
    return int(width) + left + right, int(height) + top + bottom
