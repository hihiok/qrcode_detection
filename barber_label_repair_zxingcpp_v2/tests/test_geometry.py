import numpy as np
from PIL import Image

from barber_label_repair_zxingcpp_v2.detector import preprocess
from barber_label_repair_zxingcpp_v2.geometry import (
    apply_homography, best_vertex_assignment, geometric_order,
    quad_geometry_issue, rotate_image_and_h, semantic_to_manual, solve_homography,
)


def test_homography_round_trip():
    src = np.array([[3, 4], [90, 8], [80, 70], [4, 60]], float)
    dst = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], float)
    h = solve_homography(src, dst)
    assert np.allclose(apply_homography(h, src), dst)
    assert np.allclose(apply_homography(np.linalg.inv(h), dst), src)


def test_rotations_inverse_coordinates():
    image = Image.new("RGB", (240, 320), "white")
    p = np.array([[0, 0], [239, 0], [239, 319], [0, 319]], float)
    for k in range(4):
        _, h = rotate_image_and_h(image, np.eye(3), k)
        assert np.allclose(apply_homography(np.linalg.inv(h),
                                           apply_homography(h, p)), p)


def test_geometric_order_preserves_vertices():
    p = np.array([[90, 80], [10, 10], [10, 80], [90, 10]], float)
    idx = geometric_order(p)
    assert idx.tolist() == [1, 3, 0, 2]
    assert sorted(idx.tolist()) == [0, 1, 2, 3]


def test_assignment_and_semantic_order():
    manual = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    detected = manual[[2, 3, 0, 1]] + .1
    mapping, _, maximum = best_vertex_assignment(detected, manual)
    assert mapping == (2, 3, 0, 1)
    assert maximum < .01
    assert semantic_to_manual((0, 1, 2, 3), mapping) == mapping


def test_preprocessors_no_cv2_and_shape():
    im = Image.new("RGB", (64, 48), (120, 130, 140))
    for name in ("rgb", "gray", "gamma_0.7", "gamma_1.4", "unsharp",
                 "otsu", "adaptive", "invert"):
        assert preprocess(im, name).size == im.size


def test_degenerate_boundary_quad_is_identified():
    boundary = np.array([[240, 10], [240, 40], [240, 80], [240, 120]], float)
    assert quad_geometry_issue(boundary) == "bbox_width_lt_4px"
    valid = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], float)
    assert quad_geometry_issue(valid) is None
