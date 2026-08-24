#!/usr/bin/env python3
"""Two-stage QR inference: coarse quad detection then semantic corner ordering."""
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np

from infer_fsd_qr import QRDetector, draw_results, list_images, valid_quad
from qr_common import SEMANTIC_CORNER_ORDER, corners_to_bbox, validate_semantic_corners
from qr_stage2_dataset import STAGE2_MIN_BOXES, STAGE2_SIZE
from qr_two_stage_geometry import (convex_iou, order_quad_for_crop,
                                   transform_points, warp_qr_roi)


class TwoStageQRDetector(object):
    """Compose two identical FSD structures with different task semantics.

    Stage 1 returns a coarse QR quadrilateral.  Its predicted point identities
    are deliberately ignored and reordered only by image geometry for cropping.
    Stage 2 sees one rectified ROI and predicts QR-native P0,P1,P2,P3.
    """
    def __init__(self, fsd_repo, stage1_checkpoint, stage2_checkpoint,
                 device="cuda:0", stage1_score_threshold=0.50,
                 stage2_score_threshold=0.70, nms_threshold=0.30,
                 max_detections=20, crop_margin=0.20,
                 stage2_min_geometry_iou=0.20, stage1_opencv_refine=False):
        self.crop_margin = float(crop_margin)
        self.stage2_min_geometry_iou = float(stage2_min_geometry_iou)
        self.stage1 = QRDetector(
            fsd_repo, stage1_checkpoint, device, 240,
            stage1_score_threshold, nms_threshold, 400, max_detections,
            stage1_opencv_refine, 0.18, 0.20, 0.40)
        self.stage2 = QRDetector(
            fsd_repo, stage2_checkpoint, device, STAGE2_SIZE,
            stage2_score_threshold, nms_threshold, 100, 5,
            False, 0.18, 0.20, 0.40, STAGE2_SIZE, STAGE2_SIZE,
            STAGE2_MIN_BOXES)
        stage1_shapes = {key: tuple(value.shape)
                         for key, value in self.stage1.model.state_dict().items()}
        stage2_shapes = {key: tuple(value.shape)
                         for key, value in self.stage2.model.state_dict().items()}
        if stage1_shapes != stage2_shapes:
            raise RuntimeError("stage-1 and stage-2 network structures differ")
        self.num_parameters = sum(
            parameter.numel() for parameter in self.stage1.model.parameters())
        print("TWO_STAGE_ARCHITECTURE_MATCH parameters=%d outputs=confidence2+corners8" %
              self.num_parameters)

    def _run_stage2(self, image_bgr, stage1_detection):
        coarse = np.asarray(stage1_detection["ordered_corners"],
                            np.float32).reshape(4, 2)
        geometry = order_quad_for_crop(coarse)
        roi, crop_quad, matrix, inverse = warp_qr_roi(
            image_bgr, geometry, self.crop_margin,
            output_width=STAGE2_SIZE, output_height=STAGE2_SIZE)
        expected_roi_geometry = transform_points(geometry, matrix).reshape(4, 2)
        candidates = self.stage2.predict(roi)
        ranked = []
        for candidate in candidates:
            semantic_roi = np.asarray(candidate["ordered_corners"],
                                      np.float32).reshape(4, 2)
            try:
                overlap = convex_iou(semantic_roi, expected_roi_geometry)
            except ValueError:
                continue
            rank_score = float(candidate["score"]) + 0.25 * overlap
            ranked.append((rank_score, overlap, candidate, semantic_roi))
        if not ranked:
            return None, {"reason": "stage2_no_valid_candidate",
                          "stage1_score": float(stage1_detection["score"])}
        _, overlap, selected, semantic_roi = max(ranked, key=lambda item: item[0])
        if overlap < self.stage2_min_geometry_iou:
            return None, {"reason": "stage2_geometry_mismatch",
                          "stage1_score": float(stage1_detection["score"]),
                          "stage2_score": float(selected["score"]),
                          "stage2_geometry_iou": float(overlap)}
        semantic_full = transform_points(semantic_roi, inverse).reshape(4, 2)
        height, width = image_bgr.shape[:2]
        semantic_full[:, 0] = np.clip(semantic_full[:, 0], 0, width - 1)
        semantic_full[:, 1] = np.clip(semantic_full[:, 1], 0, height - 1)
        try:
            validate_semantic_corners(semantic_full, "two-stage output")
        except ValueError as exc:
            return None, {"reason": "invalid_semantic_quad",
                          "detail": str(exc),
                          "stage1_score": float(stage1_detection["score"]),
                          "stage2_score": float(selected["score"])}
        if not valid_quad(semantic_full, width, height):
            return None, {"reason": "invalid_full_frame_quad",
                          "stage1_score": float(stage1_detection["score"]),
                          "stage2_score": float(selected["score"])}
        bbox = corners_to_bbox(semantic_full, width, height)
        stage1_score = float(stage1_detection["score"])
        stage2_score = float(selected["score"])
        result = {
            "score": min(stage1_score, stage2_score),
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "stage2_geometry_iou": float(overlap),
            "ordered_corners": [[float(x), float(y)] for x, y in semantic_full],
            "corner_order": list(SEMANTIC_CORNER_ORDER),
            "derived_bbox_xyxy": [float(value) for value in bbox],
            "stage1_geometry_corners": geometry.tolist(),
            "stage2_roi_corners": semantic_roi.tolist(),
            "stage2_crop_quad_source_xy": crop_quad.tolist(),
        }
        return result, None

    def predict(self, image_bgr, return_rejected=False):
        stage1_detections = self.stage1.predict(image_bgr)
        accepted = []
        rejected = []
        for index, detection in enumerate(stage1_detections):
            result, rejection = self._run_stage2(image_bgr, detection)
            if result is not None:
                accepted.append(result)
            else:
                rejection["stage1_index"] = index
                rejected.append(rejection)
        if return_rejected:
            return accepted, rejected, stage1_detections
        return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-score-threshold", type=float, default=0.50)
    parser.add_argument("--stage2-score-threshold", type=float, default=0.70)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--crop-margin", type=float, default=0.20)
    parser.add_argument("--stage2-min-geometry-iou", type=float, default=0.20)
    parser.add_argument("--stage1-opencv-refine", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    detector = TwoStageQRDetector(
        args.fsd_repo, args.stage1_checkpoint, args.stage2_checkpoint,
        args.device, args.stage1_score_threshold,
        args.stage2_score_threshold, args.nms_threshold,
        args.max_detections, args.crop_margin,
        args.stage2_min_geometry_iou, args.stage1_opencv_refine)
    records = []
    for path in list_images(args.input):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print("SKIP unreadable: %s" % path)
            continue
        detections, rejected, stage1 = detector.predict(image, True)
        records.append({"image": path, "stage1_detections": len(stage1),
                        "accepted": detections, "rejected": rejected})
        cv2.imwrite(os.path.join(args.output, os.path.basename(path)),
                    draw_results(image, detections))
        print("%s: stage1=%d ordered=%d rejected=%d" %
              (path, len(stage1), len(detections), len(rejected)))
    with open(os.path.join(args.output, "two_stage_detections.json"), "w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()
