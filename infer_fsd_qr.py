#!/usr/bin/env python3
"""Multi-QR inference for confidence(2)+ordered_corners(8), YUV input."""
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np
import torch

from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, SEMANTIC_CORNER_ORDER,
                       corners_tensor_to_boxes, corners_to_bbox,
                       decode_ordered_corners, generate_portrait_priors,
                       hard_nms, load_qr_checkpoint_strict, unpack_outputs)
from qr_dataset import bgr_to_yuv_tensor
from qr_model import build_ordered_corner_fsd


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def letterbox(image_bgr):
    old_h, old_w = image_bgr.shape[:2]
    scale = min(float(INPUT_WIDTH) / old_w, float(INPUT_HEIGHT) / old_h)
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    left = (INPUT_WIDTH - new_w) // 2
    top = (INPUT_HEIGHT - new_h) // 2
    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 127, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, {"scale": scale, "left": left, "top": top,
                    "source_width": old_w, "source_height": old_h}


def preprocess(image_bgr):
    network_image, meta = letterbox(image_bgr)
    return bgr_to_yuv_tensor(network_image).unsqueeze(0), meta


def restore_points(points_normalized, meta):
    points = np.asarray(points_normalized, np.float32).reshape(4, 2).copy()
    points[:, 0] = (points[:, 0] * INPUT_WIDTH - meta["left"]) / meta["scale"]
    points[:, 1] = (points[:, 1] * INPUT_HEIGHT - meta["top"]) / meta["scale"]
    points[:, 0] = np.clip(points[:, 0], 0, meta["source_width"] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, meta["source_height"] - 1)
    return points


def valid_quad(points, image_width, image_height, min_area_ratio=0.0005,
               max_edge_ratio=8.0):
    contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    area = abs(float(cv2.contourArea(contour)))
    if area < min_area_ratio * image_width * image_height:
        return False
    edges = np.sqrt(np.sum((points - np.roll(points, -1, axis=0)) ** 2, axis=1))
    if float(edges.min()) < 1.0 or float(edges.max() / edges.min()) > max_edge_ratio:
        return False
    return True


class QRDetector(object):
    def __init__(self, fsd_repo, checkpoint, device="cuda:0",
                 input_size_key=240, score_threshold=0.80,
                 nms_threshold=0.3, candidate_size=400, max_detections=20):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.candidate_size = int(candidate_size)
        self.max_detections = int(max_detections)
        self.priors, self.feature_shapes = generate_portrait_priors()
        self.priors = self.priors.to(self.device)
        self.model = build_ordered_corner_fsd(
            fsd_repo, is_test=False, device=str(self.device),
            input_size_key=input_size_key)
        load_qr_checkpoint_strict(self.model, checkpoint)
        self.model.to(self.device).eval()
        dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=self.device)
        with torch.no_grad():
            confidence, corners = unpack_outputs(
                self.model(dummy), self.priors.size(0))
        print("QRDetector YUV NCHW=%s priors=%d outputs=%s/%s "
              "bbox_head=NONE max_detections=%d device=%s" %
              (tuple(dummy.shape), self.priors.size(0), tuple(confidence.shape),
               tuple(corners.shape), self.max_detections, self.device))

    def predict(self, image_bgr):
        tensor, meta = preprocess(image_bgr)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            confidence, encoded_corners = unpack_outputs(
                self.model(tensor), self.priors.size(0))
            scores = torch.softmax(confidence[0], dim=1)[:, 1]
            points = decode_ordered_corners(encoded_corners[0], self.priors)
            derived_boxes = corners_tensor_to_boxes(
                points.reshape(points.size(0), 8)).clamp(0.0, 1.0)
            indices = torch.nonzero(
                scores >= self.score_threshold, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                return []
            order = torch.argsort(scores[indices], descending=True)
            indices = indices[order[:self.candidate_size]]
            kept_local = hard_nms(
                derived_boxes[indices], scores[indices],
                self.nms_threshold, top_k=self.max_detections)
            selected = indices[kept_local]
            selected_points = points[selected].detach().cpu().numpy()
            selected_scores = scores[selected].detach().cpu().numpy()

        detections = []
        for normalized, score in zip(selected_points, selected_scores):
            corners = restore_points(normalized, meta)
            if not valid_quad(corners, meta["source_width"], meta["source_height"]):
                continue
            derived_bbox = corners_to_bbox(
                corners, meta["source_width"], meta["source_height"])
            detections.append({
                "score": float(score),
                "ordered_corners": [[float(x), float(y)] for x, y in corners],
                "corner_order": list(SEMANTIC_CORNER_ORDER),
                "derived_bbox_xyxy": [float(value) for value in derived_bbox]})
        return detections


def list_images(path):
    if os.path.isfile(path):
        return [path]
    result = []
    for parent, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                result.append(os.path.join(parent, name))
    return sorted(result)


def draw_results(image, detections):
    out = image.copy()
    palette = [(0, 255, 0), (255, 128, 0), (0, 200, 255),
               (255, 0, 255), (255, 255, 0)]
    for detection_index, result in enumerate(detections):
        color = palette[detection_index % len(palette)]
        corners = np.asarray(result["ordered_corners"], np.int32)
        cv2.polylines(out, [corners], True, color, 2)
        for point_index, (x, y) in enumerate(corners):
            cv2.circle(out, (int(x), int(y)), 3, color, -1)
            cv2.putText(out, "%d:P%d" % (detection_index, point_index),
                        (int(x) + 3, int(y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        x1, y1 = corners_to_bbox(corners)[:2].astype(np.int32)
        cv2.putText(out, "QR%d %.3f" % (detection_index, result["score"]),
                    (int(x1), max(14, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", choices=("yuv", "yuv444"), default="yuv",
                        help="Compatibility flag; inference is always 3-channel YUV444")
    parser.add_argument("--input-size-key", type=int, default=240)
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--candidate-size", type=int, default=400)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--rotate-landscape-cw", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, args.input_size_key,
        args.score_threshold, args.nms_threshold, args.candidate_size,
        args.max_detections)
    records = []
    for path in list_images(args.input):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print("SKIP unreadable: %s" % path)
            continue
        rotated = False
        if args.rotate_landscape_cw and image.shape[1] > image.shape[0]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated = True
        detections = detector.predict(image)
        records.append({
            "image": path, "processed_width": image.shape[1],
            "processed_height": image.shape[0], "rotated_clockwise": rotated,
            "detections": detections})
        name = os.path.basename(path)
        cv2.imwrite(os.path.join(args.output, name), draw_results(image, detections))
        print("%s: %d QR" % (name, len(detections)))
    with open(os.path.join(args.output, "detections.json"), "w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()
