#!/usr/bin/env python3
"""Inference for confidence(2)+semantic ordered_corners(8), no bbox output head."""
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
from qr_model import build_ordered_corner_fsd


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preprocess(image_bgr, input_mode):
    if image_bgr.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
        raise ValueError("Network input must be portrait W,H=240,320; got %d,%d" %
                         (image_bgr.shape[1], image_bgr.shape[0]))
    if input_mode == "y":
        converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)[:, :, 0:1]
    elif input_mode == "rgb":
        converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif input_mode == "yuv444":
        converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    else:
        raise ValueError("Unknown input mode %s" % input_mode)
    tensor = torch.from_numpy(np.ascontiguousarray(converted.transpose(2, 0, 1)))
    return tensor.float().div_(255.0).unsqueeze(0)


class QRDetector(object):
    def __init__(self, fsd_repo, checkpoint, device="cuda:0", input_mode="y",
                 input_size_key=240, score_threshold=0.5,
                 nms_threshold=0.3, candidate_size=200):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.input_mode = input_mode
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.candidate_size = int(candidate_size)
        self.priors, self.feature_shapes = generate_portrait_priors()
        self.priors = self.priors.to(self.device)
        self.model = build_ordered_corner_fsd(
            fsd_repo, is_test=False, device=str(self.device),
            input_size_key=input_size_key)
        load_qr_checkpoint_strict(self.model, checkpoint)
        self.model.to(self.device).eval()
        dummy = torch.zeros(1, 1 if input_mode == "y" else 3,
                            INPUT_HEIGHT, INPUT_WIDTH, device=self.device)
        with torch.no_grad():
            confidence, corners = unpack_outputs(
                self.model(dummy), self.priors.size(0))
        print("QRDetector W,H=240,320 priors=%d outputs=%s/%s bbox_head=NONE device=%s" %
              (self.priors.size(0), tuple(confidence.shape),
               tuple(corners.shape), self.device))

    def predict(self, image_bgr):
        tensor = preprocess(image_bgr, self.input_mode).to(self.device)
        with torch.no_grad():
            confidence, encoded_corners = unpack_outputs(
                self.model(tensor), self.priors.size(0))
            scores = torch.softmax(confidence[0], dim=1)[:, 1]
            points = decode_ordered_corners(encoded_corners[0], self.priors)
            flat = points.reshape(points.size(0), 8)
            # Axis-aligned boxes exist only as derived post-processing values.
            derived_boxes = corners_tensor_to_boxes(flat).clamp(0.0, 1.0)
            candidate = scores >= self.score_threshold
            if not candidate.any():
                return None
            indices = torch.nonzero(candidate, as_tuple=False).squeeze(1)
            candidate_scores = scores[indices]
            order = torch.argsort(candidate_scores, descending=True)[:self.candidate_size]
            indices = indices[order]
            kept_local = hard_nms(derived_boxes[indices], scores[indices],
                                  self.nms_threshold, top_k=1)
            selected = int(indices[int(kept_local[0].item())].item())
            corners = points[selected].detach().cpu().numpy()
            score = float(scores[selected].item())
        corners[:, 0] *= INPUT_WIDTH
        corners[:, 1] *= INPUT_HEIGHT
        corners[:, 0] = np.clip(corners[:, 0], 0, INPUT_WIDTH - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, INPUT_HEIGHT - 1)
        # Never reorder here: output index is the learned QR-native direction.
        derived_bbox = corners_to_bbox(corners, INPUT_WIDTH, INPUT_HEIGHT)
        return {
            "score": score,
            "ordered_corners": [[float(x), float(y)] for x, y in corners],
            "corner_order": list(SEMANTIC_CORNER_ORDER),
            "derived_bbox_xyxy": [float(value) for value in derived_bbox]}


def list_images(path):
    if os.path.isfile(path):
        return [path]
    result = []
    for parent, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                result.append(os.path.join(parent, name))
    return sorted(result)


def draw_result(image, result):
    if result is None:
        return image
    out = image.copy()
    corners = np.asarray(result["ordered_corners"], np.int32)
    cv2.polylines(out, [corners], True, (0, 255, 0), 2)
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
    for index, (x, y) in enumerate(corners):
        cv2.circle(out, (int(x), int(y)), 4, colors[index], -1)
        cv2.putText(out, "P%d" % index, (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[index], 1)
    cv2.putText(out, "QR %.3f" % result["score"], (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", choices=("y", "rgb", "yuv444"), default="y")
    parser.add_argument("--input-size-key", type=int, default=240)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--candidate-size", type=int, default=200)
    parser.add_argument("--rotate-landscape-cw", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    detector = QRDetector(
        args.fsd_repo, args.checkpoint, args.device, args.input_mode,
        args.input_size_key, args.score_threshold, args.nms_threshold,
        args.candidate_size)
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
        result = detector.predict(image)
        records.append({"image": path, "processed_width": image.shape[1],
                        "processed_height": image.shape[0],
                        "rotated_clockwise": rotated, "detection": result})
        name = os.path.basename(path)
        cv2.imwrite(os.path.join(args.output, name), draw_result(image, result))
        print("%s: %s" % (name, "QR" if result else "no QR"))
    with open(os.path.join(args.output, "predictions.json"), "w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()
