#!/usr/bin/env python3
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_elementwise(predicted, target):
    difference = torch.abs(predicted - target)
    return torch.where(difference < 1.0, 0.5 * difference * difference,
                       difference - 0.5)


class QROrderedCornerLoss(nn.Module):
    """SSD hard-negative classification + ordered P0..P3 regression."""
    def __init__(self, neg_pos_ratio=3, corner_weight=2.0,
                 classification_weight=1.0, min_negatives_per_image=64,
                 priors=None, center_variance=0.1,
                 normalized_corner_weight=0.0, edge_weight=0.0,
                 orientation_weight=0.0):
        super(QROrderedCornerLoss, self).__init__()
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.corner_weight = float(corner_weight)
        self.classification_weight = float(classification_weight)
        self.min_negatives_per_image = int(min_negatives_per_image)
        self.center_variance = float(center_variance)
        self.normalized_corner_weight = float(normalized_corner_weight)
        self.edge_weight = float(edge_weight)
        self.orientation_weight = float(orientation_weight)
        if priors is None:
            self.register_buffer("priors", torch.empty(0, 4))
        else:
            self.register_buffer("priors", torch.as_tensor(
                priors, dtype=torch.float32).reshape(-1, 4).clone())

    def _decode(self, values):
        if self.priors.numel() == 0:
            raise ValueError("Decoded geometry losses require priors")
        if values.size(1) != self.priors.size(0):
            raise ValueError("Prediction/prior mismatch: %d vs %d" %
                             (values.size(1), self.priors.size(0)))
        points = values.reshape(values.size(0), values.size(1), 4, 2)
        scales = self.center_variance * self.priors[:, 2:]
        centers = self.priors[:, :2]
        return points * scales[None, :, None, :] + centers[None, :, None, :]

    def _geometry_losses(self, corners, target_corners, positive):
        zero = corners.sum() * 0.0
        if (not positive.any() or
                (self.normalized_corner_weight <= 0.0 and
                 self.edge_weight <= 0.0 and
                 self.orientation_weight <= 0.0)):
            return zero, zero, zero
        predicted = self._decode(corners)[positive]
        target = self._decode(target_corners)[positive]
        minimum = target.min(dim=1)[0]
        maximum = target.max(dim=1)[0]
        diagonal = torch.sqrt(torch.clamp(
            ((maximum - minimum) ** 2).sum(dim=1), min=1e-8))

        point_distance = torch.sqrt(torch.clamp(
            ((predicted - target) ** 2).sum(dim=2), min=1e-12))
        normalized_corner = (point_distance / diagonal[:, None]).mean()

        predicted_edges = torch.roll(predicted, shifts=-1, dims=1) - predicted
        target_edges = torch.roll(target, shifts=-1, dims=1) - target
        predicted_lengths = torch.sqrt(torch.clamp(
            (predicted_edges ** 2).sum(dim=2), min=1e-12))
        target_lengths = torch.sqrt(torch.clamp(
            (target_edges ** 2).sum(dim=2), min=1e-12))
        edge = (torch.abs(predicted_lengths - target_lengths) /
                diagonal[:, None]).mean()

        next_edges = torch.roll(predicted_edges, shifts=-1, dims=1)
        cross = (predicted_edges[:, :, 0] * next_edges[:, :, 1] -
                 predicted_edges[:, :, 1] * next_edges[:, :, 0])
        normalized_cross = cross / (diagonal[:, None] ** 2)
        # P0->P1->P2->P3 has positive signed area in image coordinates.
        orientation = torch.relu(1e-4 - normalized_cross).mean()
        return normalized_corner, edge, orientation

    def forward(self, confidence, corners, labels, target_corners):
        positive = labels > 0
        num_positive_per_image = positive.long().sum(dim=1)
        normalizer = torch.clamp(num_positive_per_image.sum().float(), min=1.0)
        if positive.any():
            corner_loss = smooth_l1_elementwise(
                corners[positive], target_corners[positive]).sum() / normalizer
        else:
            corner_loss = corners.sum() * 0.0

        normalized_corner_loss, edge_loss, orientation_loss = \
            self._geometry_losses(corners, target_corners, positive)

        with torch.no_grad():
            background_loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            background_loss[positive] = -1e9
            _, order = background_loss.sort(dim=1, descending=True)
            _, rank = order.sort(dim=1)
            wanted = self.neg_pos_ratio * num_positive_per_image
            minimum = torch.full_like(wanted, self.min_negatives_per_image)
            wanted = torch.max(wanted, minimum)
            max_negatives = confidence.size(1) - num_positive_per_image
            num_negative = torch.min(wanted, max_negatives)
            negative = rank < num_negative.unsqueeze(1)
        selected = positive | negative
        if selected.any():
            classification_normalizer = torch.max(
                normalizer,
                selected.long().sum().float() / float(self.neg_pos_ratio + 1))
            classification_loss = F.cross_entropy(
                confidence[selected], labels[selected], reduction="sum") / \
                classification_normalizer
        else:
            classification_loss = confidence.sum() * 0.0
        total = (self.corner_weight * corner_loss +
                 self.classification_weight * classification_loss +
                 self.normalized_corner_weight * normalized_corner_loss +
                 self.edge_weight * edge_loss +
                 self.orientation_weight * orientation_loss)
        return {"total": total, "corner": corner_loss,
                "classification": classification_loss,
                "normalized_corner": normalized_corner_loss,
                "edge": edge_loss, "orientation": orientation_loss}
