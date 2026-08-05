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
                 classification_weight=1.0):
        super(QROrderedCornerLoss, self).__init__()
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.corner_weight = float(corner_weight)
        self.classification_weight = float(classification_weight)

    def forward(self, confidence, corners, labels, target_corners):
        positive = labels > 0
        num_positive_per_image = positive.long().sum(dim=1)
        normalizer = torch.clamp(num_positive_per_image.sum().float(), min=1.0)
        if positive.any():
            corner_loss = smooth_l1_elementwise(
                corners[positive], target_corners[positive]).sum() / normalizer
        else:
            corner_loss = corners.sum() * 0.0

        with torch.no_grad():
            background_loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            background_loss[positive] = -1e9
            _, order = background_loss.sort(dim=1, descending=True)
            _, rank = order.sort(dim=1)
            max_negatives = confidence.size(1) - 1
            num_negative = torch.clamp(
                self.neg_pos_ratio * num_positive_per_image, max=max_negatives)
            negative = rank < num_negative.unsqueeze(1)
        selected = positive | negative
        if selected.any():
            classification_loss = F.cross_entropy(
                confidence[selected], labels[selected], reduction="sum") / normalizer
        else:
            classification_loss = confidence.sum() * 0.0
        total = (self.corner_weight * corner_loss +
                 self.classification_weight * classification_loss)
        return {"total": total, "corner": corner_loss,
                "classification": classification_loss}
