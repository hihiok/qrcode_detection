#!/usr/bin/env python3
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_loss import QROrderedCornerLoss
from qr_common import generate_portrait_priors


def test_negative_only_batch_has_classification_gradient():
    confidence = torch.zeros(2, 100, 2, requires_grad=True)
    corners = torch.zeros(2, 100, 8, requires_grad=True)
    labels = torch.zeros(2, 100, dtype=torch.long)
    targets = torch.zeros(2, 100, 8)
    losses = QROrderedCornerLoss(min_negatives_per_image=16)(
        confidence, corners, labels, targets)
    assert float(losses["classification"].item()) > 0
    assert float(losses["corner"].item()) == 0
    losses["total"].backward()
    assert confidence.grad is not None


def test_decoded_geometry_losses_do_not_change_output_contract():
    priors, _ = generate_portrait_priors()
    count = priors.size(0)
    confidence = torch.zeros(1, count, 2, requires_grad=True)
    predicted = torch.zeros(1, count, 8, requires_grad=True)
    target = torch.zeros(1, count, 8)
    labels = torch.zeros(1, count, dtype=torch.long)
    labels[0, 0] = 1
    target[0, 0] = 0.1
    criterion = QROrderedCornerLoss(
        priors=priors, normalized_corner_weight=4.0,
        edge_weight=1.0, orientation_weight=0.25,
        min_negatives_per_image=8)
    losses = criterion(confidence, predicted, labels, target)
    assert set(("normalized_corner", "edge", "orientation")) <= set(losses)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert predicted.grad is not None


if __name__ == "__main__":
    test_negative_only_batch_has_classification_gradient()
    test_decoded_geometry_losses_do_not_change_output_contract()
    print("PASS")
