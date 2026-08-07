#!/usr/bin/env python3
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_loss import QROrderedCornerLoss


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


if __name__ == "__main__":
    test_negative_only_batch_has_classification_gradient()
    print("PASS")
