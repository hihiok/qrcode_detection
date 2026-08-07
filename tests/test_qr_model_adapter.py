#!/usr/bin/env python3
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_model import (convert_input_to_yuv,
                      convert_regression_head_to_ordered_corners)


class FakeFSD(nn.Module):
    def __init__(self):
        super(FakeFSD, self).__init__()
        self.num_classes = 2
        self.base_net = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1))
        self.classification_headers = nn.ModuleList([
            nn.Conv2d(4, 3 * 2, 1), nn.Conv2d(4, 2 * 2, 1)])
        self.regression_headers = nn.ModuleList([
            nn.Conv2d(4, 3 * 4, 1),
            nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.Conv2d(4, 2 * 4, 1))])

    def forward(self, features):
        confidences, corners = [], []
        for index, feature in enumerate(features):
            confidence, corner = self.compute_header(index, feature)
            confidences.append(confidence)
            corners.append(corner)
        return torch.cat(confidences, 1), torch.cat(corners, 1)


def test_adapter():
    model = convert_input_to_yuv(FakeFSD())
    assert model.base_net[0].in_channels == 3
    assert torch.allclose(model.base_net[0].weight[:, 1:],
                          torch.zeros_like(model.base_net[0].weight[:, 1:]))
    model = convert_regression_head_to_ordered_corners(model)
    confidence, corners = model([
        torch.zeros(1, 4, 3, 2), torch.zeros(1, 4, 2, 1)])
    assert confidence.shape == (1, 22, 2)
    assert corners.shape == (1, 22, 8)
    assert model.regression_headers[0].out_channels == 24
    assert model.regression_headers[1][-1].out_channels == 16


if __name__ == "__main__":
    test_adapter()
    print("PASS")
