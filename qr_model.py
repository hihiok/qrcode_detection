#!/usr/bin/env python3
"""Adapt the single-Y FSD to YUV input and ordered QR corners."""
from __future__ import print_function

import copy
import inspect

import torch
from torch import nn

from qr_common import NUM_CLASSES, add_repo_to_path


_ORDERED_SUBCLASSES = {}


def _set_child(root, path, value):
    parent = root
    pieces = path.split(".")
    for piece in pieces[:-1]:
        parent = parent[int(piece)] if piece.isdigit() else getattr(parent, piece)
    leaf = pieces[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def convert_input_to_yuv(model):
    """Replace the first 1-channel Conv2d with a 3-channel YUV Conv2d.

    The old kernel is copied to Y and U/V start at zero, so a single-Y
    checkpoint has identical initial behavior before chroma fine-tuning.
    """
    first = None
    search_root = getattr(model, "base_net", model)
    prefix = "base_net." if search_root is not model else ""
    for name, module in search_root.named_modules():
        if name and isinstance(module, nn.Conv2d):
            first = (prefix + name, module)
            break
    if first is None:
        for name, module in model.named_modules():
            if name and isinstance(module, nn.Conv2d):
                first = (name, module)
                break
    if first is None:
        raise RuntimeError("FSD contains no Conv2d input layer")
    name, conv = first
    if conv.in_channels == 3:
        model.yuv_input = True
        return model
    if conv.in_channels != 1 or conv.groups != 1:
        raise RuntimeError("Expected first Conv2d with one input channel; got %s" % conv)
    replacement = nn.Conv2d(
        3, conv.out_channels, conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=1,
        bias=conv.bias is not None, padding_mode=conv.padding_mode)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, 0:1].copy_(conv.weight)
        if replacement.bias is not None:
            replacement.bias.copy_(conv.bias)
    _set_child(model, name, replacement)
    model.yuv_input = True
    model.yuv_input_conv_name = name
    return model


def _new_conv_like(conv, out_channels):
    replacement = nn.Conv2d(
        conv.in_channels, int(out_channels), conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None)
    nn.init.xavier_uniform_(replacement.weight)
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)
    return replacement


def _replace_last_conv(module, multiplier):
    """Deep-copy a header and multiply its final Conv2d output channels."""
    cloned = copy.deepcopy(module)
    if isinstance(cloned, nn.Conv2d):
        return _new_conv_like(cloned, cloned.out_channels * multiplier)
    candidates = [(name, child) for name, child in cloned.named_modules()
                  if name and isinstance(child, nn.Conv2d)]
    if not candidates:
        raise RuntimeError("Regression header contains no Conv2d: %s" % type(module))
    path, final_conv = candidates[-1]
    parent = cloned
    pieces = path.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece) if not piece.isdigit() else parent[int(piece)]
    leaf = pieces[-1]
    replacement = _new_conv_like(final_conv, final_conv.out_channels * multiplier)
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)
    return cloned


def _last_conv_out_channels(module):
    values = [child.out_channels for child in module.modules()
              if isinstance(child, nn.Conv2d)]
    if not values:
        raise RuntimeError("Header contains no Conv2d: %s" % type(module))
    return int(values[-1])


def _compute_ordered_corner_header(self, index, feature):
    confidence = self.classification_headers[index](feature)
    confidence = confidence.permute(0, 2, 3, 1).contiguous()
    confidence = confidence.view(confidence.size(0), -1, self.num_classes)
    corners = self.regression_headers[index](feature)
    corners = corners.permute(0, 2, 3, 1).contiguous()
    corners = corners.view(corners.size(0), -1, 8)
    return confidence, corners


def convert_regression_head_to_ordered_corners(model, num_classes=NUM_CLASSES):
    """Change only the existing single regression branch from A*4 to A*8."""
    if not hasattr(model, "regression_headers") or not hasattr(model, "classification_headers"):
        raise RuntimeError("Base FSD lacks regression_headers/classification_headers")
    if len(model.regression_headers) != len(model.classification_headers):
        raise RuntimeError("Regression/classification header counts differ")
    for index in range(len(model.regression_headers)):
        old_out = _last_conv_out_channels(model.regression_headers[index])
        class_out = _last_conv_out_channels(model.classification_headers[index])
        if old_out % 4 != 0 or class_out % int(num_classes) != 0:
            raise RuntimeError("Header %d has incompatible outputs reg=%d cls=%d" %
                               (index, old_out, class_out))
        anchors_reg = old_out // 4
        anchors_cls = class_out // int(num_classes)
        if anchors_reg != anchors_cls:
            raise RuntimeError("Header %d anchor mismatch: %d vs %d" %
                               (index, anchors_reg, anchors_cls))
        model.regression_headers[index] = _replace_last_conv(
            model.regression_headers[index], 2)
    # Override on the class, not with an instance-bound MethodType. DataParallel
    # shallow-copies module __dict__, which can leave a bound method pointing to
    # the original device. A dynamic subclass replicates safely on every GPU.
    base_class = model.__class__
    if base_class not in _ORDERED_SUBCLASSES:
        ordered_class = type(
            "OrderedCorner%s" % base_class.__name__, (base_class,),
            {"compute_header": _compute_ordered_corner_header})
        ordered_class.__module__ = __name__
        _ORDERED_SUBCLASSES[base_class] = ordered_class
    model.__class__ = _ORDERED_SUBCLASSES[base_class]
    model.ordered_corner_output = True
    return model


def build_ordered_corner_fsd(repo_root, num_classes=NUM_CLASSES, is_test=False,
                             device="cpu", input_size_key=240):
    """Build the exact nodilation factory, then minimally change its 4-D head."""
    add_repo_to_path(repo_root)
    from vision.ssd.config.fd_config import define_img_size
    try:
        define_img_size(input_size_key)
    except KeyError:
        if int(input_size_key) != 112:
            raise
        # The original fd_config enumerates only 4:3 presets.  The network is
        # fully convolutional and we always request raw outputs (is_test=False),
        # so a 112x112 Stage-2 forward does not consume fd_config.priors.
        # Our 112x112 priors are generated explicitly in qr_common.
        print("Using custom raw-output FSD input 112x112; skipping fd_config priors")
    from vision.ssd.mb_tiny_RFB_fd_3 import create_Mb_Tiny_RFB_fd_3_nodilation
    signature = inspect.signature(create_Mb_Tiny_RFB_fd_3_nodilation)
    kwargs = {}
    if "is_test" in signature.parameters:
        # Always keep raw logits/offsets. The original test path decodes 4-D
        # boxes and therefore must not run for the 8-D corner representation.
        kwargs["is_test"] = False
    if "device" in signature.parameters:
        kwargs["device"] = device
    model = create_Mb_Tiny_RFB_fd_3_nodilation(num_classes, **kwargs)
    model = convert_input_to_yuv(model)
    return convert_regression_head_to_ordered_corners(model, num_classes)
