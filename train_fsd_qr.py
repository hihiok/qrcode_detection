#!/usr/bin/env python3
"""Train multi-QR FSD: three-channel YUV -> confidence(2)+corners(8)."""
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, SEMANTIC_CORNER_ORDER,
                       generate_portrait_priors, load_fd_pretrained,
                       load_qr_checkpoint_strict, save_json, unpack_outputs)
from qr_dataset import QRDataset
from qr_loss import QROrderedCornerLoss
from qr_model import build_ordered_corner_fsd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pretrained-fd", default=None,
                        help="Original bbox(4) FSD checkpoint; its reg head is skipped")
    parser.add_argument("--resume", default=None,
                        help="corners(8) QR checkpoint; loaded strictly")
    parser.add_argument("--input-mode", choices=("yuv", "yuv444"), default="yuv",
                        help="Compatibility flag; training is always 3-channel YUV444")
    parser.add_argument("--input-size-key", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", default="80,100,150")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--iou-threshold", type=float, default=0.35)
    parser.add_argument("--corner-weight", type=float, default=2.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--min-negatives-per-image", type=int, default=64)
    parser.add_argument("--freeze-base-net", action="store_true")
    parser.add_argument("--freeze-net", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def configure_trainable(model, args):
    if args.freeze_base_net and args.freeze_net:
        raise ValueError("Choose at most one freeze option")
    if args.freeze_base_net:
        freeze(model.base_net)
    if args.freeze_net:
        for name in ("base_net", "source_layer_add_ons", "extras"):
            freeze(getattr(model, name))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters")
    return parameters


def compute_losses(model, batch, criterion, device, num_priors):
    images, labels, target_corners = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    target_corners = target_corners.to(device, non_blocking=True)
    confidence, corners = unpack_outputs(model(images), num_priors)
    return criterion(confidence, corners, labels, target_corners)


def run_epoch(model, loader, criterion, device, num_priors,
              optimizer=None, epoch=0, log_interval=100):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "corner": 0.0, "classification": 0.0}
    batches = 0
    started = time.time()
    for step, batch in enumerate(loader):
        if training:
            optimizer.zero_grad()
            losses = compute_losses(model, batch, criterion, device, num_priors)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite loss at epoch=%d step=%d" %
                                         (epoch, step))
            losses["total"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = compute_losses(model, batch, criterion, device, num_priors)
        for key in sums:
            sums[key] += float(losses[key].item())
        batches += 1
        if training and step and step % log_interval == 0:
            logging.info("epoch=%d step=%d total=%.5f corner=%.5f cls=%.5f",
                         epoch, step, sums["total"] / batches,
                         sums["corner"] / batches,
                         sums["classification"] / batches)
    result = {key: value / max(batches, 1) for key, value in sums.items()}
    result["seconds"] = time.time() - started
    return result


def preflight(model, priors, device):
    channels = 3
    model.eval()
    dummy = torch.zeros(1, channels, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    with torch.no_grad():
        confidence, corners = unpack_outputs(model(dummy), priors.size(0))
    print("PRECHECK factory=create_Mb_Tiny_RFB_fd_3_nodilation")
    print("PRECHECK tensor NCHW=%s; W,H=%d,%d" %
          (tuple(dummy.shape), INPUT_WIDTH, INPUT_HEIGHT))
    print("PRECHECK feature_maps=40x30,20x15,10x8,5x4 priors=%d" % priors.size(0))
    print("PRECHECK confidence=%s ordered_corners=%s; bbox_output=NONE" %
          (tuple(confidence.shape), tuple(corners.shape)))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.pretrained_fd and args.resume:
        raise ValueError("Use either --pretrained-fd or --resume, not both")
    set_seed(args.seed)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    use_cuda = torch.cuda.is_available() and args.gpus.lower() != "cpu"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    priors, feature_shapes = generate_portrait_priors()
    model = build_ordered_corner_fsd(
        args.fsd_repo, is_test=False, device=str(device),
        input_size_key=args.input_size_key)
    if args.resume:
        load_qr_checkpoint_strict(model, args.resume)
    elif args.pretrained_fd:
        load_fd_pretrained(model, args.pretrained_fd)
    model.to(device)
    priors_device = priors.to(device)
    preflight(model, priors_device, device)
    parameters = configure_trainable(model, args)
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    train_data = QRDataset(
        os.path.join(args.data_root, "train"), priors, True,
        args.iou_threshold, args.seed)
    val_data = QRDataset(
        os.path.join(args.data_root, "val"), priors, False,
        args.iou_threshold, args.seed + 1)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_cuda, drop_last=True)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda)
    criterion = QROrderedCornerLoss(
        3, args.corner_weight, args.classification_weight,
        args.min_negatives_per_image).to(device)
    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    milestones = [int(value) for value in args.milestones.split(",") if value.strip()]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    metadata = vars(args).copy()
    metadata.update({
        "input_width": INPUT_WIDTH, "input_height": INPUT_HEIGHT,
        "feature_shapes": feature_shapes, "num_priors": int(priors.size(0)),
        "factory": "create_Mb_Tiny_RFB_fd_3_nodilation",
        "model_outputs": {"confidence": 2, "ordered_corners": 8},
        "bbox_model_output": False,
        "input_format": "YUV444",
        "supports_zero_or_more_qr": True,
        "corner_order": list(SEMANTIC_CORNER_ORDER)})
    save_json(os.path.join(args.checkpoint_dir, "training_config.json"), metadata)
    best = float("inf")
    history_path = os.path.join(args.checkpoint_dir, "history.jsonl")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device,
                                  priors.size(0), optimizer, epoch, args.log_interval)
        val_metrics = run_epoch(model, val_loader, criterion, device,
                                priors.size(0), None, epoch, args.log_interval)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train": train_metrics, "val": val_metrics}
        with open(history_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        state = unwrap(model).state_dict()
        torch.save(state, os.path.join(args.checkpoint_dir, "qr_fsd_latest.pth"))
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(state, os.path.join(args.checkpoint_dir, "qr_fsd_best.pth"))
        logging.info("epoch=%d train=%s val=%s best=%.6f",
                     epoch, train_metrics, val_metrics, best)


if __name__ == "__main__":
    main()
