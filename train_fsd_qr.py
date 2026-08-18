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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from dataset_v2_manifest import training_sources
from qr_common import (INPUT_HEIGHT, INPUT_WIDTH, SEMANTIC_CORNER_ORDER,
                       generate_portrait_priors, load_fd_pretrained,
                       load_qr_checkpoint_strict, save_json, unpack_outputs)
from qr_dataset import QRDataset
from qr_loss import QROrderedCornerLoss
from qr_model import build_ordered_corner_fsd
from qr_schema import SCHEMA_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd-repo", required=True)
    parser.add_argument("--data-root", dest="data_roots", action="append",
                        help="Dataset root containing train/val; repeat for multiple datasets")
    parser.add_argument("--dataset-manifest", default=None,
                        help="Frozen qr_dataset_v2_manifest_v1 with source weights")
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Weighted training draws; 0 uses total source length")
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
    parser.add_argument("--normalized-corner-weight", type=float, default=4.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.25)
    parser.add_argument("--neg-pos-ratio", type=int, default=3)
    parser.add_argument("--min-negatives-per-image", type=int, default=64)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
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
              optimizer=None, epoch=0, log_interval=100, gradient_clip=0.0):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "corner": 0.0, "classification": 0.0,
            "normalized_corner": 0.0, "edge": 0.0,
            "orientation": 0.0}
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
            if gradient_clip and gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                losses = compute_losses(model, batch, criterion, device, num_priors)
        for key in sums:
            sums[key] += float(losses[key].item())
        batches += 1
        if training and step and step % log_interval == 0:
            logging.info("epoch=%d step=%d total=%.5f corner=%.5f "
                         "norm_corner=%.5f edge=%.5f orient=%.5f cls=%.5f",
                         epoch, step, sums["total"] / batches,
                         sums["corner"] / batches,
                         sums["normalized_corner"] / batches,
                         sums["edge"] / batches,
                         sums["orientation"] / batches,
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


def source_weighted_sampler(datasets, source_weights, samples_per_epoch):
    if len(datasets) != len(source_weights):
        raise ValueError("Dataset/weight mismatch")
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Cannot sample an empty source")
    normalized = np.asarray(source_weights, dtype=np.float64)
    if (normalized <= 0).any():
        raise ValueError("Source sampling weights must be positive")
    normalized /= normalized.sum()
    per_item = []
    for dataset, probability in zip(datasets, normalized):
        per_item.extend([float(probability) / len(dataset)] * len(dataset))
    draws = int(samples_per_epoch) if samples_per_epoch > 0 else sum(
        len(dataset) for dataset in datasets)
    return WeightedRandomSampler(
        torch.as_tensor(per_item, dtype=torch.double), draws,
        replacement=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.pretrained_fd and args.resume:
        raise ValueError("Use either --pretrained-fd or --resume, not both")
    if args.dataset_manifest and args.data_roots:
        raise ValueError("Use --dataset-manifest or --data-root, not both")
    if not args.dataset_manifest and not args.data_roots:
        raise ValueError("One --dataset-manifest or at least one --data-root is required")
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
    if args.dataset_manifest:
        source_specs = training_sources(args.dataset_manifest)
        data_roots = [item[0] for item in source_specs]
        source_weights = [item[1] for item in source_specs]
        source_names = [item[2] for item in source_specs]
    else:
        data_roots = list(args.data_roots)
        source_weights = None
        source_names = [os.path.basename(os.path.abspath(root)) for root in data_roots]
    train_sets = [
        QRDataset(os.path.join(root, "train"), priors, True,
                  args.iou_threshold, args.seed + index)
        for index, root in enumerate(data_roots)
    ]
    val_sets = [
        QRDataset(os.path.join(root, "val"), priors, False,
                  args.iou_threshold, args.seed + 1000 + index)
        for index, root in enumerate(data_roots)
    ]
    train_data = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_data = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    sampler = None
    if source_weights is not None:
        sampler = source_weighted_sampler(
            train_sets, source_weights, args.samples_per_epoch)
    logging.info("dataset_sources=%s roots=%s weights=%s train_images=%d "
                 "val_images=%d weighted_draws=%s",
                 source_names, data_roots, source_weights, len(train_data),
                 len(val_data), len(sampler) if sampler is not None else None)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=use_cuda, drop_last=True)
    val_loaders = [DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda)
        for dataset in val_sets]
    criterion = QROrderedCornerLoss(
        args.neg_pos_ratio, args.corner_weight, args.classification_weight,
        args.min_negatives_per_image, priors=priors,
        normalized_corner_weight=args.normalized_corner_weight,
        edge_weight=args.edge_weight,
        orientation_weight=args.orientation_weight).to(device)
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
        "annotation_schema": SCHEMA_VERSION,
        "dataset_roots_resolved": data_roots,
        "dataset_source_names": source_names,
        "dataset_source_sampling_weights": source_weights,
        "supports_zero_or_more_qr": True,
        "corner_order": list(SEMANTIC_CORNER_ORDER)})
    save_json(os.path.join(args.checkpoint_dir, "training_config.json"), metadata)
    best = float("inf")
    history_path = os.path.join(args.checkpoint_dir, "history.jsonl")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device,
                                  priors.size(0), optimizer, epoch,
                                  args.log_interval, args.gradient_clip)
        val_by_source = {}
        for source_name, val_loader in zip(source_names, val_loaders):
            val_by_source[source_name] = run_epoch(
                model, val_loader, criterion, device,
                priors.size(0), None, epoch, args.log_interval)
        metric_keys = ("total", "corner", "classification",
                       "normalized_corner", "edge", "orientation")
        val_metrics = dict(
            (key, sum(item[key] for item in val_by_source.values()) /
             float(len(val_by_source))) for key in metric_keys)
        val_metrics["seconds"] = sum(
            item["seconds"] for item in val_by_source.values())
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train": train_metrics, "val": val_metrics,
               "val_by_source": val_by_source,
               "checkpoint_selection": "macro_source_val_total"}
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
