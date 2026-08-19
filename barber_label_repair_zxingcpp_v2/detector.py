from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .geometry import FIELD_NAMES


@dataclass(frozen=True)
class SafeResult:
    valid: bool
    error_type: str | None
    position: np.ndarray
    orientation: int
    payload_sha256: str | None
    mirrored: bool | None


def preprocess(image: Image.Image, family: str) -> Image.Image:
    if family == "rgb":
        return image.convert("RGB")
    g = image.convert("L")
    if family == "gray":
        return g
    if family.startswith("gamma_"):
        gamma = float(family.split("_", 1)[1])
        lut = [int(round(255 * ((i / 255.0) ** gamma))) for i in range(256)]
        return g.point(lut)
    if family == "unsharp":
        return g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
    if family == "invert":
        return ImageOps.invert(g)
    if family == "otsu":
        arr = np.asarray(g, dtype=np.uint8)
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = arr.size
        sum_total = float(np.dot(np.arange(256), hist))
        weight_b = sum_b = 0.0
        best_var = -1.0
        threshold = 0
        for t in range(256):
            weight_b += hist[t]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            sum_b += t * hist[t]
            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f
            between = weight_b * weight_f * (mean_b - mean_f) ** 2
            if between > best_var:
                best_var, threshold = between, t
        return Image.fromarray(np.where(arr > threshold, 255, 0).astype(np.uint8), "L")
    if family == "adaptive":
        arr = np.asarray(g, dtype=np.float64)
        radius = 12
        pad = np.pad(arr, radius, mode="reflect")
        integ = np.pad(pad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        k = 2 * radius + 1
        sums = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
        mean = sums / (k * k)
        return Image.fromarray(np.where(arr > mean - 7, 255, 0).astype(np.uint8), "L")
    raise ValueError(f"unknown preprocessing family: {family}")


class ZXingSafeDetector:
    """PIL-only detector that never exposes decoded content."""

    def __init__(self):
        import zxingcpp
        self.z = zxingcpp

    def _binarizer(self, name: str):
        try:
            return getattr(self.z.Binarizer, name)
        except AttributeError as exc:
            raise RuntimeError(f"zxing-cpp lacks Binarizer.{name}") from exc

    def read(self, image: Image.Image, binarizer: str, return_errors: bool) -> list[SafeResult]:
        raw = self.z.read_barcodes(
            image, formats=self.z.BarcodeFormat.QRCode, try_rotate=True,
            try_downscale=True, try_invert=True, binarizer=self._binarizer(binarizer),
            return_errors=return_errors)
        out: list[SafeResult] = []
        for r in raw:
            if r.format != self.z.BarcodeFormat.QRCode:
                continue
            pos = r.position
            points = np.array([[getattr(pos, name).x, getattr(pos, name).y]
                               for name in FIELD_NAMES], dtype=np.float64)
            if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4:
                continue
            err = getattr(r, "error", None)
            err_type = None if err is None else str(getattr(err, "type", err))
            payload = bytes(r.bytes) if bool(r.valid) else b""
            payload_hash = hashlib.sha256(payload).hexdigest() if payload else None
            mirrored = getattr(r, "is_mirrored", None)
            if callable(mirrored):
                mirrored = bool(mirrored())
            elif mirrored is not None:
                mirrored = bool(mirrored)
            out.append(SafeResult(bool(r.valid), err_type, points,
                                  int(getattr(r, "orientation", 0)),
                                  payload_hash, mirrored))
        return out

    def negative_control(self) -> None:
        blank = Image.new("RGB", (240, 320), "white")
        if self.read(blank, "LocalAverage", False):
            raise RuntimeError("blank-image negative control produced a ZXing result")
        if self.read(blank, "LocalAverage", True):
            raise RuntimeError("blank-image return_errors control produced a ZXing result")
