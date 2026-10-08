"""Inspect exported .tflite files without TensorFlow."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

# MediaPipe Model Maker input sizes per supported model.
MODEL_INPUT_SIZES = {
    "mobilenet_v2": 256,
    "mobilenet_v2_i320": 320,
    "mobilenet_multi_avg": 256,
    "mobilenet_multi_avg_i384": 384,
}


def embedded_labels(model_path: Path) -> list[str] | None:
    """Labels packed into a .tflite with metadata (the file is also a zip archive)."""
    if not zipfile.is_zipfile(model_path):
        return None
    with zipfile.ZipFile(model_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        preferred = [n for n in names if "label" in n.lower()] or names
        if not preferred:
            return None
        text = zf.read(preferred[0]).decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def strip_background(labels: list[str]) -> list[str]:
    return labels[1:] if labels and labels[0].lower() in ("background", "???") else labels


def labels_match(model_labels: list[str], expected: list[str]) -> bool:
    return strip_background(model_labels) == strip_background(expected)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
