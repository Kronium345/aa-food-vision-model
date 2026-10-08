"""Validate classes.yaml and write the app label file without a trained model.

    python scripts/export_labels.py                     # validate + summary
    python scripts/export_labels.py --out exports/fridge-labels-dev.json

The dev file lets the app start on the OCR dictionary, chips and pantry
categories before a model exists (detector.modelFile is null).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import ClassesError, load_catalog  # noqa: E402
from fridgevision.labels_export import build_labels_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        catalog = load_catalog()
    except ClassesError as e:
        print(e)
        return 1

    modes = Counter(c.detect for c in catalog.classes)
    labels = build_labels_json(catalog)
    print(
        f"classes.yaml v{catalog.version}: {len(catalog.classes)} ingredients "
        f"({modes['visual']} visual, {modes['both']} visual+OCR, {modes['ocr']} OCR-only), "
        f"{len(catalog.visual_classes)} detector classes, {len(labels['ocr']['keywords'])} OCR phrases"
    )
    no_source = [c.name for c in catalog.visual_classes if not c.open_images]
    print(f"Detector classes with no Open Images source (need Roboflow/own photos): {', '.join(no_source)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
