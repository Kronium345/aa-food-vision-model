"""Evaluate an exported .tflite on the held-out test split (real-fridge photos).

    python scripts/evaluate.py --model runs/<run>/model_fp16.tflite --data data/processed/v1

Runs the model through MediaPipe Tasks' ObjectDetector (the same decoding + NMS
the app gets from MediaPipe on-device) and checks the Phase 1 exit criteria:
mAP@0.5 >= 0.6 and model size < 8 MB. Writes <model>.eval.json next to the model.

Latency here is measured on THIS machine's CPU; it's only useful for comparing
runs. Measure real latency on an iPhone 12 / mid-range Android in the dev build.
"""

from __future__ import annotations

import os

# MediaPipe imports matplotlib at start-up. Colab exports
# MPLBACKEND=module://matplotlib_inline..., which this separate Python 3.11
# environment doesn't have, so importing MediaPipe would crash. Nothing here
# draws, so use the headless backend.
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import load_catalog  # noqa: E402
from fridgevision.metrics import Detection, GroundTruth, evaluate_detections  # noqa: E402
from fridgevision.model_file import embedded_labels, labels_match  # noqa: E402

MAP_TARGET = 0.6
MAX_SIZE_MB = 8.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="data/processed/<version>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="for precision/recall")
    parser.add_argument("--strict", action="store_true", help="exit 1 if exit criteria fail")
    args = parser.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    catalog = load_catalog()
    labels = embedded_labels(args.model)
    if labels is None or not labels_match(labels, catalog.detector_labels):
        print("ERROR: model has no embedded labels or they don't match classes.yaml")
        return 1

    split_dir = args.data / args.split
    coco = json.loads((split_dir / "labels.json").read_text(encoding="utf-8"))
    if not coco["images"]:
        print(f"ERROR: {split_dir} has no images. Add a role: test source to datasets.yaml.")
        return 1
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    ground_truth = [
        GroundTruth(str(a["image_id"]), cat_name[a["category_id"]], tuple(a["bbox"]))
        for a in coco["annotations"]
    ]

    detector = vision.ObjectDetector.create_from_options(
        vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=0.05,  # low, so the precision/recall curve is complete
            max_results=100,
        )
    )

    detections: list[Detection] = []
    timings_ms: list[float] = []
    for img in coco["images"]:
        image = mp.Image.create_from_file(str(split_dir / "images" / img["file_name"]))
        t0 = time.perf_counter()
        result = detector.detect(image)
        timings_ms.append((time.perf_counter() - t0) * 1000)
        for d in result.detections:
            cat = d.categories[0]
            b = d.bounding_box
            detections.append(
                Detection(str(img["id"]), cat.category_name, float(cat.score),
                          (float(b.origin_x), float(b.origin_y), float(b.width), float(b.height)))
            )

    metrics = evaluate_detections(ground_truth, detections, iou_threshold=0.5,
                                  score_threshold=args.score_threshold)
    size_mb = args.model.stat().st_size / 1e6
    timings_ms.sort()
    report = {
        "model": str(args.model),
        "data": str(split_dir),
        "images": len(coco["images"]),
        "size_mb": round(size_mb, 2),
        "host_latency_ms": {
            "mean": round(statistics.mean(timings_ms), 1),
            "p95": round(timings_ms[int(0.95 * (len(timings_ms) - 1))], 1),
        },
        "map50": round(metrics["map"], 4),
        "exit_criteria": {
            "map50>=0.6": metrics["map"] >= MAP_TARGET,
            "size<8MB": size_mb < MAX_SIZE_MB,
        },
        "metrics": metrics,
    }
    out = args.model.with_suffix(".eval.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'class':<18}{'gt':>5}{'AP50':>8}{'P':>7}{'R':>7}")
    for name, m in sorted(metrics["per_class"].items(), key=lambda kv: (kv[1]["ap"] is None, kv[1]["ap"] or 0)):
        fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"  # noqa: E731
        print(f"{name:<18}{m['gt']:>5}  {fmt(m['ap'])} {fmt(m['precision'])} {fmt(m['recall'])}")
    print(f"\nmAP@0.5 = {metrics['map']:.3f} over {metrics['classes_evaluated']} classes "
          f"(target {MAP_TARGET})")
    print(f"size = {size_mb:.1f} MB (limit {MAX_SIZE_MB})")
    print(f"host CPU latency: mean {report['host_latency_ms']['mean']} ms, p95 {report['host_latency_ms']['p95']} ms")
    print(f"Report: {out}")

    passed = all(report["exit_criteria"].values())
    print("EXIT CRITERIA: " + ("PASS" if passed else "FAIL"))
    return 1 if (args.strict and not passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
