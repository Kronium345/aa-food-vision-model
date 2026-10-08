"""Train the fridge detector with MediaPipe Model Maker and export .tflite files.

    python scripts/train.py --data data/processed/v1 --model mobilenet_multi_avg_i384 --epochs 60

Writes runs/<timestamp>-<model>/:
    model_fp32.tflite    float model
    model_fp16.tflite    float16 weights (usually the one to ship: ~half size, same accuracy)
    model_int8.tflite    only with --int8 (quantisation-aware training, slower to train)
    train_report.json    config, label order, validation COCO metrics

Linux / Colab only (Python 3.11, requirements-train.txt). Use a GPU runtime;
CPU training works but takes hours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog  # noqa: E402
from fridgevision.model_file import MODEL_INPUT_SIZES, labels_match  # noqa: E402


def _floats(metrics: dict) -> dict:
    return {str(k): float(v) for k, v in (metrics or {}).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data/processed/<version>")
    parser.add_argument("--model", choices=sorted(MODEL_INPUT_SIZES), default="mobilenet_multi_avg_i384")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--int8", action="store_true", help="also run QAT and export an int8 model")
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/cache")
    args = parser.parse_args()

    from mediapipe_model_maker import object_detector, quantization

    catalog = load_catalog()
    run_dir = args.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}"
    run_dir.mkdir(parents=True)

    dataset_version = args.data.name
    cache = args.cache_dir / dataset_version
    train_data = object_detector.Dataset.from_coco_folder(str(args.data / "train"), cache_dir=str(cache / "train"))
    val_data = object_detector.Dataset.from_coco_folder(
        str(args.data / "validation"), cache_dir=str(cache / "validation")
    )
    label_names = list(train_data.label_names)
    if not labels_match(label_names, catalog.detector_labels):
        print("ERROR: dataset labels don't match classes.yaml. Rebuild the dataset.")
        print(f"  dataset:      {label_names}")
        print(f"  classes.yaml: {catalog.detector_labels}")
        return 1
    print(f"train={train_data.size} validation={val_data.size} classes={len(label_names)}")

    hparams_kwargs = {"export_dir": str(run_dir)}
    if args.epochs:
        hparams_kwargs["epochs"] = args.epochs
    if args.batch_size:
        hparams_kwargs["batch_size"] = args.batch_size
    if args.learning_rate:
        hparams_kwargs["learning_rate"] = args.learning_rate
    options = object_detector.ObjectDetectorOptions(
        supported_model=getattr(object_detector.SupportedModels, args.model.upper()),
        hparams=object_detector.HParams(**hparams_kwargs),
    )

    started = time.time()
    model = object_detector.ObjectDetector.create(
        train_data=train_data, validation_data=val_data, options=options
    )
    loss, coco_metrics = model.evaluate(val_data, batch_size=args.batch_size or 8)
    print(f"validation loss={loss} AP50={coco_metrics.get('AP50')}")

    # Export float variants BEFORE QAT: after QAT the model can only export int8.
    model.export_model("model_fp32.tflite")
    model.export_model(
        "model_fp16.tflite", quantization_config=quantization.QuantizationConfig.for_float16()
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.data),
        "dataset_version": dataset_version,
        "classes_version": catalog.version,
        "model": args.model,
        "input_size": MODEL_INPUT_SIZES[args.model],
        "hparams": {k: v for k, v in hparams_kwargs.items() if k != "export_dir"},
        "label_names": label_names,
        "train_images": train_data.size,
        "validation_images": val_data.size,
        "validation": {"loss": [float(x) for x in loss] if isinstance(loss, (list, tuple)) else float(loss),
                       "coco": _floats(coco_metrics)},
        "exports": ["model_fp32.tflite", "model_fp16.tflite"],
    }

    if args.int8:
        qat = object_detector.QATHParams(
            learning_rate=0.3, batch_size=4, epochs=args.qat_epochs, decay_steps=6, decay_rate=0.96
        )
        model.quantization_aware_training(train_data, val_data, qat_hparams=qat)
        qat_loss, qat_metrics = model.evaluate(val_data)
        model.export_model("model_int8.tflite")
        report["validation_int8"] = {"coco": _floats(qat_metrics)}
        report["exports"].append("model_int8.tflite")
        print(f"int8 AP50={qat_metrics.get('AP50')}")

    report["train_seconds"] = round(time.time() - started)
    (run_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in report["exports"]:
        size = (run_dir / name).stat().st_size / 1e6
        print(f"  {name}: {size:.1f} MB")
    print(f"\nRun saved to {run_dir}\nNext: python scripts/evaluate.py --model {run_dir / 'model_fp16.tflite'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
