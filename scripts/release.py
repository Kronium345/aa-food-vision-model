"""Package a trained model as a versioned release for the Expo app.

    python scripts/release.py --run runs/<run> --variant fp16 --version 1 \
        [--app-repo ../oq-agile-athletes/oq-agile-athletes]

Writes exports/fridge-detector-v<N>.tflite, exports/fridge-labels-v<N>.json and
exports/fridge-anchors-v<N>.bin (+ fridge-detector-v<N>.eval.json if the model
was evaluated). The labels file carries `detector.decoding` and the .bin holds
the SSD anchors, which the app needs to decode the raw model outputs itself
(see fridgevision/ssd_decoding.py). With --app-repo it also copies all three
into <app>/assets/models/. Tag the model repo afterwards:
    git tag model-v<N> && git push --tags
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog  # noqa: E402
from fridgevision.labels_export import build_labels_json  # noqa: E402
from fridgevision.model_file import embedded_labels, labels_match, sha256  # noqa: E402
from fridgevision.ssd_decoding import read_ssd_decoding  # noqa: E402

MAX_SIZE_MB = 8.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", choices=["fp16", "int8", "fp32"], default="fp16")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--exports", type=Path, default=REPO_ROOT / "exports")
    parser.add_argument("--app-repo", type=Path, help="Expo app repo root to copy assets/models into")
    parser.add_argument("--allow-failed-eval", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite an existing release")
    args = parser.parse_args()

    catalog = load_catalog()
    model = args.run / f"model_{args.variant}.tflite"
    if not model.exists():
        print(f"ERROR: {model} not found")
        return 1

    labels = embedded_labels(model)
    if labels is None or not labels_match(labels, catalog.detector_labels):
        print("ERROR: the model's embedded labels don't match classes.yaml; refusing to release.")
        return 1

    size_mb = model.stat().st_size / 1e6
    if size_mb >= MAX_SIZE_MB:
        print(f"ERROR: model is {size_mb:.1f} MB (limit {MAX_SIZE_MB} MB). Try --variant int8.")
        return 1

    eval_path = model.with_suffix(".eval.json")
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    if evaluation is None:
        print("WARNING: no evaluation report; run scripts/evaluate.py first.")
        if not args.allow_failed_eval:
            return 1
    elif not all(evaluation["exit_criteria"].values()) and not args.allow_failed_eval:
        print(f"ERROR: exit criteria failed: {evaluation['exit_criteria']} (use --allow-failed-eval for a test build)")
        return 1

    train_report_path = args.run / "train_report.json"
    train_report = json.loads(train_report_path.read_text(encoding="utf-8")) if train_report_path.exists() else {}

    model_name = f"fridge-detector-v{args.version}.tflite"
    labels_name = f"fridge-labels-v{args.version}.json"
    anchors_name = f"fridge-anchors-v{args.version}.bin"

    decoding, input_size, anchors = read_ssd_decoding(model, anchors_name)
    if decoding["numClasses"] != len(labels):
        print(f"ERROR: model scores {decoding['numClasses']} classes but has {len(labels)} labels")
        return 1
    args.exports.mkdir(parents=True, exist_ok=True)
    dest_model = args.exports / model_name
    if dest_model.exists() and not args.force:
        print(f"ERROR: {dest_model} already exists. Releases are immutable; bump --version.")
        return 1

    model_info = {
        "releaseVersion": args.version,
        "variant": args.variant,
        "inputSize": input_size,
        "architecture": train_report.get("model"),
        "datasetVersion": train_report.get("dataset_version"),
        "sizeBytes": model.stat().st_size,
        "sha256": sha256(model),
        "releasedAt": datetime.now(timezone.utc).isoformat(),
        "testMap50": evaluation["map50"] if evaluation else None,
        "runtime": "react-native-fast-tflite; app decodes raw SSD outputs with detector.decoding + anchors",
        "decoding": decoding,
    }
    labels_json = build_labels_json(catalog, model_file=model_name, model_info=model_info)
    # The app maps score column i -> labels[i], so ship the model's own label
    # order exactly (with or without "background", as embedded in the model).
    labels_json["detector"]["labels"] = labels

    shutil.copy2(model, dest_model)
    anchors.astype("<f4").tofile(args.exports / anchors_name)
    (args.exports / labels_name).write_text(json.dumps(labels_json, indent=2, ensure_ascii=False), encoding="utf-8")
    if evaluation:
        shutil.copy2(eval_path, args.exports / f"fridge-detector-v{args.version}.eval.json")
    print(f"Released {dest_model} ({size_mb:.1f} MB), {args.exports / labels_name} "
          f"and {args.exports / anchors_name} ({len(anchors)} anchors)")

    if args.app_repo:
        models_dir = args.app_repo / "assets" / "models"
        if not (args.app_repo / "package.json").exists():
            print(f"ERROR: {args.app_repo} doesn't look like the app repo (no package.json)")
            return 1
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest_model, models_dir / model_name)
        shutil.copy2(args.exports / labels_name, models_dir / labels_name)
        shutil.copy2(args.exports / anchors_name, models_dir / anchors_name)
        print(f"Copied to {models_dir}")
        print("Then in the app: import the new labels file in lib/fridgeScan/catalog.ts and set
"
              "ACTIVE_DETECTOR = releasedDetector(require(...tflite), require(...bin)) in modelConfig.ts")
    print(f"\nNext: git tag model-v{args.version} && git push --tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
