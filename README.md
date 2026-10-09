# oq-food-vision-model

Training pipeline for the Food Tracker's **live fridge scanning**. It builds an
on-device ingredient detector plus the label / OCR dictionary the Expo app ships
alongside it.

**This repo is never deployed.** Detection and OCR run on the phone, and frames
never leave the device. The only output is three files per release, which get
copied into the app repo's `assets/models/`:

```
exports/fridge-detector-vN.tflite   # MediaPipe object detector with metadata (labels, anchors)
exports/fridge-labels-vN.json       # label order, decoding params, display names, categories, OCR dictionary
exports/fridge-anchors-vN.bin       # SSD anchors (float32 LE: x_center, y_center, w, h per anchor)
```

There's no Render service and no URL to give the app. The only server-side part
of the feature is the optional text-only `POST /pantry/meal-ideas` route, and
that belongs in the existing API (`api-oq-agile-athletes`), not here. The
`Dockerfile` in this repo is a reproducible Python 3.11 **tooling** image for
running the pipeline on Windows.

## Layout

```
classes.yaml        source of truth: detector classes, pantry categories, OCR keywords (EN/FR/ES)
datasets.yaml       dataset sources + licence allowlist + held-out test set
fridgevision/       library code (validation, dataset merge, mAP, OCR matcher, label export, SSD decoding export)
scripts/
  export_labels.py        validate classes.yaml, write a dev labels file (no model needed)
  download_open_images.py pull Open Images V7 boxes via FiftyOne
  import_roboflow.py      pull a Roboflow Universe dataset, show how its labels map
  build_dataset.py        merge sources -> data/processed/vN (Model Maker COCO layout)
  train.py                MediaPipe Model Maker training -> fp32 / fp16 / int8 .tflite
  evaluate.py             mAP@0.5 on real-fridge test photos + Phase 1 exit criteria
  colab_preflight.py      Colab: repair training venv, migrate old datasets, check imports
  release.py              checks + versioned export: model, labels, anchors (+ optional copy into the app repo)
notebooks/train_colab.ipynb   GPU training on Colab
tests/              pytest suite for everything that doesn't need TensorFlow
```

## Environments

| What | Where | Python |
|---|---|---|
| classes, dataset build, tests | anywhere (Windows fine) | any 3.10+ (`pip install -r requirements-dev.txt`) |
| Open Images / Roboflow download | anywhere, or `docker --target data` | 3.10+ (`requirements-data.txt`) |
| training + evaluation | **Colab (GPU)** or `docker --target train` | **3.11 only** (`requirements-train.lock`) |

Model Maker needs TensorFlow 2.15, which only supports Python 3.9–3.11 and has
no Windows wheels for `tensorflow-text`. Don't install it into the system Python
3.14. Colab's default Python is 3.12, so the notebook creates a 3.11 venv with `uv`.

### Docker (Windows, local)

Start Docker Desktop first.

```powershell
docker build --target test -t fridge-vision:test .
docker run --rm fridge-vision:test                       # runs pytest

docker build --target train -t fridge-vision:train .
docker run --rm -it -v "${PWD}:/workspace" fridge-vision:train `
  python scripts/train.py --data data/processed/v1 --epochs 60
```

CPU training works but takes hours. With an NVIDIA GPU + WSL2, build with
`--build-arg GPU=true` and run with `--gpus all`. Otherwise use Colab.

## Workflow

```bash
# 0. Check classes.yaml (run after every edit)
python scripts/export_labels.py --out exports/fridge-labels-dev.json

# 1. Data
python scripts/download_open_images.py --per-class 800          # ~28 classes exist in Open Images
python scripts/import_roboflow.py --workspace W --project P --version N --name rf_x
#    + your own fridge photos labelled in CVAT / Label Studio (COCO export) under data/raw/own/
#    then enable each source in datasets.yaml, with its licence copied from the dataset page

# 2. Merge (refuses non-allowlisted licences; prints per-class counts + gaps)
python scripts/build_dataset.py --version v1

# 2b. Colab only, before training: repair the training venv + check imports (ends with READY)
python scripts/colab_preflight.py

# 3. Train (Colab / docker train)
python scripts/train.py --data data/processed/v1 --model mobilenet_multi_avg_i384 --epochs 60

# 4. Evaluate on held-out real-fridge photos
python scripts/evaluate.py --model runs/<run>/model_fp16.tflite --data data/processed/v1

# 5. Release + copy into the app
python scripts/release.py --run runs/<run> --variant fp16 --version 1 \
    --app-repo ../oq-agile-athletes/oq-agile-athletes
git tag model-v1 && git push --tags
```

### Getting to shippable accuracy

`export_labels.py` lists the 22 detector classes that have **no Open Images
source** (lime, avocado, onion, garlic, lettuce, butter…). They need Roboflow data
or your own photos, or the model can't learn them. `build_dataset.py` warns about
any class with fewer than 50 training boxes.

- 50–150 labelled images per class taken **inside real fridges**: bad light, items
  partly hidden, at an angle, on door shelves.
- Keep a separate batch of real-fridge photos as `role: test` in `datasets.yaml`.
  Never train on it. The Phase 1 exit criterion (**mAP@0.5 ≥ 0.6, model < 8 MB**)
  is measured on that set.
- Label names in your tool can be anything listed in `classes.yaml` (name, alias
  or Open Images name). Anything else goes in the source's `label_map`, or it
  gets dropped (the build reports what it dropped).
- Model choice: `mobilenet_multi_avg_i384` (384 px) is the default because fridge
  items are small in frame. `mobilenet_v2_i320` is faster if on-device latency
  is over budget.
- Licences: Open Images boxes are CC BY 4.0 (the images are listed as CC BY 2.0).
  Check every Roboflow dataset. **No AGPL (Ultralytics YOLO) weights** without an
  Enterprise licence. The build refuses any licence that isn't in `allowed_licences`.

### Changing classes

Detector indices follow the order of `visual`/`both` entries in `classes.yaml`.
**Never reorder or remove a released class.** Append new ones and bump `version`.
`train.py`, `evaluate.py` and `release.py` all check that the model's embedded
labels match `classes.yaml` exactly, and refuse to continue if they don't.

## App integration contract

**Runtime.** Model Maker exports the raw SSD outputs (box encodings + scores for
every anchor). MediaPipe Tasks normally decodes them using the DETECTOR_METADATA
block packed into the model. The app (Phase 0 chose the
**`react-native-fast-tflite` route**) runs the `.tflite` directly and decodes in a
worklet (`lib/fridgeScan/decode.ts → decodeMediaPipeSsd`), so `release.py`:

- reads the decoding options + anchors with `fridgevision/ssd_decoding.py`,
- writes them to `detector.decoding` in the labels file and to `fridge-anchors-vN.bin`,
- ships `detector.labels` in the model's own score-column order.

The app's decoder reproduces MediaPipe Tasks' `ObjectDetector` output (boxes and
scores within 1%) on MediaPipe's own `efficientdet_lite0` export, and
`evaluate.py` uses MediaPipe Tasks, so its numbers carry over. Decoding cost scales
with anchors × classes: `mobilenet_v2_i320` (19,206 anchors) is cheaper per frame
than `mobilenet_multi_avg_i384` (more anchors), which is worth checking on a
mid-range Android in the dev build.

**`fridge-labels-vN.json`:**

```jsonc
{
  "schemaVersion": 1,
  "classesVersion": 1,
  "detector": {
    "modelFile": "fridge-detector-v1.tflite",
    "labels": ["background", "egg", "milk", ...],   // index = model score column (model's own order)
    "inputSize": 384, "variant": "fp16", "sha256": "...", "testMap50": 0.63,
    "decoding": {
      "format": "mediapipe-ssd", "anchorsFile": "fridge-anchors-v1.bin",
      "numBoxes": 27621, "numClasses": 51, "boxOrder": "yxhw",
      "xScale": 1, "yScale": 1, "wScale": 1, "hScale": 1,
      "applyExponentialOnBoxSize": true, "sigmoidScore": false,
      "scoresOutput": 0, "boxesOutput": 1,
      "input": { "dtype": "float32", "mean": 127.5, "std": 127.5 }
    }
  },
  "categories": [{ "id": "dairy_eggs", "display": "Dairy & eggs" }, ...],
  "classes": { "yogurt": { "display": "Yogurt", "category": "dairy_eggs", "detect": "ocr" }, ... },
  "ocr": {
    "normalize": { ... },                       // lowercase, œ->oe etc, strip accents, [^a-z0-9]+ -> " "
    "keywords": [["semi skimmed", "milk"], ["milk chocolate", ""], ...]   // longest first; "" = ignore
  }
}
```

The app reads everything from this file: `lib/fridgeScan/catalog.ts` imports it,
`labelDictionary.ts` is a port of `fridgevision/ocr_match.py` driven by
`ocr.keywords` (it passes every case in `tests/test_text_and_ocr.py`), and
`ingredients.ts` builds chips and pantry grouping from `classes` / `categories`
(only emoji are app-side). Until a model is released, the app ships
`fridge-labels-dev.json` from `export_labels.py`; re-copy it after editing
`classes.yaml`.

**Switching the app to a release:** run `release.py ... --app-repo <app>`, then in
the app import `fridge-labels-vN.json` in `lib/fridgeScan/catalog.ts` and set
`ACTIVE_DETECTOR = releasedDetector(require('…/fridge-detector-vN.tflite'), require('…/fridge-anchors-vN.bin'))`
in `lib/fridgeScan/modelConfig.ts`. The model and anchors are assets, so a new
model can ship with an EAS Update (no store build) once the dev build with the
camera code is out.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
