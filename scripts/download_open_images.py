"""Pull Open Images V7 boxes for our visual classes and export them as COCO.

    python scripts/download_open_images.py --per-class 800
    -> data/raw/openimages/{data/, labels.json}

Needs requirements-data.txt (FiftyOne). Per split it picks up to --per-class
random images for each class straight from the split's box-label CSV (read once,
in chunks), then loads that union of images through FiftyOne with labels for ALL
our classes, so an image picked for "apple" still keeps its banana boxes
(otherwise the banana would be trained as background).

Asking FiftyOne for each class separately re-parses the ~2 GB train CSV per
class (28x), which is where most of the time used to go.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog  # noqa: E402


def _labels_csv(dataset_dir: Path, split: str) -> Path:
    for candidate in (dataset_dir / split / "labels" / "detections.csv", dataset_dir / "labels" / "detections.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no detections.csv for split '{split}' under {dataset_dir}")


def _classes_csv(labels_csv: Path) -> Path:
    return labels_csv.parent.parent / "metadata" / "classes.csv"


def select_image_ids(
    labels_csv: Path, classes_csv: Path, class_names: list[str], per_class: int, seed: int = 51
) -> dict[str, list[str]]:
    """Up to `per_class` random image ids per Open Images class, from one pass over the CSV."""
    import random

    import pandas as pd

    mids = pd.read_csv(classes_csv, header=None, names=["mid", "name"])
    mid_to_name = {m: n for m, n in zip(mids["mid"], mids["name"]) if n in set(class_names)}

    images: dict[str, set[str]] = {name: set() for name in class_names}
    for chunk in pd.read_csv(labels_csv, usecols=["ImageID", "LabelName"], chunksize=2_000_000):
        chunk = chunk[chunk["LabelName"].isin(mid_to_name.keys())]
        for mid, ids in chunk.groupby("LabelName")["ImageID"]:
            images[mid_to_name[mid]].update(ids)

    rng = random.Random(seed)
    picked: dict[str, list[str]] = {}
    for name in class_names:
        ids = sorted(images[name])
        rng.shuffle(ids)
        picked[name] = ids[:per_class]
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/raw/openimages")
    parser.add_argument("--per-class", type=int, default=800, help="max images per class per split")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--classes", nargs="*", help="only these of our class names (default: all mapped)")
    parser.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    args = parser.parse_args()

    import fiftyone as fo
    import fiftyone.utils.openimages as fouo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    catalog = load_catalog()
    available = set(fouo.get_classes(version="v7"))

    oi_to_ours: dict[str, str] = {}
    for c in catalog.visual_classes:
        if args.classes and c.name not in args.classes:
            continue
        for oi_name in c.open_images:
            if oi_name in available:
                oi_to_ours[oi_name] = c.name
            else:
                print(f"WARNING: '{oi_name}' (for {c.name}) is not an Open Images V7 class; skipping")
    if not oi_to_ours:
        print("No Open Images classes to download.")
        return 1
    print(f"Open Images classes: {sorted(oi_to_ours)}")

    ours = sorted(set(oi_to_ours.values()))
    merged = fo.Dataset("fridge-openimages-merged", overwrite=True)

    for split in args.splits:
        # Downloads the split's label CSVs (once; cached afterwards) plus one image.
        _, dataset_dir = foz.download_zoo_dataset(
            "open-images-v7", split=split, label_types=["detections"], max_samples=1
        )
        labels_csv = _labels_csv(Path(dataset_dir), split)
        picked = select_image_ids(labels_csv, _classes_csv(labels_csv), sorted(oi_to_ours), args.per_class)
        image_ids: set[str] = set()
        for oi_name, ids in picked.items():
            print(f"[{split}] {oi_name:<18} {len(ids)} images")
            image_ids.update(ids)
        print(f"[{split}] {len(image_ids)} unique images")
        if not image_ids:
            continue

        full = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=sorted(oi_to_ours),
            image_ids=sorted(image_ids),
            only_matching=True,
            num_workers=args.workers,
            dataset_name=f"fridge-oi-{split}",
            # Replace the FiftyOne dataset entry but keep the downloaded files;
            # overwrite=True would re-download the split (incl. a ~1 GB CSV) each call.
            drop_existing_dataset=True,
        )
        full = full.map_labels("ground_truth", oi_to_ours)
        view = full.filter_labels("ground_truth", F("label").is_in(ours), only_matches=True)
        merged.add_collection(view)

    args.out.mkdir(parents=True, exist_ok=True)
    merged.export(
        export_dir=str(args.out),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        classes=ours,
        overwrite=True,
    )
    counts = merged.count_values("ground_truth.detections.label")
    print(f"Exported {len(merged)} images to {args.out}")
    for name in ours:
        print(f"  {name:<16} {counts.get(name, 0)} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
