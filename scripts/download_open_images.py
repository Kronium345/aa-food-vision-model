"""Pull Open Images V7 boxes for our visual classes and export them as COCO.

    python scripts/download_open_images.py --per-class 800
    -> data/raw/openimages/{data/, labels.json}

Needs requirements-data.txt (FiftyOne). Runs two passes per split: first it finds
up to --per-class images for each class, then it reloads that union of images
with labels for ALL our classes, so an image picked for "apple" still keeps its
banana boxes (otherwise the banana would be trained as background).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog  # noqa: E402


def _open_images_ids(ds) -> list[str]:
    """Open Images ids of the samples in a zoo dataset.

    Some FiftyOne versions store them in an `open_images_id` field, others don't,
    but the zoo always saves each image as `<image id>.jpg`, so fall back to that.
    """
    if ds.has_field("open_images_id"):
        return [i for i in ds.values("open_images_id") if i]
    return [Path(p).stem for p in ds.values("filepath")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/raw/openimages")
    parser.add_argument("--per-class", type=int, default=800, help="max images per class per split")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--classes", nargs="*", help="only these of our class names (default: all mapped)")
    parser.add_argument("--workers", type=int, default=8)
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
        image_ids: set[str] = set()
        for oi_name in sorted(oi_to_ours):
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[oi_name],
                max_samples=args.per_class,
                shuffle=True,
                seed=51,
                num_workers=args.workers,
                dataset_name=f"fridge-oi-{split}-{oi_name}",
                drop_existing_dataset=True,
            )
            image_ids.update(_open_images_ids(ds))
            ds.delete()
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
