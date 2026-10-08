"""Merge COCO-format sources into the layout MediaPipe Model Maker trains on.

Output (per split):
    <out>/<split>/data/<images>
    <out>/<split>/labels.json      # COCO; category ids 1..N in classes.yaml order

Every split lists ALL detector categories (even with zero boxes) so the label map
is identical across splits and matches fridge-labels-vN.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fridgevision.classes import ClassCatalog, resolve_label

SPLITS = ("train", "validation", "test")
MIN_BOX_PX = 2.0


class DatasetError(ValueError):
    pass


@dataclass
class Source:
    name: str
    path: Path
    licence: str
    role: str = "train"  # train -> train/validation split; test -> held-out test only
    annotations: str | None = None  # labels file relative to path (auto-detected if None)
    images: str | None = None  # images dir relative to path (auto-detected if None)
    label_map: dict[str, str] = field(default_factory=dict)
    url: str | None = None


@dataclass
class DatasetConfig:
    allowed_licences: tuple[str, ...]
    validation_fraction: float
    sources: list[Source]


def load_dataset_config(path: Path, base_dir: Path | None = None) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = base_dir or Path(path).resolve().parent
    allowed = tuple(str(x) for x in raw.get("allowed_licences") or [])
    sources = []
    for s in raw.get("sources") or []:
        if s.get("enabled", True) is False:
            continue
        role = s.get("role", "train")
        if role not in ("train", "test"):
            raise DatasetError(f"source '{s.get('name')}': role must be train or test")
        sources.append(
            Source(
                name=str(s["name"]),
                path=(base / s["path"]).resolve(),
                licence=str(s.get("licence", "")),
                role=role,
                annotations=s.get("annotations"),
                images=s.get("images"),
                label_map={str(k): (v or "") for k, v in (s.get("label_map") or {}).items()},
                url=s.get("url"),
            )
        )
    return DatasetConfig(
        allowed_licences=allowed,
        validation_fraction=float(raw.get("validation_fraction", 0.1)),
        sources=sources,
    )


def check_licences(config: DatasetConfig) -> None:
    bad = [s for s in config.sources if s.licence not in config.allowed_licences]
    if bad:
        lines = [f"{s.name}: '{s.licence or '<missing>'}'" for s in bad]
        raise DatasetError(
            "Sources with licences not in allowed_licences (check the dataset page, "
            "then add the licence to the allowlist only if it permits commercial use):\n  "
            + "\n  ".join(lines)
        )


def _find_coco(source: Source) -> tuple[Path, Path]:
    """Locate (labels json, images dir) for FiftyOne / Roboflow / CVAT / Label Studio exports."""
    root = source.path
    if source.annotations:
        ann = root / source.annotations
    else:
        candidates = [
            root / "labels.json",  # FiftyOne COCODetectionDataset
            root / "_annotations.coco.json",  # Roboflow
            root / "annotations" / "instances_default.json",  # CVAT
            root / "result.json",  # Label Studio
        ]
        ann = next((c for c in candidates if c.exists()), None)
        if ann is None:
            raise DatasetError(
                f"source '{source.name}': no COCO labels file in {root} "
                "(set `annotations:` in datasets.yaml)"
            )
    if source.images:
        images = root / source.images
    else:
        images = next(
            (d for d in (root / "data", root / "images", root) if d.is_dir()), root
        )
    if not ann.exists():
        raise DatasetError(f"source '{source.name}': {ann} not found")
    return ann, images


def _split_for(key: str, validation_fraction: float) -> str:
    """Deterministic train/validation assignment that survives re-runs and new sources."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if h < validation_fraction else "train"


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _clean_box(bbox, width: int, height: int) -> list[float] | None:
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def build_dataset(
    catalog: ClassCatalog,
    config: DatasetConfig,
    out_dir: Path,
    *,
    keep_empty: bool = False,
    link: bool = False,
) -> dict:
    """Merge all sources into out_dir. Returns stats (also written to out_dir/stats.json)."""
    check_licences(config)
    if not config.sources:
        raise DatasetError("datasets.yaml has no enabled sources")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DatasetError(f"{out_dir} is not empty; pick a new version or delete it")

    visual = catalog.visual_classes
    cat_id = {c.name: i + 1 for i, c in enumerate(visual)}
    categories = [{"id": cat_id[c.name], "name": c.name, "supercategory": c.category} for c in visual]

    splits: dict[str, dict] = {s: {"images": [], "annotations": []} for s in SPLITS}
    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    dropped: dict[str, Counter] = defaultdict(Counter)
    seen_hashes: dict[str, str] = {}
    duplicates = 0
    per_source: dict[str, dict] = {}

    for split in SPLITS:
        (out_dir / split / "data").mkdir(parents=True, exist_ok=True)

    for source in config.sources:
        ann_path, images_dir = _find_coco(source)
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        resolver = catalog.label_resolver(source.label_map)
        src_cat = {c["id"]: c["name"] for c in coco.get("categories", [])}
        anns_by_image: dict = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_image[a["image_id"]].append(a)

        kept_images = kept_boxes = 0
        for img in coco.get("images", []):
            file_name = img["file_name"]
            src_file = images_dir / file_name
            if not src_file.exists():
                src_file = images_dir / Path(file_name).name
            if not src_file.exists():
                dropped[source.name]["<missing image>"] += 1
                continue

            digest = _file_hash(src_file)
            if digest in seen_hashes:
                duplicates += 1
                continue

            width, height = img.get("width"), img.get("height")
            if not width or not height:
                width, height = _image_size(src_file)

            boxes = []
            for a in anns_by_image.get(img["id"], []):
                raw_label = src_cat.get(a["category_id"], str(a["category_id"]))
                target = resolve_label(resolver, raw_label)
                if target is None:
                    dropped[source.name][raw_label] += 1
                    continue
                box = _clean_box(a["bbox"], width, height)
                if box is None:
                    dropped[source.name]["<degenerate box>"] += 1
                    continue
                boxes.append((target, box))

            if not boxes and not keep_empty:
                continue
            seen_hashes[digest] = source.name

            split = "test" if source.role == "test" else _split_for(
                f"{source.name}/{file_name}", config.validation_fraction
            )
            out_name = f"{source.name}__{digest[:12]}{src_file.suffix.lower()}"
            dest = out_dir / split / "data" / out_name
            if link:
                try:
                    dest.hardlink_to(src_file)
                except OSError:
                    shutil.copy2(src_file, dest)
            else:
                shutil.copy2(src_file, dest)

            image_id = len(splits[split]["images"]) + 1
            splits[split]["images"].append(
                {"id": image_id, "file_name": out_name, "width": width, "height": height}
            )
            for target, box in boxes:
                splits[split]["annotations"].append(
                    {
                        "id": len(splits[split]["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": cat_id[target],
                        "bbox": box,
                        "area": round(box[2] * box[3], 2),
                        "iscrowd": 0,
                    }
                )
                counts[split][target] += 1
            kept_images += 1
            kept_boxes += len(boxes)

        per_source[source.name] = {
            "licence": source.licence,
            "role": source.role,
            "url": source.url,
            "images": kept_images,
            "boxes": kept_boxes,
            "dropped": dict(dropped[source.name].most_common()),
        }

    for split in SPLITS:
        payload = {"categories": categories, **splits[split]}
        (out_dir / split / "labels.json").write_text(json.dumps(payload), encoding="utf-8")

    stats = {
        "classes_version": catalog.version,
        "splits": {
            s: {
                "images": len(splits[s]["images"]),
                "boxes": len(splits[s]["annotations"]),
                "per_class": {c.name: counts[s][c.name] for c in visual},
            }
            for s in SPLITS
        },
        "duplicates_skipped": duplicates,
        "sources": per_source,
        "warnings": dataset_warnings(catalog, counts),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def flatten_coco_folders(folders: list[tuple[Path, Path]], out_dir: Path) -> dict:
    """Combine several COCO exports (labels json, images dir) into out_dir/labels.json + out_dir/data.

    Used for Roboflow downloads (train/valid/test folders) and multi-task CVAT
    exports. Categories are merged by name; image ids are renumbered.
    """
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    categories: dict[str, int] = {}
    images, annotations = [], []
    for ann_path, images_dir in folders:
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        cat_remap = {}
        for c in coco.get("categories", []):
            cat_remap[c["id"]] = categories.setdefault(c["name"], len(categories) + 1)
        img_remap = {}
        prefix = ann_path.parent.name
        for img in coco.get("images", []):
            src = images_dir / img["file_name"]
            if not src.exists():
                continue
            new_name = f"{prefix}__{Path(img['file_name']).name}"
            shutil.copy2(src, out_dir / "data" / new_name)
            img_remap[img["id"]] = len(images) + 1
            images.append({**img, "id": img_remap[img["id"]], "file_name": new_name})
        for a in coco.get("annotations", []):
            if a["image_id"] not in img_remap:
                continue
            annotations.append(
                {
                    **a,
                    "id": len(annotations) + 1,
                    "image_id": img_remap[a["image_id"]],
                    "category_id": cat_remap[a["category_id"]],
                }
            )
    payload = {
        "categories": [{"id": i, "name": n} for n, i in categories.items()],
        "images": images,
        "annotations": annotations,
    }
    (out_dir / "labels.json").write_text(json.dumps(payload), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations), "categories": list(categories)}


def dataset_warnings(catalog: ClassCatalog, counts: dict[str, Counter], min_train: int = 50) -> list[str]:
    warnings = []
    for c in catalog.visual_classes:
        n = counts["train"][c.name]
        if n == 0:
            warnings.append(f"{c.name}: no training boxes (the model cannot learn it)")
        elif n < min_train:
            warnings.append(f"{c.name}: only {n} training boxes (aim for 50-150 real-fridge images)")
    if sum(counts["test"].values()) == 0:
        warnings.append(
            "test split is empty: add a held-out set of your own real-fridge photos "
            "(role: test) or the exit criteria can't be checked"
        )
    return warnings
