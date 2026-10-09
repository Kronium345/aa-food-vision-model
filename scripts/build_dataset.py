"""Merge every enabled source in datasets.yaml into a versioned training dataset.

    python scripts/build_dataset.py --version v1
    -> data/processed/v1/{train,validation,test}/{images/, labels.json} + stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog  # noqa: E402
from fridgevision.dataset import DatasetError, build_dataset, load_dataset_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="dataset version, e.g. v1")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "datasets.yaml")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--keep-empty", action="store_true", help="keep images with no boxes")
    parser.add_argument("--link", action="store_true", help="hardlink images instead of copying")
    args = parser.parse_args()

    catalog = load_catalog()
    try:
        stats = build_dataset(
            catalog,
            load_dataset_config(args.config),
            args.out_root / args.version,
            keep_empty=args.keep_empty,
            link=args.link,
        )
    except DatasetError as e:
        print(f"ERROR: {e}")
        return 1

    for split, s in stats["splits"].items():
        print(f"{split:<11} {s['images']:>6} images {s['boxes']:>7} boxes")
    print(f"duplicates skipped: {stats['duplicates_skipped']}")
    for name, src in stats["sources"].items():
        top = ", ".join(f"{k} x{v}" for k, v in list(src["dropped"].items())[:5])
        print(f"  {name}: {src['images']} images, {src['boxes']} boxes" + (f" (dropped: {top})" if top else ""))
    for w in stats["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nWrote {args.out_root / args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
