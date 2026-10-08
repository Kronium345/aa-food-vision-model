"""Download a Roboflow Universe dataset version as COCO and flatten it into one folder.

    set ROBOFLOW_API_KEY=...        (PowerShell: $env:ROBOFLOW_API_KEY="...")
    python scripts/import_roboflow.py --workspace acme --project fridge-items --version 3 \
        --name rf_fridge_items

Output: data/raw/roboflow/<name>/{data/, labels.json}. Then add a source entry to
datasets.yaml, copying the licence exactly as the dataset page shows it; the
dataset build refuses licences that aren't on the allowlist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import REPO_ROOT, load_catalog, resolve_label  # noqa: E402
from fridgevision.dataset import flatten_coco_folders  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--name", required=True, help="source name used in datasets.yaml")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/raw/roboflow")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set")
        return 1

    from roboflow import Roboflow

    out_dir = args.out_root / args.name
    if out_dir.exists():
        print(f"{out_dir} already exists; delete it or choose another --name")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        rf = Roboflow(api_key=api_key)
        version = rf.workspace(args.workspace).project(args.project).version(args.version)
        version.download("coco", location=tmp, overwrite=True)
        folders = [
            (ann, ann.parent) for ann in sorted(Path(tmp).glob("*/_annotations.coco.json"))
        ]
        if not folders:
            print("Download contained no _annotations.coco.json files")
            return 1
        info = flatten_coco_folders(folders, out_dir)

    url = f"https://universe.roboflow.com/{args.workspace}/{args.project}/dataset/{args.version}"
    (out_dir / "SOURCE.txt").write_text(f"{url}\n", encoding="utf-8")

    # Show how the dataset's labels map onto our classes so label_map can be filled in.
    coco = json.loads((out_dir / "labels.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    usage = Counter(names[a["category_id"]] for a in coco["annotations"])
    resolver = load_catalog().label_resolver()
    print(f"Saved {info['images']} images / {info['annotations']} boxes to {out_dir}\n")
    print(f"{'dataset label':<28}{'boxes':>7}  -> our class")
    for label, n in usage.most_common():
        target = resolve_label(resolver, label) or "UNMAPPED (add to label_map or it is dropped)"
        print(f"{label:<28}{n:>7}  -> {target}")
    print(
        f"\nNext: add to datasets.yaml\n"
        f"  - name: {args.name}\n"
        f"    path: data/raw/roboflow/{args.name}\n"
        f"    licence: <copy from {url}>\n"
        f"    url: {url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
