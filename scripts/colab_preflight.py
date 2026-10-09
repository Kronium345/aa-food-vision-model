"""Pre-flight check before training on Colab (run it right before cell 5).

    !git -C /content/repo pull
    !python scripts/colab_preflight.py

Uses only the standard library, so it runs under Colab's default Python. It:

1. Pins the two training-env packages that broke earlier sessions, inside the
   Python 3.11 venv from cell 3 (a fresh cell 3 already installs these from
   requirements-train.lock; this repairs a venv created before the fix):
     - setuptools<81, which still ships pkg_resources (tensorflow-hub imports it)
     - tensorflow-metadata==1.15.0, the last release built for protobuf 4 (TF 2.15)
2. Renames <split>/data -> <split>/images in datasets built before
   build_dataset.py switched to the folder name Model Maker reads.
3. Prints each dataset's image/box counts and checks the training imports.

Safe to re-run: every step is skipped when there's nothing to do.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS = ["setuptools==80.9.0", "tensorflow-metadata==1.15.0"]
SPLITS = ("train", "validation", "test")

# The same imports train.py needs, run inside the training venv.
IMPORT_CHECK = """
import os, warnings
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")
import pkg_resources
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow as tf
from mediapipe_model_maker import object_detector
print("tensorflow", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))
"""


def fix_venv(python: Path) -> bool:
    if not python.exists():
        print(f"[venv] {python} not found: run cell 3 first (Colab may have restarted).")
        return False
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "-q", "--python", str(python), *PINS]
    else:
        cmd = [str(python), "-m", "pip", "install", "-q", *PINS]
    print(f"[venv] pinning {', '.join(PINS)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[venv] install failed (see the output above)")
        return False
    return True


def migrate_datasets(processed: Path) -> None:
    if not processed.is_dir():
        print(f"[data] {processed} doesn't exist yet: run cells 4a and 4b first.")
        return
    versions = sorted(p for p in processed.iterdir() if p.is_dir())
    if not versions:
        print(f"[data] no datasets in {processed}: run cell 4b first.")
    for version in versions:
        stats_path = version / "stats.json"
        if not stats_path.exists():
            # stats.json is written last, so its absence means the build didn't finish.
            print(f"[data] {version.name}: incomplete build. Delete it and re-run cell 4b:")
            print(f"         !rm -rf {version}")
            continue
        renamed = []
        for split in SPLITS:
            old, new = version / split / "data", version / split / "images"
            if old.is_dir() and not new.exists():
                os.rename(old, new)
                renamed.append(split)
        if renamed:
            print(f"[data] {version.name}: renamed data/ -> images/ in {', '.join(renamed)}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"[data] {version.name}:")
        for split, info in stats["splits"].items():
            folder = "ok" if (version / split / "images").is_dir() else "MISSING images/"
            print(f"         {split:<11} {info['images']:>6} images {info['boxes']:>7} boxes  ({folder})")


def check_imports(python: Path) -> bool:
    if not python.exists():
        return False
    print("[check] importing TensorFlow + Model Maker in the training venv (about a minute)...")
    result = subprocess.run([str(python), "-c", IMPORT_CHECK], capture_output=True, text=True)
    if result.returncode == 0:
        print("[check] OK:", result.stdout.strip().splitlines()[-1])
        return True
    print("[check] FAILED. Last lines of the error:")
    print("\n".join(result.stderr.strip().splitlines()[-8:]))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", type=Path, default=Path("/content/py311/bin/python"),
                        help="training venv's Python (created by cell 3)")
    parser.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()

    venv_ok = fix_venv(args.python)
    migrate_datasets(args.processed)
    imports_ok = venv_ok and check_imports(args.python)

    print("\nREADY: run cell 5." if imports_ok else "\nNOT READY: fix the items above, then re-run this cell.")
    return 0 if imports_ok else 1


if __name__ == "__main__":
    sys.exit(main())
