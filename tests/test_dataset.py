import json
import zipfile

import pytest
from PIL import Image

from fridgevision.classes import parse_catalog
from fridgevision.dataset import (
    DatasetConfig,
    DatasetError,
    Source,
    build_dataset,
    flatten_coco_folders,
)
from fridgevision.model_file import embedded_labels, labels_match

CATALOG = parse_catalog({
    "version": 1,
    "categories": {"fruits": "Fruits"},
    "classes": [
        {"name": "apple", "display": "Apple", "category": "fruits", "detect": "visual",
         "open_images": ["Apple"]},
        {"name": "lime", "display": "Lime", "category": "fruits", "detect": "visual",
         "aliases": ["limes"]},
    ],
})


def make_source(root, seed, images, categories, annotations):
    (root / "data").mkdir(parents=True)
    for i, img in enumerate(images):
        # distinct pixels per (seed, image) so content hashes differ
        Image.new("RGB", (img["width"], img["height"]), (i * 7 % 255, seed, 3)).save(
            root / "data" / img["file_name"]
        )
    (root / "labels.json").write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )
    return root


def cfg(sources, allowed=("own", "CC BY 4.0")):
    return DatasetConfig(allowed_licences=allowed, validation_fraction=0.0, sources=sources)


def test_build_merges_remaps_and_cleans(tmp_path):
    src = make_source(
        tmp_path / "oi", 1,
        images=[{"id": 10, "file_name": "a.jpg", "width": 100, "height": 80},
                {"id": 11, "file_name": "b.jpg", "width": 100, "height": 80},
                {"id": 12, "file_name": "c.jpg", "width": 100, "height": 80}],
        categories=[{"id": 5, "name": "Apple"}, {"id": 6, "name": "Limes"}, {"id": 7, "name": "Car"}],
        annotations=[
            {"id": 1, "image_id": 10, "category_id": 5, "bbox": [90, 70, 30, 30]},  # clamped
            {"id": 2, "image_id": 10, "category_id": 6, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 11, "category_id": 7, "bbox": [0, 0, 10, 10]},  # unmapped
            {"id": 4, "image_id": 12, "category_id": 5, "bbox": [0, 0, 1, 40]},  # degenerate
        ],
    )
    test_src = make_source(
        tmp_path / "own_test", 2,
        images=[{"id": 1, "file_name": "t.jpg", "width": 50, "height": 50}],
        categories=[{"id": 1, "name": "lime"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20]}],
    )
    out = tmp_path / "out"
    stats = build_dataset(CATALOG, cfg([
        Source("oi", src, "CC BY 4.0"),
        Source("own_test", test_src, "own", role="test"),
    ]), out)

    train = json.loads((out / "train" / "labels.json").read_text())
    assert train["categories"] == [
        {"id": 1, "name": "apple", "supercategory": "fruits"},
        {"id": 2, "name": "lime", "supercategory": "fruits"},
    ]
    assert len(train["images"]) == 1  # b (no mapped boxes) and c (degenerate only) dropped
    boxes = {a["category_id"]: a["bbox"] for a in train["annotations"]}
    assert boxes[1] == [90.0, 70.0, 10.0, 10.0]
    assert boxes[2] == [0.0, 0.0, 10.0, 10.0]
    assert (out / "train" / "images" / train["images"][0]["file_name"]).exists()

    test = json.loads((out / "test" / "labels.json").read_text())
    assert len(test["images"]) == 1 and test["annotations"][0]["category_id"] == 2
    validation = json.loads((out / "validation" / "labels.json").read_text())
    assert validation["categories"] == train["categories"]  # same label map in every split

    assert stats["sources"]["oi"]["dropped"] == {"Car": 1, "<degenerate box>": 1}
    assert stats["splits"]["train"]["per_class"] == {"apple": 1, "lime": 1}


def test_duplicate_images_across_sources_are_skipped(tmp_path):
    kwargs = dict(images=[{"id": 1, "file_name": "x.jpg", "width": 20, "height": 20}],
                  categories=[{"id": 1, "name": "apple"}],
                  annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}])
    a = make_source(tmp_path / "a", 9, **kwargs)
    b = make_source(tmp_path / "b", 9, **kwargs)
    stats = build_dataset(CATALOG, cfg([Source("a", a, "own"), Source("b", b, "own")]), tmp_path / "out")
    assert stats["duplicates_skipped"] == 1
    assert stats["splits"]["train"]["images"] == 1


def test_disallowed_licence_is_refused(tmp_path):
    with pytest.raises(DatasetError, match="AGPL"):
        build_dataset(CATALOG, cfg([Source("x", tmp_path, "AGPL-3.0")]), tmp_path / "out")


def test_non_empty_output_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    src = make_source(tmp_path / "s", 3, [], [], [])
    with pytest.raises(DatasetError, match="not empty"):
        build_dataset(CATALOG, cfg([Source("s", src, "own")]), out)


def test_flatten_roboflow_style_folders(tmp_path):
    folders = []
    for split, cats in (("train", [{"id": 0, "name": "fridge"}, {"id": 1, "name": "lime"}]),
                        ("valid", [{"id": 0, "name": "fridge"}, {"id": 1, "name": "apple"}])):
        d = tmp_path / "rf" / split
        d.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(d / "img.jpg")
        (d / "_annotations.coco.json").write_text(json.dumps({
            "categories": cats,
            "images": [{"id": 0, "file_name": "img.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        }))
        folders.append((d / "_annotations.coco.json", d))
    info = flatten_coco_folders(folders, tmp_path / "flat")
    coco = json.loads((tmp_path / "flat" / "labels.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert info["images"] == 2
    assert sorted(names[a["category_id"]] for a in coco["annotations"]) == ["apple", "lime"]
    assert len({i["file_name"] for i in coco["images"]}) == 2  # same name in both folders kept apart


def test_embedded_labels_from_tflite_zip(tmp_path):
    model = tmp_path / "m.tflite"
    with zipfile.ZipFile(model, "w") as zf:
        zf.writestr("labels.txt", "background\napple\nlime\n")
    labels = embedded_labels(model)
    assert labels == ["background", "apple", "lime"]
    assert labels_match(labels, CATALOG.detector_labels)
    assert labels_match(["apple", "lime"], CATALOG.detector_labels)
    assert not labels_match(["lime", "apple"], CATALOG.detector_labels)
    plain = tmp_path / "plain.tflite"
    plain.write_bytes(b"TFL3")
    assert embedded_labels(plain) is None
