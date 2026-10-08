import copy

import pytest

from fridgevision.classes import ClassesError, parse_catalog, resolve_label
from fridgevision.labels_export import build_labels_json

BASE = {
    "version": 1,
    "categories": {"fruits": "Fruits", "dairy_eggs": "Dairy & eggs"},
    "classes": [
        {"name": "apple", "display": "Apple", "category": "fruits", "detect": "visual",
         "aliases": ["apples"], "open_images": ["Apple"]},
        {"name": "milk", "display": "Milk", "category": "dairy_eggs", "detect": "both",
         "ocr": {"en": ["milk"]}},
        {"name": "yogurt", "display": "Yogurt", "category": "dairy_eggs", "detect": "ocr",
         "ocr": {"en": ["yogurt"], "fr": ["yaourt"]}},
    ],
}


def mutate(fn):
    raw = copy.deepcopy(BASE)
    fn(raw)
    return raw


def test_real_classes_yaml_is_valid(catalog):
    assert catalog.detector_labels[0] == "background"
    assert len(catalog.visual_classes) >= 40


def test_detector_labels_follow_file_order_and_skip_ocr_only():
    assert parse_catalog(BASE).detector_labels == ["background", "apple", "milk"]


@pytest.mark.parametrize(
    "fn, message",
    [
        (lambda r: r["classes"].append(dict(r["classes"][0])), "duplicate name"),
        (lambda r: r["classes"][0].update(category="nope"), "unknown category"),
        (lambda r: r["classes"][0].update(detect="maybe"), "detect must be"),
        (lambda r: r["classes"][0].update(name="Apple Pie"), "snake_case"),
        (lambda r: r["classes"][1].update(ocr={}), "needs at least one OCR keyword"),
        (lambda r: r["classes"][0].update(ocr={"en": ["apple"]}), "use detect=both"),
        (lambda r: r["classes"][2]["ocr"].update(de=["joghurt"]), "unsupported OCR language"),
        (lambda r: r["classes"][2]["ocr"].update(en=["Milk"]), "already belongs to 'milk'"),
        (lambda r: r["classes"][1].update(aliases=["apples"]), "already maps to 'apple'"),
        (lambda r: r["classes"][2].update(aliases=["yoghurt_pot"]), "OCR-only classes"),
        (lambda r: r.update(ocr_ignore=["milk"]), "ocr_ignore phrase"),
    ],
)
def test_invalid_catalogs_are_rejected(fn, message):
    with pytest.raises(ClassesError, match=message):
        parse_catalog(mutate(fn))


def test_label_resolver_normalises_and_applies_overrides():
    cat = parse_catalog(BASE)
    resolver = cat.label_resolver({"Red Delicious": "apple", "fruit": None})
    assert resolve_label(resolver, "Apples") == "apple"
    assert resolve_label(resolver, "APPLE") == "apple"
    assert resolve_label(resolver, "red-delicious") == "apple"
    assert resolve_label(resolver, "fruit") is None
    assert resolve_label(resolver, "yogurt") is None  # OCR-only: never a box label
    with pytest.raises(ClassesError):
        cat.label_resolver({"x": "yogurt"})


def test_labels_json_shape():
    out = build_labels_json(parse_catalog(BASE), model_file="m.tflite", model_info={"inputSize": 384})
    assert out["detector"] == {
        "modelFile": "m.tflite",
        "labels": ["background", "apple", "milk"],
        "inputSize": 384,
    }
    assert out["classes"]["yogurt"] == {"display": "Yogurt", "category": "dairy_eggs", "detect": "ocr"}
    assert ["yaourt", "yogurt"] in out["ocr"]["keywords"]
