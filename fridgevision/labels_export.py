"""Build fridge-labels-vN.json, the file the app ships next to the model."""

from __future__ import annotations

from fridgevision.classes import ClassCatalog
from fridgevision.ocr_match import build_keyword_table
from fridgevision.text import SPECIAL_CHARS

LABELS_SCHEMA_VERSION = 1


def build_labels_json(
    catalog: ClassCatalog,
    *,
    model_file: str | None = None,
    model_info: dict | None = None,
) -> dict:
    """App-facing label file.

    - `detector.labels[i]` is the name for model output class index i (0 = background).
    - `classes` holds display name / category / detect mode for every ingredient,
      including OCR-only ones, so chips and the pantry can be rendered.
    - `ocr.keywords` is the matcher table, longest phrase first; an empty
      ingredient means "ignore phrase" (e.g. "milk chocolate").
    """
    return {
        "schemaVersion": LABELS_SCHEMA_VERSION,
        "classesVersion": catalog.version,
        "detector": {
            "modelFile": model_file,
            "labels": catalog.detector_labels,
            **(model_info or {}),
        },
        "categories": [{"id": k, "display": v} for k, v in catalog.categories.items()],
        "classes": {
            c.name: {"display": c.display, "category": c.category, "detect": c.detect}
            for c in catalog.classes
        },
        "ocr": {
            "languages": ["en", "fr", "es"],
            "normalize": {
                "lowercase": True,
                "specialChars": SPECIAL_CHARS,
                "stripDiacritics": "NFD, drop combining marks",
                "nonAlphanumeric": "replace [^a-z0-9]+ with a single space, trim",
            },
            "matching": "whole words; longest phrase claims its words first",
            "keywords": [[phrase, ingredient] for phrase, ingredient in build_keyword_table(catalog)],
        },
    }
