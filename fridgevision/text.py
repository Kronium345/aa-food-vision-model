"""Text normalisation shared by OCR matching and dataset label mapping.

The app's TypeScript matcher must normalise exactly the same way; the rules are
exported into fridge-labels-vN.json (see labels_export.py) so it can.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures and letters that NFD decomposition doesn't reduce to ASCII.
SPECIAL_CHARS = {"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l"}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, turn punctuation into spaces, collapse whitespace.

    "Crème Fraîche" -> "creme fraiche", "Semi-Skimmed" -> "semi skimmed",
    "Œufs" -> "oeufs", "jus d'orange" -> "jus d orange".
    """
    text = text.lower()
    for src, dst in SPECIAL_CHARS.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return _NON_ALNUM.sub(" ", text).strip()
