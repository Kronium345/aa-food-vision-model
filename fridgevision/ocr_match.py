"""Reference OCR text -> ingredient matcher.

This is the spec for the app's lib/fridgeScan/labelDictionary.ts: same
normalisation, same longest-phrase-wins rule, driven by the `ocr` block in
fridge-labels-vN.json. Keep the two in sync (tests/test_ocr_match.py holds the
cases to port).
"""

from __future__ import annotations

from dataclasses import dataclass

from fridgevision.classes import ClassCatalog
from fridgevision.text import normalize_text

IGNORE = ""  # target for ocr_ignore phrases


@dataclass(frozen=True)
class OcrMatch:
    ingredient: str
    keyword: str
    start: int  # word index into the normalised text
    end: int


def build_keyword_table(catalog: ClassCatalog) -> list[tuple[str, str]]:
    """(normalised phrase, ingredient) pairs, longest phrase first.

    Ignore phrases map to "" so they win over the shorter keywords they contain.
    """
    table: dict[str, str] = {}
    for c in catalog.classes:
        for words in c.ocr.values():
            for w in words:
                table[normalize_text(w)] = c.name
    for phrase in catalog.ocr_ignore:
        table[normalize_text(phrase)] = IGNORE
    return sorted(table.items(), key=lambda kv: (-len(kv[0].split()), -len(kv[0]), kv[0]))


def match_text(text: str, table: list[tuple[str, str]]) -> list[OcrMatch]:
    """Find ingredients in OCR text. Whole-word matches; longest phrase claims its words first."""
    words = normalize_text(text).split()
    if not words:
        return []
    claimed = [False] * len(words)
    matches: list[OcrMatch] = []
    for phrase, ingredient in table:
        pwords = phrase.split()
        n = len(pwords)
        for i in range(len(words) - n + 1):
            if words[i : i + n] != pwords or any(claimed[i : i + n]):
                continue
            for j in range(i, i + n):
                claimed[j] = True
            if ingredient != IGNORE:
                matches.append(OcrMatch(ingredient, phrase, i, i + n))
    return sorted(matches, key=lambda m: m.start)


def ingredients_in_text(text: str, table: list[tuple[str, str]]) -> list[str]:
    """Distinct ingredients in reading order."""
    out: list[str] = []
    for m in match_text(text, table):
        if m.ingredient not in out:
            out.append(m.ingredient)
    return out
