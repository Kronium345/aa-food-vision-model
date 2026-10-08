"""Load and validate classes.yaml, the single source of truth for labels."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fridgevision.text import normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES_PATH = REPO_ROOT / "classes.yaml"

DETECT_MODES = ("visual", "both", "ocr")
OCR_LANGUAGES = ("en", "fr", "es")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ClassesError(ValueError):
    """classes.yaml is malformed or inconsistent."""


@dataclass(frozen=True)
class IngredientClass:
    name: str
    display: str
    category: str
    detect: str
    aliases: tuple[str, ...] = ()
    open_images: tuple[str, ...] = ()
    ocr: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return self.detect in ("visual", "both")

    @property
    def is_ocr(self) -> bool:
        return self.detect in ("ocr", "both")


@dataclass(frozen=True)
class ClassCatalog:
    version: int
    categories: dict[str, str]
    classes: tuple[IngredientClass, ...]
    ocr_ignore: tuple[str, ...]

    @property
    def visual_classes(self) -> tuple[IngredientClass, ...]:
        """Detector classes in model index order (index = position + 1)."""
        return tuple(c for c in self.classes if c.is_visual)

    @property
    def detector_labels(self) -> list[str]:
        """Model output labels, index 0 = background (MediaPipe convention)."""
        return ["background"] + [c.name for c in self.visual_classes]

    def by_name(self, name: str) -> IngredientClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    def label_resolver(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Map normalised dataset label -> visual class name.

        Covers class names, aliases and Open Images names. `extra` holds
        per-source overrides from datasets.yaml; mapping a label to null/""
        drops it explicitly.
        """
        resolver: dict[str, str] = {}
        for c in self.visual_classes:
            for label in (c.name, *c.aliases, *c.open_images):
                resolver[_label_key(label)] = c.name
        for raw, target in (extra or {}).items():
            if target and target not in {c.name for c in self.visual_classes}:
                raise ClassesError(
                    f"label_map target '{target}' is not a visual class in classes.yaml"
                )
            resolver[_label_key(raw)] = target or ""
        return resolver


def _label_key(label: str) -> str:
    """Normalise dataset labels: 'Bell pepper', 'bell-pepper', 'Bell_Pepper' -> 'bell pepper'."""
    return normalize_text(label.replace("_", " ").replace("-", " "))


def resolve_label(resolver: dict[str, str], label: str) -> str | None:
    target = resolver.get(_label_key(label))
    return target or None


def load_catalog(path: Path | str = DEFAULT_CLASSES_PATH) -> ClassCatalog:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_catalog(raw)


def parse_catalog(raw: dict) -> ClassCatalog:
    if not isinstance(raw, dict):
        raise ClassesError("classes.yaml must be a mapping")
    errors: list[str] = []

    version = raw.get("version")
    if not isinstance(version, int) or version < 1:
        errors.append("version must be a positive integer")

    categories = raw.get("categories") or {}
    if not isinstance(categories, dict) or not categories:
        errors.append("categories must be a non-empty mapping")
        categories = {}

    classes: list[IngredientClass] = []
    seen: set[str] = set()
    label_owner: dict[str, str] = {}
    keyword_owner: dict[str, str] = {}

    for i, entry in enumerate(raw.get("classes") or []):
        where = f"classes[{i}]"
        name = entry.get("name", "")
        if not NAME_RE.match(str(name)):
            errors.append(f"{where}: name '{name}' must be snake_case")
            continue
        where = f"class '{name}'"
        if name in seen:
            errors.append(f"{where}: duplicate name")
        seen.add(name)

        detect = entry.get("detect")
        if detect not in DETECT_MODES:
            errors.append(f"{where}: detect must be one of {DETECT_MODES}")
        category = entry.get("category")
        if category not in categories:
            errors.append(f"{where}: unknown category '{category}'")
        if not entry.get("display"):
            errors.append(f"{where}: display is required")

        ocr_raw = entry.get("ocr") or {}
        ocr: dict[str, tuple[str, ...]] = {}
        for lang, words in ocr_raw.items():
            if lang not in OCR_LANGUAGES:
                errors.append(f"{where}: unsupported OCR language '{lang}'")
                continue
            ocr[lang] = tuple(str(w) for w in words or [])
        if detect in ("ocr", "both") and not any(ocr.values()):
            errors.append(f"{where}: detect={detect} needs at least one OCR keyword")
        if detect == "visual" and ocr:
            errors.append(f"{where}: detect=visual but has OCR keywords (use detect=both)")

        for words in ocr.values():
            for w in words:
                key = normalize_text(w)
                owner = keyword_owner.setdefault(key, name)
                if owner != name:
                    errors.append(f"{where}: OCR keyword '{w}' already belongs to '{owner}'")

        aliases = tuple(str(a) for a in entry.get("aliases") or [])
        open_images = tuple(str(o) for o in entry.get("open_images") or [])
        if detect == "ocr" and (aliases or open_images):
            errors.append(f"{where}: OCR-only classes cannot have dataset aliases/open_images")
        for label in (name, *aliases, *open_images):
            key = _label_key(label)
            owner = label_owner.setdefault(key, name)
            if owner != name:
                errors.append(f"{where}: dataset label '{label}' already maps to '{owner}'")

        classes.append(
            IngredientClass(
                name=name,
                display=str(entry.get("display", "")),
                category=str(category),
                detect=str(detect),
                aliases=aliases,
                open_images=open_images,
                ocr=ocr,
            )
        )

    if not classes:
        errors.append("classes must not be empty")

    ocr_ignore = tuple(str(p) for p in raw.get("ocr_ignore") or [])
    for phrase in ocr_ignore:
        if normalize_text(phrase) in keyword_owner:
            errors.append(
                f"ocr_ignore phrase '{phrase}' is also a keyword of "
                f"'{keyword_owner[normalize_text(phrase)]}'"
            )

    if errors:
        raise ClassesError("classes.yaml is invalid:\n  - " + "\n  - ".join(errors))

    return ClassCatalog(
        version=version,
        categories={str(k): str(v) for k, v in categories.items()},
        classes=tuple(classes),
        ocr_ignore=ocr_ignore,
    )
