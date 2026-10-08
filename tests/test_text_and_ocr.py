import pytest

from fridgevision.ocr_match import build_keyword_table, ingredients_in_text
from fridgevision.text import normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Crème Fraîche", "creme fraiche"),
        ("Semi-Skimmed MILK", "semi skimmed milk"),
        ("Œufs frais", "oeufs frais"),
        ("jus d'orange", "jus d orange"),
        ("  Yogur\nGRIEGO!! ", "yogur griego"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.fixture(scope="module")
def table(catalog):
    return build_keyword_table(catalog)


# These cases are the spec for the app's labelDictionary.ts; port them as-is.
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cravendale SEMI-SKIMMED Milk 2L", ["milk"]),
        ("Elmlea Double Cream 300ml", ["cream"]),
        ("Philadelphia Original Cream Cheese", ["cream_cheese"]),
        ("Président Crème Fraîche épaisse", ["sour_cream"]),
        ("Danone Yaourt nature", ["yogurt"]),
        ("Yogur griego natural", ["yogurt"]),
        ("Leche semidesnatada", ["milk"]),
        ("Lait demi-écrémé UHT", ["milk"]),
        ("Heinz Tomato Ketchup", ["ketchup"]),
        ("Milk Chocolate Digestives", []),
        ("Whole Earth Peanut Butter", ["peanut_butter"]),
        ("Alpro Oat Milk", []),
        ("Ben & Jerry's Ice Cream", []),
        ("Best before: see lid. Keep refrigerated", []),
        ("Birmingham ham", ["ham"]),  # whole words only: no match inside "Birmingham"
        ("Free Range Eggs x6", ["egg"]),
        ("Chicken breast fillets and smoked salmon", ["chicken", "salmon"]),
        ("", []),
    ],
)
def test_ocr_matching(table, text, expected):
    assert ingredients_in_text(text, table) == expected


def test_keyword_table_is_longest_first(table):
    lengths = [len(p.split()) for p, _ in table]
    assert lengths == sorted(lengths, reverse=True)
