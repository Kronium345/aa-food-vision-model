import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fridgevision.classes import load_catalog  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()
