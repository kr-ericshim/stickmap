"""Skip the image-fixture tests when the fixtures are not here.

`fixtures/` holds frames captured from other people's transcription videos, so it stays out of
the public repository (see .gitignore). Test modules that read it are skipped on a clean clone.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Modules that read a fixture at import time cannot even be collected without it.
collect_ignore = [] if FIXTURES.is_dir() else ["test_xiangsong.py"]


def pytest_collection_modifyitems(config, items):
    if FIXTURES.is_dir():
        return
    skip = pytest.mark.skip(reason="tools/truth_video/tests/fixtures is not in the public repository")
    reads: dict[Path, bool] = {}
    for item in items:
        path = Path(str(item.fspath))
        if path.parent != FIXTURES.parent:
            continue
        if path not in reads:
            reads[path] = "'fixtures" in path.read_text(encoding="utf-8")
        if reads[path]:
            item.add_marker(skip)
