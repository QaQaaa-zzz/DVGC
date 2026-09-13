from __future__ import annotations

from pathlib import Path
import sys

import pytest


JIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = JIT_ROOT.parent
JIT_SOURCE_ROOT = JIT_ROOT / "src"
if str(JIT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(JIT_SOURCE_ROOT))


@pytest.fixture
def jit_root() -> Path:
    return JIT_ROOT


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a visible CUDA JAX backend")
