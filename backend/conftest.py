"""Pytest configuration for the backend test suite.

The backend modules are top-level (``import main``, ``import model_service``)
rather than a package, so the backend directory has to be importable. Adding it
here means ``pytest`` works from any directory, not only via ``python -m pytest``
from ``backend/`` (which happens to work because ``python -m`` puts the current
directory on ``sys.path``).
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
