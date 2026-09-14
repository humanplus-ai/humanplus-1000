"""Make ``humanplus_viewer`` importable from a source checkout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def setup() -> Path:
    """Put the toolkit root on ``sys.path`` and register the package name."""
    root = Path(__file__).resolve().parents[1]
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        import humanplus_viewer  # noqa: F401
        return root
    except ImportError:
        pass

    pkg_name = "humanplus_viewer"
    init_path = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {pkg_name} from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)
    return root
