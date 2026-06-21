"""Shared path resolution for the FootballAI monorepo.

All scripts default to the monorepo root for shared runtime directories such as
``data/``, ``models/``, and ``external/``. The root is discovered by walking up
from this installed package and looking for the workspace manifest.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    """Return the monorepo root directory.

    The search starts at this file, walks upward looking for ``pyproject.toml``
    containing the uv workspace marker, and falls back to the first directory
    containing a ``.git`` folder.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        marker = parent / "pyproject.toml"
        if marker.exists():
            try:
                content = marker.read_text(encoding="utf-8")
                if "[tool.uv.workspace]" in content:
                    return parent
            except OSError:
                pass
        if (parent / ".git").is_dir():
            return parent
    # Fallback: package source root -> packages/footballai -> packages -> repo root
    return here.parent.parent.parent.parent


REPO_ROOT = find_repo_root()
