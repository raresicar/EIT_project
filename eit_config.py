"""
Project root configuration.

Every script imports this to get PROJECT_ROOT and add submodules to sys.path.

Usage (from any script anywhere in the project):
    import sys
    from pathlib import Path
    # Walk up to find eit_config.py
    _dir = Path(__file__).resolve().parent
    while not (_dir / "eit_config.py").exists():
        _dir = _dir.parent
    sys.path.insert(0, str(_dir))

    from eit_config import PROJECT_ROOT, MESH_DIR, DATA_DIR
"""

from pathlib import Path
import sys

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Submodule paths
FORWARD_DIR = PROJECT_ROOT / "forward"
MESHES_DIR = PROJECT_ROOT / "meshes"

# Data paths (override with CLI args or env vars if needed)
BRAINWEB_SUBJECTS_DIR = PROJECT_ROOT / "brainweb_subjects"
BRAINWEB_MESHES_DIR = PROJECT_ROOT / "brainweb_meshes"
MONITORING_DATA_DIR = PROJECT_ROOT / "monitoring_data"
STROKE_DATA_DIR = PROJECT_ROOT / "brainweb_strokes"

# Add submodules to sys.path so cross-imports work
for p in [FORWARD_DIR, MESHES_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))