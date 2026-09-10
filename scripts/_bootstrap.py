"""
Makes `src` importable from every script in this folder, regardless of
which directory you're standing in when you run one.

Why this is needed: `python scripts/some_script.py` puts scripts/ ITSELF
on sys.path - not the project root above it, and not your current working
directory. So `from src.config import ...` fails with "ModuleNotFoundError:
No module named 'src'" unless the project root is added to sys.path
explicitly, which is all this file does.

Every script under scripts/ should import this FIRST, before any `from
src...` import:

    import _bootstrap  # noqa: F401  (adds project root to sys.path)
    from src.config import build_config
    ...

This works because scripts/'s own directory is always on sys.path
automatically (that part Python does for you) - so `import _bootstrap`
always succeeds as a plain sibling-file import, even before the project
root has been added.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
