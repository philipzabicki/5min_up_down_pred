import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)

if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

loaded_features = sys.modules.get("features")
if loaded_features is not None:
    module_file = str(getattr(loaded_features, "__file__", "") or "")
    if not module_file.startswith(str(REPO_ROOT)):
        del sys.modules["features"]
