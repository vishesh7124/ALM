# config/__init__.py
import sys
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent.parent

# Add Mellow to Python path
MELLOW_PATH = BASE_DIR / "external_models" / "mellow"
if MELLOW_PATH.exists() and str(MELLOW_PATH) not in sys.path:
    sys.path.insert(0, str(MELLOW_PATH))
    print(f"[Config] Added Mellow to path: {MELLOW_PATH}")
