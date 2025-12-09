# config/paths.py
import sys
from pathlib import Path

# Add external models to Python path
BASE_DIR = Path(__file__).parent.parent
EXTERNAL_MODELS_DIR = BASE_DIR / "external_models"

# Add CLAP to path
CLAP_PATH = EXTERNAL_MODELS_DIR / "CLAP" / "src"
if CLAP_PATH.exists():
    sys.path.insert(0, str(CLAP_PATH))

# Add Mellow to path
MELLOW_PATH = EXTERNAL_MODELS_DIR / "mellow"
if MELLOW_PATH.exists():
    sys.path.insert(0, str(MELLOW_PATH))

print(f"✓ CLAP path added: {CLAP_PATH}")
print(f"✓ Mellow path added: {MELLOW_PATH}")
