# diagnose.py - Windows-compatible diagnostics (UPDATED)
import sys
from pathlib import Path
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("LTUAS Diagnostics")
print("="*60)

# Use ASCII characters instead of Unicode checkmarks
OK = "[OK]"
FAIL = "[X]"

# Check current directory
print(f"\n1. Current Directory:")
print(f"   {Path.cwd()}")

# Check if external_models exists
print(f"\n2. Checking folder structure:")
base = Path.cwd()
folders_to_check = [
    "external_models",
    "external_models/CLAP",
    "external_models/mellow",
    "config",
    "models",
    "core",
]

for folder in folders_to_check:
    folder_path = base / folder
    status = OK if folder_path.exists() else FAIL
    print(f"   {status} {folder}")

# Check Python path
print(f"\n3. Python executable:")
print(f"   {sys.executable}")

# Check if we can find the model folders
print(f"\n4. Looking for model files:")
mellow_path = base / "external_models" / "mellow"
clap_path = base / "external_models" / "CLAP"

if mellow_path.exists():
    mellow_files = list(mellow_path.glob("*"))
    print(f"   {OK} Mellow folder contains {len(mellow_files)} items")
    has_setup = (mellow_path / "setup.py").exists()
    has_module = (mellow_path / "mellow").exists()
    print(f"     - setup.py: {OK if has_setup else FAIL}")
    print(f"     - mellow/ module: {OK if has_module else FAIL}")
else:
    print(f"   {FAIL} Mellow folder not found at {mellow_path}")

if clap_path.exists():
    clap_files = list(clap_path.glob("*"))
    print(f"   {OK} CLAP folder contains {len(clap_files)} items")
    has_setup = (clap_path / "setup.py").exists()
    has_src = (clap_path / "src").exists()
    print(f"     - setup.py: {OK if has_setup else FAIL}")
    print(f"     - src/ folder: {OK if has_src else FAIL}")
else:
    print(f"   {FAIL} CLAP folder not found at {clap_path}")

# Try basic imports
print(f"\n5. Testing basic imports:")
try:
    import torch
    print(f"   {OK} PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   {FAIL} PyTorch failed: {e}")

try:
    import numpy
    print(f"   {OK} NumPy {numpy.__version__}")
except ImportError as e:
    print(f"   {FAIL} NumPy failed: {e}")

try:
    from groq import Groq
    print(f"   {OK} Groq")
except ImportError as e:
    print(f"   {FAIL} Groq failed: {e}")

# Try CLAP import
print(f"\n6. Testing CLAP import:")
try:
    import laion_clap
    print(f"   {OK} laion_clap imported successfully")
    print(f"     Location: {laion_clap.__file__}")
except ImportError as e:
    print(f"   {FAIL} laion_clap import failed: {e}")
    print(f"     Try: pip install laion-clap")

# IMPORTANT: Import config to add Mellow to path
print(f"\n7. Testing Mellow import:")
print(f"   (Importing config to add Mellow to sys.path...)")
try:
    import config  # This adds Mellow to sys.path (if configured)
    print(f"   Config module loaded")
except ImportError as e:
    print(f"   {FAIL} Config import failed: {e}")

# Ensure external_models/mellow is on sys.path so 'mellow' package can be resolved
mellow_dir = base / "external_models" / "mellow"
mellow_pkg_dir = mellow_dir / "mellow"
if mellow_pkg_dir.exists():
    p = str(mellow_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
        print(f"   Added {p} to sys.path")
else:
    print(f"   {FAIL} mellow package not found at {mellow_pkg_dir}")

try:
    from mellow import MellowWrapper
    import mellow
    print(f"   {OK} mellow imported successfully")
    print(f"     Location: {mellow.__file__}")
except ImportError as e:
    print(f"   {FAIL} mellow import failed: {e}")
    print(f"     Make sure external_models/mellow/mellow exists or config/__init__.py adds Mellow to sys.path")

# Check environment variables
print(f"\n8. Environment variables:")
# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"   Loaded .env file")
except ImportError:
    print(f"   (python-dotenv not installed, checking env directly)")

groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"   {OK} GROQ_API_KEY is set ({groq_key[:10]}...)")
else:
    print(f"   {FAIL} GROQ_API_KEY not found")
    print(f"     Create .env file with: GROQ_API_KEY=your_key_here")

print("\n" + "="*60)
print("Diagnostics complete!")
print("="*60)
