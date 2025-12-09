# quick_setup.py - Quick setup for Windows
import sys
from pathlib import Path

print("="*60)
print("LTUAS Quick Setup")
print("="*60)

# Step 1: Create config/__init__.py
print("\n1. Creating config/__init__.py...")
config_init = Path("config/__init__.py")
config_init.parent.mkdir(exist_ok=True)

config_init_content = '''# config/__init__.py
import sys
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent.parent

# Add Mellow to Python path
MELLOW_PATH = BASE_DIR / "external_models" / "mellow"
if MELLOW_PATH.exists() and str(MELLOW_PATH) not in sys.path:
    sys.path.insert(0, str(MELLOW_PATH))
    print(f"[Config] Added Mellow to path: {MELLOW_PATH}")
'''

config_init.write_text(config_init_content)
print("   [OK] Created config/__init__.py")

# Step 2: Create .env.example
print("\n2. Creating .env.example...")
env_example = Path(".env.example")
env_example_content = '''# .env.example
# Copy this file to .env and add your actual API key
GROQ_API_KEY=your_groq_api_key_here
'''
env_example.write_text(env_example_content)
print("   [OK] Created .env.example")

# Step 3: Check if .env exists
env_file = Path(".env")
if not env_file.exists():
    print("\n3. Creating .env file...")
    print("   [!] Please edit .env and add your GROQ_API_KEY")
    env_file.write_text(env_example_content)
else:
    print("\n3. .env file already exists")

# Step 4: Test imports
print("\n4. Testing imports...")
try:
    import config  # This adds Mellow to path
    from mellow import MellowWrapper
    print("   [OK] Mellow can now be imported!")
except ImportError as e:
    print(f"   [X] Mellow import failed: {e}")

try:
    import laion_clap
    print("   [OK] CLAP is available")
except ImportError as e:
    print(f"   [X] CLAP import failed: {e}")

try:
    from groq import Groq
    print("   [OK] Groq is available")
except ImportError as e:
    print(f"   [X] Groq import failed: {e}")

print("\n" + "="*60)
print("Setup complete!")
print("Next steps:")
print("1. Edit .env and add your GROQ_API_KEY")
print("2. Run: python diagnose.py (to verify)")
print("3. Run: python main.py path/to/audio.wav")
print("="*60)
