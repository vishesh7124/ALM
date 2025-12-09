#!/usr/bin/env python3
"""Verify LTUAS setup is correct"""

import sys
from pathlib import Path

def verify_imports():
    """Check if all required packages can be imported"""
    print("Verifying imports...\n")
    
    tests = {
        "PyTorch": "import torch",
        "CLAP": "import laion_clap",
        "Mellow": "from mellow import MellowWrapper",
        "Groq": "from groq import Groq",
        "NumPy": "import numpy",
        "Librosa": "import librosa",
    }
    
    results = {}
    for name, import_str in tests.items():
        try:
            exec(import_str)
            results[name] = "✓"
            print(f"  ✓ {name} imported successfully")
        except Exception as e:
            results[name] = "✗"
            print(f"  ✗ {name} failed: {e}")
    
    print()
    return all(v == "✓" for v in results.values())

def verify_cuda():
    """Check CUDA availability"""
    import torch
    print("CUDA Status:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print()

def verify_models():
    """Test model loading"""
    print("Testing model loading...\n")
    
    try:
        import laion_clap
        print("  Loading CLAP...")
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        print("  ✓ CLAP model loaded successfully")
    except Exception as e:
        print(f"  ✗ CLAP loading failed: {e}")
    
    try:
        from mellow import MellowWrapper
        import torch
        print("\n  Loading MELLOW...")
        device = 0 if torch.cuda.is_available() else "cpu"
        mellow = MellowWrapper(
            config="v0",
            model="v0",
            device=device,
            use_cuda=torch.cuda.is_available()
        )
        print("  ✓ MELLOW model loaded successfully")
    except Exception as e:
        print(f"  ✗ MELLOW loading failed: {e}")
    
    print()

def verify_api_keys():
    """Check environment variables"""
    import os
    print("API Keys:")
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"  ✓ GROQ_API_KEY is set ({groq_key[:10]}...)")
    else:
        print("  ✗ GROQ_API_KEY not found in environment")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("LTUAS Setup Verification")
    print("=" * 60 + "\n")
    
    all_good = verify_imports()
    verify_cuda()
    verify_api_keys()
    verify_models()
    
    print("=" * 60)
    if all_good:
        print("✓ Setup verification complete! Ready to use LTUAS.")
    else:
        print("✗ Some components failed. Please check errors above.")
    print("=" * 60)
