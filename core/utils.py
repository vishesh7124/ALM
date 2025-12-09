from pathlib import Path
from typing import List
import os

def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all audio files in directory"""
    if extensions is None:
        extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    
    audio_files = []
    directory = Path(directory)
    
    for ext in extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
    
    return [str(f) for f in sorted(audio_files)]

def validate_audio_file(file_path: str) -> bool:
    """Validate audio file exists and is readable"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if not path.is_file():
        print(f"❌ Not a file: {file_path}")
        return False
    
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    if path.suffix.lower() not in valid_extensions:
        print(f"❌ Invalid audio format: {path.suffix}")
        return False
    
    return True

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
