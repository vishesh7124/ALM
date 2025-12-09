#!/usr/bin/env python3
import os, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "layered_datasets(MAD)"
OUTPUT_RESULTS_DIR = ROOT / "outputs/results"
TMP_BATCH_DIR = ROOT / "tmp_batch"
MAIN_PY = ROOT / "main.py"
PYTHON_CMD = "python"
MAX_FILES = 500
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".WAV", ".MP3", ".FLAC", ".OGG"}

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def get_audio_files(lang_audio_dir: Path, n: int):
    if not lang_audio_dir.exists():
        print(f" ⚠️ Audio dir does not exist: {lang_audio_dir}")
        return []
    files = []
    for f in lang_audio_dir.rglob("*"):
        if f.is_file() and f.suffix in AUDIO_EXTS:
            files.append(f)
    files.sort()
    print(f" → Found {len(files)} audio files in {lang_audio_dir} (showing first 10):")
    for f in files[:10]:
        print("    ", f)
    return files[:n]

def prepare_batch(lang: str, audio_files):
    tmp = TMP_BATCH_DIR / lang
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)
    for f in audio_files:
        # Use just file name (flatten) so symlink destination is not too nested
        dst = tmp / f.name
        try:
            os.symlink(f, dst)
        except Exception as e:
            print(f" ⚠️ Failed to symlink {f} → {dst}: {e}")
    return tmp

def run_language(lang: str, batch_dir: Path):
    print(f"\n--- Processing language: {lang} ---")
    print("CMD:", PYTHON_CMD, MAIN_PY, str(batch_dir), "--batch")
    proc = subprocess.Popen(
        [PYTHON_CMD, str(MAIN_PY), str(batch_dir), "--batch"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    for line in proc.stdout:
        print(line, end="")
    return proc.wait() == 0

def copy_results(lang: str):
    dst_dir = DATASETS_DIR / lang / "jsons"
    ensure_dir(dst_dir)
    for js in OUTPUT_RESULTS_DIR.glob("*.json"):
        shutil.copy2(js, dst_dir)
    print(f" 📁 Copied JSONs to {dst_dir}")

def main():
    ensure_dir(TMP_BATCH_DIR)
    for lang_folder in DATASETS_DIR.iterdir():
        if not lang_folder.is_dir():
            continue
        lang = lang_folder.name
        audio_dir = lang_folder / "audio"
        audio_files = get_audio_files(audio_dir, MAX_FILES)
        if not audio_files:
            print(f" ⚠️ No audio files to process for {lang}, skipping.")
            continue
        batch_dir = prepare_batch(lang, audio_files)
        success = run_language(lang, batch_dir)
        if success:
            copy_results(lang)
        else:
            print(f" ❌ main.py failed for {lang}")

if __name__ == "__main__":
    main()
