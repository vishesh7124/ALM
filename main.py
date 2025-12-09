#!/usr/bin/env python3
"""
LTUAS - Listen, Think, Understand Audio System
Main entry point for audio analysis pipeline
"""

import config  # Import FIRST to add Mellow to path

import argparse
from pathlib import Path
from core.pipeline import LTUASPipeline
from core.utils import find_audio_files, validate_audio_file
from config.settings import Config

def main():
    parser = argparse.ArgumentParser(
        description="LTUAS: Joint Speech and Non-Speech Audio Understanding"
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to audio file or directory"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Optional user prompt for guidance"
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Optional reference audio for comparison"
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all audio files in directory"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LTUASPipeline()
    
    # Process based on mode
    audio_path = Path(args.audio)
    
    if args.batch or audio_path.is_dir():
        # Batch mode
        audio_files = find_audio_files(str(audio_path))
        if not audio_files:
            print(f"❌ No audio files found in {audio_path}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        pipeline.process_batch(audio_files, args.prompt)
        
    else:
        # Single file mode
        if not validate_audio_file(str(audio_path)):
            return
        
        pipeline.process_audio(
            str(audio_path),
            args.prompt,
            args.reference
        )

if __name__ == "__main__":
    main()
