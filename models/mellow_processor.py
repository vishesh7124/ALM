# models/mellow_processor.py
import config  # Import first to add Mellow to path

import torch
from pathlib import Path
from mellow import MellowWrapper
from typing import List, Dict, Optional
from config.settings import Config


class MELLOWProcessor:
    """Handles audio reasoning using MELLOW model with full debug output"""
    
    def __init__(self):
        print("\n================= MELLOW PROCESSOR INIT =================")
        self.device = 0 if Config.USE_CUDA and torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        print(f"[DEBUG] USE_CUDA flag: {Config.USE_CUDA}")
        print(f"[DEBUG] Selected device: {self.device}")

        self.model = None
        self._load_model()
        print("==========================================================\n")

    
    def _load_model(self):
        """Load MELLOW model once at initialization"""
        print("\n[DEBUG] Loading MELLOW model...")
        print("[DEBUG] Config.MELLOW_CONFIG =", Config.MELLOW_CONFIG)

        try:
            self.model = MellowWrapper(
                config=Config.MELLOW_CONFIG["config"],
                model=Config.MELLOW_CONFIG["model"],
                device=self.device,
                use_cuda=Config.USE_CUDA and torch.cuda.is_available(),
            )

            print(f"[OK] MELLOW model loaded successfully")
            print(f"[DEBUG] Model config path: {Config.MELLOW_CONFIG['config']}")
            print(f"[DEBUG] Model weights path: {Config.MELLOW_CONFIG['model']}")
            print(f"[DEBUG] Model running on device: {self.device}")

        except Exception as e:
            print("[X] ERROR: Failed to load MELLOW model:", str(e))
            import traceback
            traceback.print_exc()
            raise e


    def process(
        self, 
        audio_path: str, 
        soft_prompt: str,
        reference_audio: Optional[str] = None
    ) -> Dict:

        print("\n================= MELLOW PROCESS START =================")
        print("[DEBUG] Received arguments:")
        print(f"        audio_path       = {audio_path}")
        print(f"        soft_prompt      = {soft_prompt}")
        print(f"        reference_audio  = {reference_audio}")
        
        # Validate audio paths
        print("\n[DEBUG] Checking audio file existence...")
        if not Path(audio_path).exists():
            print(f"[X] ERROR: audio_path does NOT exist: {audio_path}")
        else:
            print(f"[DEBUG] audio_path found OK: {audio_path}")

        if reference_audio:
            if not Path(reference_audio).exists():
                print(f"[X] ERROR: reference_audio does NOT exist: {reference_audio}")
            else:
                print(f"[DEBUG] reference_audio found OK: {reference_audio}")

        try:
            print("\n[DEBUG] Constructing 'examples' list for MellowWrapper.generate()...")
            
            if reference_audio:
                examples = [[audio_path, reference_audio, soft_prompt]]
                print("[DEBUG] Using real reference audio:")
            else:
                examples = [[audio_path, audio_path, soft_prompt]]
                print("[DEBUG] No reference audio provided — using same audio twice:")

            print(f"[DEBUG] examples = {examples}")

            print("\n[DEBUG] Calling self.model.generate() with:")
            print(f"        max_len    = {Config.MELLOW_CONFIG['max_len']}")
            print(f"        top_p       = {Config.MELLOW_CONFIG['top_p']}")
            print(f"        temperature = {Config.MELLOW_CONFIG['temperature']}")
            print(f"        device      = {self.device}")

            response = self.model.generate(
                examples=examples,
                max_len=Config.MELLOW_CONFIG["max_len"],
                top_p=Config.MELLOW_CONFIG["top_p"],
                temperature=Config.MELLOW_CONFIG["temperature"],
            )

            print("\n[DEBUG] Raw model response received:")
            print("--------------------------------------------------------")
            print(response)
            print("--------------------------------------------------------")

            print("================= MELLOW PROCESS END =================\n")

            return {
                "inference": response,
                "soft_prompt_used": soft_prompt,
                "examples_used": examples,
                "success": True
            }

        except Exception as e:
            print("\n[X] MELLOW processing error:", str(e))
            print("[DEBUG] Full traceback:")
            import traceback
            traceback.print_exc()

            print("================= MELLOW PROCESS FAILED =================\n")

            return {
                "error": str(e),
                "inference": "",
                "success": False,
                "examples_used": examples if 'examples' in locals() else None
            }
