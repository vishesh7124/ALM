# models/mellow_processor.py
import config  # Import first to add Mellow to path

import torch
from pathlib import Path
from mellow import MellowWrapper
from typing import List, Dict, Optional
from config.settings import Config

class MELLOWProcessor:
    """Handles audio reasoning using MELLOW model"""
    
    def __init__(self):
        self.device = 0 if Config.USE_CUDA and torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load MELLOW model once at initialization"""
        print("Loading MELLOW model...")
        self.model = MellowWrapper(
            config=Config.MELLOW_CONFIG["config"],
            model=Config.MELLOW_CONFIG["model"],
            device=self.device,
            use_cuda=Config.USE_CUDA and torch.cuda.is_available(),
        )
        print(f"[OK] MELLOW model loaded on device {self.device}")
    
    def process(
        self, 
        audio_path: str, 
        soft_prompt: str,
        reference_audio: Optional[str] = None
    ) -> Dict:
        """Generate audio understanding using MELLOW"""
        try:
            # FIXED: Use same audio for both paths if no reference provided
            if reference_audio:
                examples = [[audio_path, reference_audio, soft_prompt]]
            else:
                # Use the same audio file twice (Mellow expects 2 audio paths)
                examples = [[audio_path, audio_path, soft_prompt]]
            
            response = self.model.generate(
                examples=examples,
                max_len=Config.MELLOW_CONFIG["max_len"],
                top_p=Config.MELLOW_CONFIG["top_p"],
                temperature=Config.MELLOW_CONFIG["temperature"],
            )
            
            return {
                "inference": response,
                "soft_prompt_used": soft_prompt,
                "success": True
            }
            
        except Exception as e:
            print(f"[X] MELLOW processing error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "inference": "",
                "success": False
            }
