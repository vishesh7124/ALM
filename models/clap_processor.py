import torch
import laion_clap
import numpy as np
from typing import List, Tuple, Dict
from config.settings import Config

class CLAPProcessor:
    """Handles non-speech audio classification using LAION-CLAP"""
    
    def __init__(self):
        self.device = "cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load CLAP model once at initialization"""
        print("Loading CLAP model...")
        self.model = laion_clap.CLAP_Module(
            enable_fusion=Config.CLAP_CONFIG["enable_fusion"]
        )
        self.model.load_ckpt()
        print(f"✓ CLAP model loaded on {self.device}")
    
    def process(self, audio_path: str) -> Dict:
        """
        Process audio file and return sound classifications
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with classifications and confidence scores
        """
        try:
            # Get embeddings
            audio_embed = self.model.get_audio_embedding_from_filelist(
                x=[audio_path], 
                use_tensor=True
            )
            text_embed = self.model.get_text_embedding(
                Config.CLAP_SOUND_CATEGORIES, 
                use_tensor=True
            )
            
            # Calculate similarity
            similarity = audio_embed @ text_embed.t()
            
            # Convert to probabilities
            probs = torch.softmax(similarity, dim=-1).detach().cpu().numpy()[0]
            
            # Rank by confidence
            ranked_sounds = sorted(
                zip(Config.CLAP_SOUND_CATEGORIES, probs),
                key=lambda x: x[1],
                reverse=True
            )
            
            
            # Build result
            result = {
                "top_sounds": [
                    {"sound": sound, "confidence": float(conf)}
                    for sound, conf in ranked_sounds[:5]
                ],
                "all_scores": {
                    sound: float(conf) 
                    for sound, conf in ranked_sounds
                },
                "dominant_sound": ranked_sounds[0][0],
                "dominant_confidence": float(ranked_sounds[0][1])
            }
            
            return result
            
        except Exception as e:
            print(f"❌ CLAP processing error: {e}")
            return {
                "error": str(e),
                "top_sounds": [],
                "dominant_sound": "unknown"
            }
    
    def generate_soft_prompt(self, clap_result: Dict) -> str:
        """Convert CLAP results to natural language soft prompt"""
        if "error" in clap_result:
            return "Audio classification unavailable"
        
        top_sounds = clap_result["top_sounds"][:3]
        
        # Build descriptive prompt
        sound_descriptions = [
            f"{s['sound']} ({s['confidence']:.1%})"
            for s in top_sounds
        ]
        
        prompt = f"Non-speech audio detected: {', '.join(sound_descriptions)}"
        
        # Add context based on dominant sound
        dominant = clap_result["dominant_sound"]
        if clap_result["dominant_confidence"] > 0.7:
            prompt += f". Primarily {dominant}."
        
        return prompt
