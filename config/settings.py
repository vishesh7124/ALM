import os
from pathlib import Path
from dotenv import load_dotenv

# Import path configuration FIRST
from config.paths import EXTERNAL_MODELS_DIR  # This adds models to sys.path

load_dotenv()

class Config:
    """Central configuration for LTUAS system"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    EXTERNAL_MODELS_DIR = EXTERNAL_MODELS_DIR
    OUTPUT_DIR = BASE_DIR / "outputs" / "results"
    RESOURCE_DIR = BASE_DIR / "resources" / "audio"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model configurations
    CLAP_CONFIG = {
        "enable_fusion": False,  # False for <10sec audio
        "model_name": "630k-audioset-best.pt",  # Default checkpoint
    }
    
    WHISPER_CONFIG = {
        "model": "whisper-large-v3",
        "temperature": 0,
        "response_format": "verbose_json",
    }
    
    LLM_CONFIG = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",  # Faster than gpt-oss-120b
        "temperature": 0.7,
        "max_tokens": 512,
        "reasoning_effort": "medium",
    }
    
    MELLOW_CONFIG = {
        "config": "v0",
        "model": "v0",
        "max_len": 300,
        "top_p": 0.8,
        "temperature": 1.0,
    }
    
    # CLAP sound categories (expand as needed)
    CLAP_SOUND_CATEGORIES = [
        # Human sounds
        "speech", "shouting", "crying", "laughing", "coughing",
        # Nature sounds
        "rain", "thunder", "wind", "ocean waves", "birds chirping",
        # Urban sounds
        "car engine", "car horn", "siren", "construction noise", "traffic",
        # Action sounds
        "gunshots", "explosions", "footsteps", "door slam", "glass breaking",
        # Musical
        "music", "drums", "guitar", "piano", "violin",
        # Ambient
        "silence", "white noise", "crowd noise", "applause",
    ]
    
    # Processing settings
    USE_CUDA = True
    NUM_WORKERS = 2  # For parallel processing
    ENABLE_CACHING = True
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize
Config.ensure_dirs()
