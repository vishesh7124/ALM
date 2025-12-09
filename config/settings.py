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
        "model_name": "630k-audioset-best.pt",
        "temperature": 0.2,  # Default checkpoint
    }
    
    WHISPER_CONFIG = {
        "model": "whisper-large-v3",
        "temperature": 0,
        "response_format": "verbose_json",
    }
    
    LLM_CONFIG = {
        "model": "llama-3.3-70b-versatile",  # Faster than gpt-oss-120b
        "temperature": 0.7,
        "max_tokens": 512,
        "reasoning_effort": "medium",
    }
    
    MELLOW_CONFIG = {
        "config": "v0",
        "model": "v0",
        "max_len": 400,
        "top_p": 0.7,
        "temperature": 0.6,
    }
    
    # CLAP sound categories (expand as needed)
    CLAP_SOUND_CATEGORIES = [
        # Human sounds
        "speech",
        # Nature sounds
        "wind",
        "female",
        # Action sounds
        "weapon sounds",
        # Musical
        "footsteps",
        # Ambient
        "beeps",
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
