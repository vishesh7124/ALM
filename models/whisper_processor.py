import os
from groq import Groq
from typing import Dict, Optional
from config.settings import Config

class WhisperProcessor:
    """Handles speech transcription using Groq Whisper API"""
    
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        print("✓ Whisper processor initialized")
    
    def process(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription and metadata
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), audio_file.read()),
                    model=Config.WHISPER_CONFIG["model"],
                    temperature=Config.WHISPER_CONFIG["temperature"],
                    response_format=Config.WHISPER_CONFIG["response_format"],
                )
            
            # Extract relevant fields from verbose_json
            result = {
                "text": transcription.text,
                "language": getattr(transcription, 'language', 'unknown'),
                "duration": getattr(transcription, 'duration', 0),
                "segments": getattr(transcription, 'segments', []),
                "has_speech": len(transcription.text.strip()) > 0,
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Whisper processing error: {e}")
            return {
                "error": str(e),
                "text": "",
                "has_speech": False,
                "language": "unknown"
            }
    
    def generate_soft_prompt(self, whisper_result: Dict) -> str:
        """Convert Whisper results to natural language soft prompt"""
        if "error" in whisper_result:
            return "Speech transcription unavailable"
        
        if not whisper_result["has_speech"]:
            return "No speech detected in audio"
        
        text = whisper_result["text"]
        language = whisper_result["language"]
        
        prompt = f"Speech transcription ({language}): \"{text}\""
        
        return prompt
