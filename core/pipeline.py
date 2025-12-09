import config  # Import FIRST to add Mellow to path

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List
from datetime import datetime

from models.clap_processor import CLAPProcessor
from models.whisper_processor import WhisperProcessor
from models.llm_layer import LLMLayer
from models.mellow_processor import MELLOWProcessor
from config.settings import Config

class LTUASPipeline:
    """Main orchestration pipeline for LTUAS system"""
    
    def __init__(self):
        print("=" * 60)
        print("Initializing LTUAS Pipeline...")
        print("=" * 60)
        
        # Initialize all processors
        self.clap = CLAPProcessor()
        self.whisper = WhisperProcessor()
        self.llm = LLMLayer()
        self.mellow = MELLOWProcessor()
        
        print("=" * 60)
        print("✓ All models loaded successfully")
        print("=" * 60 + "\n")
    
    def process_audio(
        self, 
        audio_path: str,
        user_prompt: Optional[str] = None,
        reference_audio: Optional[str] = None
    ) -> Dict:
        """
        Process audio through full LTUAS pipeline
        
        Args:
            audio_path: Path to audio file
            user_prompt: Optional user guidance
            reference_audio: Optional second audio for comparison
            
        Returns:
            Complete JSON inference
        """
        start_time = time.time()
        audio_path = str(Path(audio_path).resolve())
        
        print(f"\n{'='*60}")
        print(f"Processing: {Path(audio_path).name}")
        print(f"{'='*60}\n")
        
        # STAGE 1: Parallel CLAP and Whisper processing
        print("Stage 1: Parallel feature extraction...")
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            future_clap = executor.submit(self.clap.process, audio_path)
            future_whisper = executor.submit(self.whisper.process, audio_path)
            
            clap_result = future_clap.result()
            whisper_result = future_whisper.result()
        
        print(f"  ✓ CLAP: {clap_result['dominant_sound']} ({clap_result.get('dominant_confidence', 0):.1%})")
        print(f"  ✓ Whisper: {'Speech detected' if whisper_result['has_speech'] else 'No speech'}")
        
        # STAGE 2: Generate soft prompts
        print("\nStage 2: Generating soft prompts...")
        clap_soft_prompt = self.clap.generate_soft_prompt(clap_result)
        whisper_soft_prompt = self.whisper.generate_soft_prompt(whisper_result)
        
        print(f"  CLAP prompt: {clap_soft_prompt}")
        print(f"  Whisper prompt: {whisper_soft_prompt}")
        
        # STAGE 3: LLM layer synthesis
        print("\nStage 3: LLM layer synthesis...")
        unified_soft_prompt = self.llm.convert_to_soft_prompt(
            clap_soft_prompt,
            whisper_soft_prompt,
            user_prompt
        )

        system_prompt = f" produce a concise analysis covering: high-level summary: {unified_soft_prompt}"

        print(f"  Unified prompt: {unified_soft_prompt}")
        
        # STAGE 4: MELLOW reasoning
        print("\nStage 4: MELLOW reasoning...")
        mellow_result = self.mellow.process(
            audio_path,
            system_prompt,
            reference_audio
        )
        print(f"  ✓ Generated {len(mellow_result.get('inference', ''))} chars")
        
        # Build final JSON output
        output = {
            "metadata": {
                "audio_file": str(Path(audio_path).name),
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "user_prompt": user_prompt,
            },
            "clap_inf": clap_result,
            "speech_inf": whisper_result,
            "mellow_inf": mellow_result,
            "soft_prompts": {
                "clap": clap_soft_prompt,
                "whisper": whisper_soft_prompt,
                "unified": unified_soft_prompt,
            }
        }
        
        # Save to file
        self._save_output(output, audio_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Complete! Total time: {output['metadata']['processing_time_seconds']}s")
        print(f"{'='*60}\n")
        
        return output
    
    def _save_output(self, output: Dict, audio_path: str):
        """Save JSON output to file"""
        audio_name = Path(audio_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Config.OUTPUT_DIR / f"{audio_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Output saved: {output_file}")
    
    def process_batch(self, audio_files: List[str], user_prompt: Optional[str] = None):
        """Process multiple audio files sequentially"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(audio_files)} files")
        print(f"{'='*60}\n")
        
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing {Path(audio_path).name}...")
            result = self.process_audio(audio_path, user_prompt)
            results.append(result)
        
        print(f"\n{'='*60}")
        print(f"✓ Batch complete: {len(results)} files processed")
        print(f"{'='*60}\n")
        
        return results
