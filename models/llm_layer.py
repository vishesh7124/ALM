from groq import Groq
from typing import Dict
from typing import Optional
from config.settings import Config

class LLMLayer:
    """Converts raw CLAP/Whisper outputs to unified soft prompts for MELLOW"""
    
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        print("✓ LLM layer initialized")
    
    def convert_to_soft_prompt(
        self, 
        clap_prompt: str, 
        whisper_prompt: str,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Synthesize CLAP and Whisper outputs into unified soft prompt
        
        Args:
            clap_prompt: CLAP soft prompt
            whisper_prompt: Whisper soft prompt
            user_prompt: Optional user guidance
            
        Returns:
            Unified soft prompt for MELLOW
        """
        from prompts.templates import LLM_CONVERSION_PROMPT
        
        # Build context
        context = f"""
CLAP Analysis (Non-Speech): {clap_prompt}

Whisper Analysis (Speech): {whisper_prompt}
"""
        if user_prompt:
            context += f"\nUser Guidance: {user_prompt}"
        
        # Generate unified prompt
        try:
            completion = self.client.chat.completions.create(
                model=Config.LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": LLM_CONVERSION_PROMPT},
                    {"role": "user", "content": context}
                ],
                temperature=Config.LLM_CONFIG["temperature"],
                max_tokens=Config.LLM_CONFIG["max_tokens"],
                stream=False,  # Use streaming=False for simplicity
            )
            
            soft_prompt = completion.choices[0].message.content.strip()
            return soft_prompt
            
        except Exception as e:
            print(f"❌ LLM layer error: {e}")
            # Fallback: simple concatenation
            return f"{clap_prompt}. {whisper_prompt}"
    
    def convert_streaming(
        self, 
        clap_prompt: str, 
        whisper_prompt: str,
        user_prompt: Optional[str] = None
    ):
        """Streaming version for real-time applications"""
        from prompts.templates import LLM_CONVERSION_PROMPT
        
        context = f"""
CLAP Analysis: {clap_prompt}
Whisper Analysis: {whisper_prompt}
"""
        if user_prompt:
            context += f"\nUser Guidance: {user_prompt}"
        
        try:
            stream = self.client.chat.completions.create(
                model=Config.LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": LLM_CONVERSION_PROMPT},
                    {"role": "user", "content": context}
                ],
                temperature=Config.LLM_CONFIG["temperature"],
                max_tokens=Config.LLM_CONFIG["max_tokens"],
                stream=True,
            )
            
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as e:
            print(f"❌ LLM streaming error: {e}")
            yield f"{clap_prompt}. {whisper_prompt}"
