import torch
import laion_clap
import numpy as np
from typing import List, Tuple, Dict
from config.settings import Config


class CLAPProcessor:
    """Handles non-speech audio classification using LAION-CLAP with dynamic soft prompt boosting"""

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

    # -------------------------------------------------------
    # NEW: Dynamic contextual weighting from first script
    # -------------------------------------------------------
    def _apply_soft_prompt_dynamic(self, similarity, audio_embed, context_text):
        """
        Boost similarity scores based on contextual text (soft prompt).
        This integrates the dynamic boosting logic from your first script.
        """
        if not context_text:
            return similarity, None

        try:
            # Get text embeddings for categories and context
            text_embed = self.model.get_text_embedding(
                Config.CLAP_SOUND_CATEGORIES,
                use_tensor=True
            )

            context_embed = self.model.get_text_embedding(
                [context_text],
                use_tensor=True
            )

            # Normalize
            text_norm = text_embed / text_embed.norm(dim=-1, keepdim=True)
            ctx_norm = context_embed / context_embed.norm(dim=-1, keepdim=True)

            # Category alignment with context
            ctx_sim = (ctx_norm @ text_norm.T).squeeze()
            ctx_weights = torch.sigmoid(ctx_sim * 5)  # high contrast

            # Center around zero
            ctx_weights = ctx_weights - ctx_weights.mean()

            # Focused boost: top-k categories
            k = 2
            top_vals, top_idx = torch.topk(ctx_weights, k)
            focused = torch.zeros_like(ctx_weights)
            focused[top_idx] = ctx_weights[top_idx]

            boosted_sim = similarity.clone()
            boosted_sim[0] = boosted_sim[0] + (focused * 1.2)

            return boosted_sim, ctx_weights.detach().cpu().numpy()
        except Exception as e:
            print("⚠ Soft prompt boost failed:", e)
            return similarity, None

    # -------------------------------------------------------
    # MAIN PROCESS FUNCTION (IMPROVED)
    # -------------------------------------------------------
    def process(self, audio_path: str, soft_prompt: str = None) -> Dict:
        """
        Process audio file and return sound classifications with optional contextual boost.
        """
        try:
            # Audio embedding
            audio_embed = self.model.get_audio_embedding_from_filelist(
                x=[audio_path],
                use_tensor=True
            )

            # Category text embeddings
            text_embed = self.model.get_text_embedding(
                Config.CLAP_SOUND_CATEGORIES,
                use_tensor=True
            )

            # Base similarity
            base_similarity = audio_embed @ text_embed.t()

            # Apply dynamic context boost
            boosted_similarity, ctx_weights = self._apply_soft_prompt_dynamic(
                base_similarity,
                audio_embed,
                soft_prompt
            )

            # ReLU normalization (matches first script)
            raw = boosted_similarity.detach().cpu().numpy()[0]
            probs = np.maximum(raw, 0)
            probs = probs / (probs.sum() + 1e-8)

            ranked = sorted(
                zip(Config.CLAP_SOUND_CATEGORIES, probs),
                key=lambda x: x[1],
                reverse=True
            )

            result = {
                "dominant_sound": ranked[0][0],
                "dominant_confidence": float(ranked[0][1]),
                "top_sounds": [
                    {"sound": s, "confidence": float(p)} for s, p in ranked[:5]
                ],
                "all_scores": {s: float(p) for s, p in ranked},
                "context_weights": ctx_weights.tolist() if ctx_weights is not None else None
            }

            return result

        except Exception as e:
            print(f"❌ CLAP processing error: {e}")
            return {
                "error": str(e),
                "dominant_sound": "unknown",
                "top_sounds": []
            }

    # -------------------------------------------------------
    # GENERATE SOFT PROMPT (IMPROVED)
    # -------------------------------------------------------
    def generate_soft_prompt(self, clap_result: Dict) -> str:
        """Generate dynamic natural-language soft prompt"""
        if "error" in clap_result:
            return "Audio classification unavailable"

        top = clap_result["top_sounds"][:3]
        desc = ", ".join(f"{i['sound']} ({i['confidence']:.1%})" for i in top)

        prompt = f"Non-speech audio detected: {desc}"

        # Strengthen prompt if confidence is high
        if clap_result["dominant_confidence"] > 0.7:
            prompt += f". Dominant category: {clap_result['dominant_sound']}."

        return prompt
