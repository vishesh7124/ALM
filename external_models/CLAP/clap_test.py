import laion_clap
import torch

# Load model ONCE at startup
print("Loading CLAP model...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()
print("Model loaded!")

# Reuse the model multiple times
def process_audio(audio_path, text_queries):
    audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
    text_embed = model.get_text_embedding(text_queries, use_tensor=True)
    
    similarity = audio_embed @ text_embed.t()
    return similarity

# Call process_audio many times without reloading
result1 = process_audio('assets/2.wav', ["dog barking", "rain", "gunshots"])

print(result1)
