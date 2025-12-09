# test_mellow.py
from models.mellow_processor import MELLOWProcessor

if __name__ == "__main__":
    # ---- CONFIGURE THIS ----
    audio_path = "resources/audio/test_audio.wav"   # path to any audio file you want to test
    soft_prompt = "are there gunshots?"  # prompt for mellow
    
    # -------------------------

    print("Initializing MELLOWProcessor...\n")
    mellow = MELLOWProcessor()
    print(soft_prompt)
    print("\nRunning inference...\n")
    result = mellow.process(
        audio_path=audio_path,
        soft_prompt=soft_prompt,
        reference_audio=None  # we want to test WITHOUT reference audio
    )

    print("\n========== MELLOW OUTPUT ==========\n")
    print("Success:", result.get("success"))
    print("Soft Prompt:", result.get("soft_prompt_used"))
    print("Inference:\n", result.get("inference"))
    print("\n===================================\n")
