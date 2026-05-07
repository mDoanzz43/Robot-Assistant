"""
Demo script for Vietnamese TTS
Shows basic usage examples
"""

from tts import VietnameseTTS

def main():
    print("=" * 60)
    print("Vietnamese TTS Demo")
    print("=" * 60)
    print()
    
    # Check if model exists
    import os
    model_path = "../model/ngochuyennew.onnx"
    
    if not os.path.exists(model_path):
        print(f"⚠ Model not found at: {model_path}")
        print("Please download a Piper model or update the model_path")
        print()
        print("You can download models from:")
        print("https://github.com/rhasspy/piper/releases")
        return
    
    print(f"Loading model: {model_path}")
    print()
    
    try:
        # Initialize TTS
        tts = VietnameseTTS(
            model_path=model_path,
            enable_transliteration=True  # Enable transliteration for English words
        )
        
        print()
        print("-" * 60)
        print("Example 1: Simple text")
        print("-" * 60)
        
        text1 = "Xin chào! Hôm nay là ngày 8/3/2026."
        tts.speak(text1, output_path="output1.wav")
        
        print()
        print("-" * 60)
        print("Example 2: Numbers and currency")
        print("-" * 60)
        
        text2 = "Giá sản phẩm là 1.500.000đ, giảm giá 20%."
        tts.speak(text2, output_path="output2.wav")
        
        print()
        print("-" * 60)
        print("Example 3: Dates and times")
        print("-" * 60)
        
        text3 = "Cuộc họp diễn ra vào lúc 15h30 ngày 25/12/2025."
        tts.speak(text3, output_path="output3.wav")
        
        print()
        print("-" * 60)
        print("Example 4: Mixed Vietnamese and English")
        print("-" * 60)
        
        text4 = "Machine learning là một lĩnh vực của AI."
        tts.speak(text4, output_path="output4.wav")
        
        print()
        print("-" * 60)
        print("Example 5: Get audio array (no file)")
        print("-" * 60)
        
        text5 = "Đây là ví dụ trả về audio array."
        audio, sample_rate = tts.speak(text5)
        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
        
        print()
        print("=" * 60)
        print("✓ Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
