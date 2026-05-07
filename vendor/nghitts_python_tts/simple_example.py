"""
Simple Example - Vietnamese TTS
"""

from tts import VietnameseTTS

tts = VietnameseTTS(
    model_path="D:\\STUDY\\At_school\\asr_zipformer\\nghitts\\public\\tts-model\\vi\\deepman3909.onnx",  # Update this path
    enable_transliteration=True
)

# Example 1: Simple text
print("Example 1: Simple greeting")
tts.speak("Xin chào các bạn! Tôi là một trợ lý ảo AI có chức năng trò chuyện và giao tiếp với các bạn", output_path="hello.wav")


print("Example 2: Simple greeting")
text_1 = "Câu 1: Mồm bò mà không phải mồm bò. Đó là con gì. Đáp án: Là Con ốc sên"
tts.speak(text_1, output_path="hello1.wav")

# Example 2: Numbers and dates
print("\nExample 2: Numbers and dates")
text = "Hôm nay là ngày 8/3/2026. Tôi có 1.500.000đ. Tao thách chúng mày đấy, mày dám nói tao là đồ chó không?"
tts.speak(text, output_path="numbers.wav")

# Example 3: Get audio array instead of file
print("\nExample 3: Get audio array")
audio, sample_rate = tts.speak("Đây là ví dụ trả về audio array")
print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate} Hz")

print("\n✓ Done! Check the output WAV files.")
