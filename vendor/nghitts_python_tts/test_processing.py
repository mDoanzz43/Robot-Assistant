"""
Test Vietnamese Text Processing
Tests the text preprocessing without TTS
"""

from text_cleaner import process_text_for_tts
from vietnamese_processor import number_to_words

print("=" * 60)
print("Vietnamese Text Processing Tests")
print("=" * 60)
print()

# Test cases
test_cases = [
    "Số 123",
    "Ngày 8/3/2026",
    "Lúc 15h30",
    "Giá 1.500.000đ",
    "Giảm 50%",
    "Khoảng cách 5km",
    "Nhiệt độ 25°C",
    "Machine learning là AI",
    "Cuộc họp từ 25-26/3/2026",
    "Tăng trưởng 3,5%",
]

for i, text in enumerate(test_cases, 1):
    processed = process_text_for_tts(text, enable_transliteration=True)
    print(f"{i}. Input:  {text}")
    print(f"   Output: {processed}")
    print()

print("=" * 60)
print("Number to Words Tests")
print("=" * 60)
print()

numbers = ["0", "15", "123", "1000", "1500000", "2026"]
for num in numbers:
    words = number_to_words(num)
    print(f"{num:>10} → {words}")

print()
print("✓ All tests completed!")
