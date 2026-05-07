"""
Ví dụ điều chỉnh chất lượng giọng nói và tốc độ đọc
"""

from tts import VietnameseTTS

# Khởi tạo TTS
tts = VietnameseTTS(
    model_path="D:\\STUDY\\At_school\\nghitts\\public\\tts-model\\vi\\ngochuyennew.onnx",
    enable_transliteration=True
)

text = "Xin chào! Hôm nay là ngày 8 tháng 3 năm 2026. Giá sản phẩm là 1.500.000 đồng."

print("=" * 60)
print("So sánh các tham số điều chỉnh")
print("=" * 60)

# ==========================================
# 1. MẶC ĐỊNH (tốc độ bình thường)
# ==========================================
print("\n1. Mặc định (length_scale=1.0)")
tts.speak(
    text, 
    output_path="default.wav",
    length_scale=1.0   # Noise mặc định
)

# ==========================================
# 2. CHẬM HƠN - DỄ NGHE HƠN (Khuyến nghị)
# ==========================================
print("\n2. Chậm hơn - Dễ nghe (length_scale=1.15)")
tts.speak(
    text, 
    output_path="slow_clear.wav",
    length_scale=1.15
)

# ==========================================
# 3. CHẬM HƠN NỮA - RẤT RÕ RÀNG
# ==========================================
print("\n3. Rất chậm - Rất rõ ràng (length_scale=1.3)")
tts.speak(
    text, 
    output_path="very_slow.wav",
    length_scale=1.3     # Chậm hơn 30% → phù hợp cho học tập
)

# ==========================================
# 4. NHANH HƠN - NĂNG ĐỘNG
# ==========================================
print("\n4. Nhanh hơn - Năng động (length_scale=0.85)")
tts.speak(
    text, 
    output_path="fast.wav",
    length_scale=0.85   # Nhanh hơn 15%

)

# ==========================================
# 5. ĐIỀU CHỈNH NOISE (ảnh hưởng đến độ mượt)
# ==========================================
print("\n5. Giảm noise - Giọng mượt hơn (noise_scale=0.333)")
tts.speak(
    text, 
    output_path="smooth.wav",
    length_scale=1.15    # Chậm vừa phải
         # Ít noise → giọng mượt hơn, ít biến đổi
)

print("\n6. Tăng noise - Giọng tự nhiên hơn (noise_scale=0.9)")
tts.speak(
    text, 
    output_path="natural.wav",
    length_scale=1.15      # Nhiều noise → giọng tự nhiên, có biến đổi
)

print("\n" + "=" * 60)
print("✓ Đã tạo 6 file WAV với các cài đặt khác nhau")
print("✓ Nghe thử và chọn cài đặt phù hợp nhất!")
print("=" * 60)

print("""
KHUYẾN NGHỊ:
- Giọng tự nhiên, dễ nghe: length_scale=1.15, noise_scale=0.667
- Giọng rõ ràng cho học tập: length_scale=1.3, noise_scale=0.5
- Giọng năng động: length_scale=0.9, noise_scale=0.7

TÙY CHỈNH:
- length_scale: 
  * < 1.0 = nhanh hơn (0.8 = nhanh 20%)
  * > 1.0 = chậm hơn (1.2 = chậm 20%)
  
- noise_scale:
  * 0.1-0.3 = giọng rất mượt, đều đặn
  * 0.5-0.7 = cân bằng (khuyến nghị)
  * 0.8-1.0 = giọng tự nhiên, nhiều biến đổi
""")
