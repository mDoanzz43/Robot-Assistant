# Nếu dùng PC thì làm theo hướng dẫn sau
## Chỉ setup khi dùng PC - còn trên jetson nano thì đã cài rồi

## Step1: Cài ASR
https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h/tree/main
-> cài requirement (kèm sherpa-onnx)

Cách test bằng file infer_zipformer.py 
## Step2: Cài nghitts (TTS)
https://github.com/nghimestudio/nghitts.git
Cách 1: Nếu muốn test theo chính chủ repo thì cài requirement theo họ 

Cách 2: Cài theo python: requirement trên PC như sau:
piper-tts>=1.2.0
onnxruntime>=1.16.0
numpy>=1.24.0
thêm phonemizer-1.2.2.tgz
(folder python_tts là convert code từ repo của họ ra module python - có thể chưa đủ hết chức năng nhưng vẫn đủ để test
Cần copy các file python từ vendor\nghitts_python_tts vào nghitts/python_tts
)

## Step 3: Cài llm phụ thuộc vào máy 

