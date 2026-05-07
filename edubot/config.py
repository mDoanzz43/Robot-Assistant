"""
config.py — Cấu hình trung tâm EduRobot
Chỉnh file này để tuning toàn bộ hệ thống.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
DB_PATH     = DATA_DIR / "edubot.db"
GRAPH_PATH  = DATA_DIR / "graph" / "knowledge_graph.json"
CHROMA_DIR  = DATA_DIR / "chroma"
LOG_DIR     = DATA_DIR / "logs"
MODELS_DIR  = DATA_DIR / "models"

# ── LLM ────────────────────────────────────────────────────────
# Preferred runtime backend on Jetson: ollama (CPU-safe).
LLM_BACKEND     = os.getenv("LLM_BACKEND", "ollama")   # ollama | llama_cpp
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# Model llm: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
LLM_MODEL_PATH  = r"D:\STUDY\At_school\asr_zipformer\llm\models\qwen\qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_GPU_LAYERS  = 0         # CPU fallback; set >0 only when GPU layers are configured.
LLM_CTX         = 2048
LLM_TEMPERATURE = 0.3       # Thấp → ít hallucinate
LLM_MAX_TOKENS  = 120       # Tăng nhẹ để tránh câu trả lời bị cụt    
LLM_REPEAT_PEN  = 1.1
LLM_N_BATCH     = 512

# ── Embedding ──────────────────────────────────────────────────
# Model tự download khi khởi động lần đầu (~90MB)
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE     = DATA_DIR / "embed_cache"
CHUNK_SIZE      = 150       # tokens per chunk
CHUNK_OVERLAP   = 20

# ── Context Budget (tokens) ───────────────────────────────────
# Anthropic: "context là tài nguyên quý giá"
CTX_SYSTEM      = 80
CTX_KNOWLEDGE   = 300
CTX_HISTORY     = 80
CTX_TASK        = 60
CTX_TOTAL       = CTX_SYSTEM + CTX_KNOWLEDGE + CTX_HISTORY + CTX_TASK  # = 520

# ── Anti-Hallucination ─────────────────────────────────────────
SIM_HIGH        = 0.62      # >= → trả lời tự tin
SIM_LOW         = 0.45      # 0.45–0.62 → trả lời kèm cảnh báo
# < SIM_LOW → "Mình chưa biết" — không gọi LLM

# ── BKT (Bayesian Knowledge Tracing) ──────────────────────────
BKT_P_INIT      = 0.30      # P(biết ban đầu)
BKT_P_TRANSIT   = 0.15      # P(học được sau attempt sai)
BKT_P_GUESS     = 0.25      # P(đoán đúng dù không biết)
BKT_P_SLIP      = 0.10      # P(trả lời sai dù biết)
BKT_MASTERY_THR = 0.70      # Ngưỡng coi là "đã học xong"
BKT_EASY_THR    = 0.40      # < này → câu dễ
BKT_HARD_THR    = 0.70      # > này → câu khó

# ── Session / Workflow ─────────────────────────────────────────
TEACH_TURNS_BEFORE_CONFUSE = 4   # Số lượt dạy trước khi robot "giả ngố"
CONFUSE_TURNS              = 1   # Số lượt confuse trước khi quiz
QUIZ_TOTAL                 = 5   # Số câu quiz mỗi session
QUIZ_BANK_RATIO            = 0.7 # 70% câu từ qa_bank, 30% LLM adaptive
MAX_HISTORY_TURNS          = 3   # Số turns giữ trong working memory

# ── Hardware ───────────────────────────────────────────────────
ESP32_PORT      = os.getenv("ESP32_PORT", "COM5" if os.name == "nt" else "/dev/ttyACM0")
ESP32_BAUD      = 115200
LCD_W           = 240
LCD_H           = 320

# ── Cloud Fallback ─────────────────────────────────────────────
# Gemini fallback is intentionally disabled for local-only tutoring.
ENABLE_CLOUD_FALLBACK = False
CLOUD_CACHE     = DATA_DIR / "cloud_cache.json"

# ── Language ───────────────────────────────────────────────────
LANG            = "vi"          # "vi" | "en"

# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
LOG_FILE        = LOG_DIR / "edubot.log"

# ── ASR — NOTE ─────────────────────────────────────────────────
# Bridge: hardware/asr_bridge.py
# Model dir: set ASR_MODEL_DIR env var hoặc config dưới đây
ASR_MODEL_DIR   = r"D:/STUDY/At_school/asr_zipformer/Zipformer-30M-RNNT-Streaming-6000h"
ASR_SAMPLE_RATE = 16000

# ── TTS — NOTE ─────────────────────────────────────
# Bridge: hardware/tts_bridge.py
# Binary: set PIPER_BIN env var hoặc config dưới đây
PIPER_BIN       = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL     = r"D:\STUDY\At_school\asr_zipformer\nghitts\public\tts-model\vi\ngochuyennew.onnx"
TTS_LENGTH_SCALE = float(os.getenv("TTS_LENGTH_SCALE", "1.1"))
TTS_STORY_LENGTH_SCALE = float(os.getenv("TTS_STORY_LENGTH_SCALE", "1.5"))
# Auto-create dirs
for _d in [DATA_DIR, CHROMA_DIR, LOG_DIR, MODELS_DIR, EMBED_CACHE,
           DATA_DIR / "graph", DATA_DIR / "asr_model", DATA_DIR / "tts_model"]:
    Path(_d).mkdir(parents=True, exist_ok=True)
    
    
# print(PIPER_MODEL)