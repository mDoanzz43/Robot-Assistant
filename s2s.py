import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import ollama
import sherpa_onnx
import sounddevice as sd

# Add nghitts/python_tts path
sys.path.insert(0, str(Path(__file__).parent / "nghitts" / "python_tts"))

try:
    from tts import VietnameseTTS

    TTS_AVAILABLE = True
except ImportError:
    print("Warning: nghitts/python_tts not found. TTS disabled.")
    TTS_AVAILABLE = False


# ================= CONFIG =================

MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\Zipformer-30M-RNNT-Streaming-6000h"
NGHITTS_MODEL_PATH = (
    r"D:\STUDY\At_school\asr_zipformer\nghitts\public\tts-model\vi\deepman3909.onnx"
)

SAMPLE_RATE = 16000
CHUNK_MS = 16
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

# Ollama config
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "-1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": 512,
    "num_predict": 80,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.1,
}

SYSTEM_PROMPT = (
    "Bạn là Kitty, một trợ lý ảo AI thân thiện và thông minh, có chức năng trò chuyện và giao tiếp với con người bằng tiếng Việt. "
    "Chỉ trả lời bằng tiếng Việt một cách đơn giản và chính xác. "
    "Nếu không biết câu trả lời, hãy nói rằng \"Tôi không biết, bạn có thể công cấp thêm thông tin không?\" thay vì đoán hoặc tạo ra thông tin sai lệch."
)
MAX_HISTORY_TURNS = 6
DUPLICATE_QUERY_WINDOW_SEC = 1.5


# ================= ASSISTANT =================

class VoiceAssistant:

    def __init__(self):
        print("Loading ASR...")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=f"{MODEL_DIR}/encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            decoder=f"{MODEL_DIR}/decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            joiner=f"{MODEL_DIR}/joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            tokens=f"{MODEL_DIR}/config.json",
            sample_rate=SAMPLE_RATE,
            enable_endpoint_detection=True,
        )

        self.stream = self.recognizer.create_stream()

        print("Loading Ollama client...")
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        print(f"Using Ollama host: {OLLAMA_HOST}")
        print(f"Using Ollama model: {OLLAMA_MODEL}")

        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_user_query = ""
        self.last_user_query_ts = 0.0

        if TTS_AVAILABLE:
            print("Loading NGHITTS...")
            try:
                self.tts = VietnameseTTS(
                    model_path=NGHITTS_MODEL_PATH,
                    enable_transliteration=True,
                )
                self.tts_enabled = True
                print("NGHITTS loaded")
            except Exception as e:
                print(f"NGHITTS load failed: {e}")
                self.tts_enabled = False
        else:
            self.tts_enabled = False

        print("System ready.\n")

    def trim_history(self) -> None:
        # Keep system prompt + recent turns to reduce latency.
        max_messages = MAX_HISTORY_TURNS * 2 + 1
        if len(self.conversation_history) > max_messages:
            self.conversation_history = [
                self.conversation_history[0],
                *self.conversation_history[-(max_messages - 1):],
            ]

    # ================= LLM =================

    def ask_llm(self, text: str) -> str:
        self.conversation_history.append({"role": "user", "content": text})
        self.trim_history()

        try:
            response = self.ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=self.conversation_history,
                stream=False,
                options=OLLAMA_OPTIONS,
            )
            content = response.get("message", {}).get("content", "").strip()
            final_response = content or "Kitty se tim hieu them nhe."
            self.conversation_history.append({"role": "assistant", "content": final_response})
            self.trim_history()
            return final_response
        except Exception as e:
            print(f"Ollama error: {e}")
            fallback = "Kitty dang gap loi ket noi, ban thu lai nhe."
            self.conversation_history.append({"role": "assistant", "content": fallback})
            self.trim_history()
            return fallback

    # ================= TTS =================

    def speak(self, text: str) -> None:
        if not self.tts_enabled:
            return

        try:
            self.is_speaking = True

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            print("Speaking (nghitts)...")

            audio, sample_rate = self.tts.speak(
                text,
                output_path=None,
                length_scale=1.0,
                preprocess=True,
            )

            if audio is None or len(audio) == 0:
                print("Warning: Empty TTS audio")
                self.is_speaking = False
                return

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            sd.play(audio, sample_rate)
            sd.wait()

            print("Done speaking")

            time.sleep(0.4)

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            self.stream = self.recognizer.create_stream()
            self.is_speaking = False

            print("Ready\n")

        except Exception as e:
            print(f"TTS error: {e}")
            self.is_speaking = False

    # ================= AUDIO CALLBACK =================

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if not self.is_speaking:
            self.audio_queue.put(indata.copy())

    # ================= PIPELINE =================

    def process_query(self, text: str) -> None:
        cleaned = text.strip().lower()
        if not cleaned:
            return

        now = time.time()
        if (
            cleaned == self.last_user_query
            and now - self.last_user_query_ts <= DUPLICATE_QUERY_WINDOW_SEC
        ):
            print(f"[skip] duplicate query: {cleaned}")
            return

        self.last_user_query = cleaned
        self.last_user_query_ts = now

        print(f"\nUser: {cleaned}")
        response = self.ask_llm(cleaned)
        print(f"LLM: {response}")
        self.speak(response)

    # ================= MAIN LOOP =================

    def run_voice_mode(self) -> None:
        print("Mode 1: noi -> llm -> speaker")
        print("Speak now... (Ctrl+C to stop)")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            dtype="float32",
            callback=self.audio_callback,
        ):
            while True:
                if not self.is_speaking and not self.audio_queue.empty():
                    audio = self.audio_queue.get().squeeze()

                    self.stream.accept_waveform(SAMPLE_RATE, audio.tolist())

                    while self.recognizer.is_ready(self.stream):
                        self.recognizer.decode_stream(self.stream)

                    if self.recognizer.is_endpoint(self.stream):
                        final_text = self.recognizer.get_result(self.stream).strip()

                        if final_text:
                            self.process_query(final_text)
                        # Always reset stream after an endpoint to avoid repeating
                        # the same finalized segment in the next loop.
                        self.stream = self.recognizer.create_stream()

                time.sleep(0.01)

    def run_text_mode(self) -> None:
        print("Mode 2: nhap text -> llm -> speaker")
        print("Nhap 'exit' de thoat")
        while True:
            user_input = input("Ban: ").strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                break
            self.process_query(user_input)

    def run(self) -> None:
        print("Chon mode:")
        print("  (1): noi - llm - speaker")
        print("  (2): nhap text - llm - speaker")
        mode = input("Nhap 1 hoac 2: ").strip()

        if mode == "1":
            self.run_voice_mode()
            return
        if mode == "2":
            self.run_text_mode()
            return

        print("Mode khong hop le. Mac dinh chay mode (1).")
        self.run_voice_mode()


# ================= MAIN =================

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
