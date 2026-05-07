import os
import queue
import sys
import time
import threading
from pathlib import Path

import numpy as np
import ollama
import sherpa_onnx
import sounddevice as sd

# ================= PATH =================

sys.path.insert(0, str(Path(__file__).parent / "nghitts" / "python_tts"))

try:
    from tts import VietnameseTTS
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: TTS not found")
    TTS_AVAILABLE = False

# ================= CONFIG =================

MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\Zipformer-30M-RNNT-Streaming-6000h"
NGHITTS_MODEL_PATH = r"D:\STUDY\At_school\asr_zipformer\nghitts\public\tts-model\vi\deepman3909.onnx"

SAMPLE_RATE = 16000
CHUNK_MS = 16
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

OLLAMA_MODEL = "qwen2.5:0.5b"

OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": 512,
    "num_predict": 60,  # 👈 giảm để realtime
}

SYSTEM_PROMPT = (
    "Bạn là Kitty, trợ lý AI thân thiện. Trả lời ngắn gọn, tự nhiên, dưới 50 từ.",
    "Hãy trả lời"
)

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

        print("Loading Ollama...")
        self.ollama_client = ollama.Client()

        self.audio_queue = queue.Queue()
        self.tts_queue = queue.Queue()

        self.is_speaking = False

        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # TTS
        if TTS_AVAILABLE:
            print("Loading TTS...")
            self.tts = VietnameseTTS(
                model_path=NGHITTS_MODEL_PATH,
                enable_transliteration=True,
            )
            self.tts_enabled = True
        else:
            self.tts_enabled = False

        # start TTS worker
        threading.Thread(target=self.tts_worker, daemon=True).start()

        print("System ready!\n")

    # ================= TTS WORKER =================

    def tts_worker(self):
        while True:
            text = self.tts_queue.get()
            if not text:
                continue

            try:
                self.is_speaking = True

                audio, sr = self.tts.speak(text, output_path=None)

                if isinstance(audio, list):
                    audio = np.array(audio, dtype=np.float32)

                sd.play(audio, sr)
                sd.wait()

            except Exception as e:
                print(f"TTS error: {e}")

            self.is_speaking = False

    # ================= LLM STREAM =================

    def ask_llm_stream(self, text):
        self.conversation_history.append({"role": "user", "content": text})

        stream = self.ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=self.conversation_history,
            stream=True,
            options=OLLAMA_OPTIONS,
        )

        buffer = ""

        for chunk in stream:
            content = chunk["message"]["content"]

            if content:
                print(content, end="", flush=True)
                buffer += content

                # 👇 trigger khi đủ dài hoặc có dấu câu
                if len(buffer) > 40 or any(p in buffer for p in [".", "?", "!"]):
                    yield buffer.strip()
                    buffer = ""

        if buffer:
            yield buffer.strip()

    # ================= PROCESS =================

    def process_query(self, text):
        print(f"\nUser: {text}")
        print("LLM: ", end="", flush=True)

        full_response = ""

        for chunk_text in self.ask_llm_stream(text):
            full_response += chunk_text + " "
            self.tts_queue.put(chunk_text)  # 👈 non-block

        print("\nDone.\n")

        self.conversation_history.append({
            "role": "assistant",
            "content": full_response.strip()
        })

    # ================= AUDIO =================

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_speaking:
            self.audio_queue.put(indata.copy())

    # ================= MAIN LOOP =================

    def run_voice_mode(self):
        print("🎤 Speak now... Ctrl+C to stop")

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
                        text = self.recognizer.get_result(self.stream).strip()
                        if text:
                            self.process_query(text)

                        self.stream = self.recognizer.create_stream()

                time.sleep(0.01)

    def run_text_mode(self):
        while True:
            text = input("Bạn: ")
            self.process_query(text)

    def run(self):
        mode = input("1: voice | 2: text: ")
        if mode == "1":
            self.run_voice_mode()
        else:
            self.run_text_mode()


# ================= MAIN =================

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()