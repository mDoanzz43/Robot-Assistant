import queue
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from llama_cpp import Llama

# Add nghitts/python_tts path
sys.path.insert(0, str(Path(__file__).parent / "nghitts" / "python_tts"))

try:
    from tts import VietnameseTTS

    TTS_AVAILABLE = True
except ImportError:
    print("⚠️ nghitts/python_tts not found. TTS disabled.")
    TTS_AVAILABLE = False


# ================= CONFIG =================

MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\Zipformer-30M-RNNT-Streaming-6000h"
LLM_MODEL_PATH = r"D:\STUDY\At_school\asr_zipformer\llm\models\qwen\qwen2.5-1.5b-instruct-q4_k_m.gguf"

# NGHITTS model (ONNX)
NGHITTS_MODEL_PATH = (
    r"D:\STUDY\At_school\asr_zipformer\nghitts\public\tts-model\vi\deepman3909.onnx"
)

SAMPLE_RATE = 16000
CHUNK_MS = 16
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)


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

        print("Loading LLM...")

        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )

        self.audio_queue = queue.Queue()
        self.is_speaking = False

        if TTS_AVAILABLE:
            print("Loading NGHITTS...")
            try:
                self.tts = VietnameseTTS(
                    model_path=NGHITTS_MODEL_PATH,
                    enable_transliteration=True,
                )
                self.tts_enabled = True
                print("✓ NGHITTS loaded")
            except Exception as e:
                print(f"⚠️ NGHITTS load failed: {e}")
                self.tts_enabled = False
        else:
            self.tts_enabled = False

        print("System ready.\n")

    # ================= LLM =================

    def ask_llm(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là Kitty - robot giáo dục cho trẻ em.\n"
                    "Quy tắc:\n"
                    "1. Trả lời tiếng Việt đơn giản cho trẻ em.\n"
                    "2. Chỉ trả lời tối đa 2 câu.\n"
                    "3. Không dùng danh sách số hoặc bullet.\n"
                    "4. Trả lời chính xác khoa học.\n"
                    "5. Nếu không biết hãy nói: Kitty sẽ tìm hiểu thêm nhé."
                ),
            },
            {"role": "user", "content": text},
        ]

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=80,
            temperature=0.2,
            top_p=0.8,
            repeat_penalty=1.1,
        )

        return output["choices"][0]["message"]["content"].strip()

    # ================= TTS =================

    def speak(self, text: str) -> None:
        if not self.tts_enabled:
            return

        try:
            self.is_speaking = True

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            print("🔊 Speaking (nghitts)...")

            audio, sample_rate = self.tts.speak(
                text,
                output_path=None,
                length_scale=1.0,
                preprocess=True,
            )

            if audio is None or len(audio) == 0:
                print("⚠️ Empty TTS audio")
                self.is_speaking = False
                return

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            sd.play(audio, sample_rate)
            sd.wait()

            print("✓ Done speaking")

            time.sleep(0.4)

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            self.stream = self.recognizer.create_stream()
            self.is_speaking = False

            print("✓ Ready\n")

        except Exception as e:
            print(f"TTS error: {e}")
            self.is_speaking = False

    # ================= AUDIO CALLBACK =================

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if not self.is_speaking:
            self.audio_queue.put(indata.copy())

    # ================= MAIN LOOP =================

    def run(self) -> None:
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
                            final_text = final_text.lower()
                            print("\n👤 User:", final_text)

                            response = self.ask_llm(final_text)
                            print("🤖 LLM:", response)

                            self.speak(response)
                        else:
                            self.stream = self.recognizer.create_stream()

                time.sleep(0.01)


# ================= MAIN =================

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
