"""hardware/tts_bridge.py - Realtime TTS bridge using NGHI TTS."""
import queue
import threading
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PIPER_MODEL, TTS_LENGTH_SCALE

_SENT_END = {".", "!", "?", "…", ":", "\n"}


class TTSBridge:
    def __init__(self, model: str = PIPER_MODEL):
        self.model = self._resolve_model_path(model)
        self._lock = threading.Lock()
        self._speak_q: "queue.Queue[str]" = queue.Queue()
        self._tts = None
        self._sd = None
        self._ready = False

        self._load_runtime()
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def _resolve_model_path(self, model: str) -> str:
        p = Path(model)
        if p.is_file() and p.suffix.lower() == ".onnx":
            return str(p)
        if p.is_dir():
            candidates = sorted(p.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx file found in TTS model dir: {p}")
            return str(candidates[0])
        return model

    def _load_runtime(self):
        project_root = Path(__file__).resolve().parents[2]
        tts_module_path = project_root / "nghitts" / "python_tts"
        if str(tts_module_path) not in sys.path:
            sys.path.insert(0, str(tts_module_path))

        try:
            from tts import VietnameseTTS
            import sounddevice as sd

            self._tts = VietnameseTTS(
                model_path=self.model,
                enable_transliteration=True,
            )
            self._sd = sd
            self._ready = True
            logger.info(f"TTS ready: {self.model}")
        except Exception as e:
            self._ready = False
            logger.warning(f"TTS runtime unavailable: {e}")

    def _synthesize_and_play(self, text: str, length_scale: float = TTS_LENGTH_SCALE) -> bool:
        text = (text or "").strip()
        if not text:
            return True
        if not self._ready:
            logger.warning("TTS not ready; skip speaking")
            return False

        try:
            with self._lock:
                audio, sr = self._tts.speak(text, output_path=None, length_scale=length_scale)
                self._sd.play(audio, sr)
                self._sd.wait()
            return True
        except Exception as e:
            logger.error(f"TTS speak error: {e}")
            return False

    def _tts_worker(self):
        while True:
            item = self._speak_q.get()
            if item is None:
                continue
            if isinstance(item, tuple):
                text, length_scale = item
            else:
                text, length_scale = item, TTS_LENGTH_SCALE
            self._synthesize_and_play(text, length_scale=length_scale)

    def speak(self, text: str, blocking: bool = True, length_scale: float = TTS_LENGTH_SCALE) -> bool:
        if blocking:
            return self._synthesize_and_play(text, length_scale=length_scale)
        self._speak_q.put((text, length_scale))
        return True

    def speak_async(self, text: str, length_scale: float = TTS_LENGTH_SCALE):
        self.speak(text, blocking=False, length_scale=length_scale)

    def speak_streaming(self, token_iter: Iterator[str]):
        buf = ""
        for token in token_iter:
            buf += token
            if buf and buf[-1] in _SENT_END:
                sentence = buf.strip()
                if len(sentence) > 2:
                    self.speak_async(sentence)
                buf = ""
        if buf.strip():
            self.speak_async(buf.strip())

    def play_wav(self, wav_path: str, blocking: bool = True) -> bool:
        path = Path(wav_path)
        if not path.exists():
            logger.warning(f"WAV not found: {path}")
            return False
        if not self._ready:
            logger.warning("TTS runtime unavailable; cannot play WAV")
            return False

        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            if sample_width != 2:
                logger.warning(f"Unsupported WAV sample width: {sample_width}")
                return False

            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            with self._lock:
                self._sd.play(audio, sample_rate)
                if blocking:
                    self._sd.wait()
            return True
        except Exception as e:
            logger.error(f"play_wav error: {e}")
            return False

    def test(self) -> bool:
        return self.speak("Địa điểm du lịch của hội nhóm chúng ta là vào Cửa Lò, nơi có rất nhiều cảnh đẹp. Nếu bạn muốn, hãy liên hệ với tôi. Xin chào, tôi là rô bốt thông minh, hôm nay tớ với cậu sẽ học về chủ đề gì đây nhỉ? Ngày này năm sau là ngày 20 tháng 8 năm 2025.", blocking=True)


# # kiểm tra giọng nói tts xem có phát ra chưa
# if __name__ == "__main__":
#     bridge = TTSBridge()
#     bridge.test()