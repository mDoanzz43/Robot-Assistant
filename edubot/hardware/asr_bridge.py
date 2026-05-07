"""hardware/asr_bridge.py - Realtime Sherpa-ONNX ASR bridge."""
import queue
import threading
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ASR_MODEL_DIR, ASR_SAMPLE_RATE


class ASRBridge:
    """Microphone -> sherpa-onnx stream -> utterance callback."""

    def __init__(self, model_dir: str = ASR_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.sample_rate = ASR_SAMPLE_RATE
        self.chunk_ms = 16
        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)

        self._rec = None
        self._stream = None
        self._running = False
        self._paused = False
        self._audio_q: "queue.Queue" = queue.Queue(maxsize=200)

    def _pick_one(self, pattern: str) -> Path:
        matches = sorted(self.model_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing ASR file pattern: {pattern}")
        return matches[0]

    def _resolve_model_paths(self) -> Tuple[Path, Path, Path, Path]:
        encoder = self._pick_one("encoder-*.onnx")
        decoder = self._pick_one("decoder-*.onnx")
        joiner = self._pick_one("joiner-*.onnx")

        tokens_txt = self.model_dir / "tokens.txt"
        config_json = self.model_dir / "config.json"
        if tokens_txt.exists():
            tokens = tokens_txt
        elif config_json.exists():
            tokens = config_json
        else:
            raise FileNotFoundError("Missing tokens.txt or config.json in ASR model dir")
        return encoder, decoder, joiner, tokens

    def init(self) -> "ASRBridge":
        if self._rec is not None and self._stream is not None:
            return self
        try:
            import sherpa_onnx

            encoder, decoder, joiner, tokens = self._resolve_model_paths()
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                sample_rate=self.sample_rate,
                enable_endpoint_detection=True,
            )
            self._stream = self._rec.create_stream()
            logger.info(f"ASR ready: {self.model_dir}")
            return self
        except ImportError:
            logger.error("sherpa-onnx not installed")
            raise
        except Exception as e:
            logger.error(f"ASR init failed: {e}")
            raise

    @staticmethod
    def _extract_text(result_obj) -> str:
        if result_obj is None:
            return ""
        if isinstance(result_obj, str):
            return result_obj.strip()
        txt = getattr(result_obj, "text", "")
        return txt.strip() if isinstance(txt, str) else ""

    @staticmethod
    def _normalize_asr_text(text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            return ""

        # Normalize common spoken borrow words so downstream intent/tts stays stable.
        t = re.sub(r"\b(ok|oke|okay|okey|ô kê|ôkê|ô cê|oc|ôk)\b", "ô kê", t)
        t = re.sub(r"\bwow\b", "uao", t)

        # A frequent ASR collapse for "ô kê" in short utterances.
        if t in {"ố", "o"}:
            t = "ô kê"

        return t

    def audio_callback(self, indata, frames, time_info, status):
        if self._paused:
            return
        try:
            self._audio_q.put_nowait(indata.copy())
        except queue.Full:
            # Drop oldest chunk under heavy load to keep realtime behavior.
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                pass

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def listen_loop(self, callback: Callable[[str], None], stop_event: Optional[threading.Event] = None):
        """Blocking listen loop. Should run in a dedicated thread."""
        if self._rec is None:
            raise RuntimeError("ASRBridge.init() must be called before listen_loop()")

        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return

        self._running = True
        logger.info("ASR listening...")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                dtype="float32",
                callback=self.audio_callback,
            ):
                while self._running:
                    if stop_event and stop_event.is_set():
                        break

                    try:
                        indata = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    audio = indata.squeeze().astype("float32")
                    self._stream.accept_waveform(self.sample_rate, audio.tolist())

                    while self._rec.is_ready(self._stream):
                        self._rec.decode_stream(self._stream)

                    if self._rec.is_endpoint(self._stream):
                        text = self._normalize_asr_text(self._extract_text(self._rec.get_result(self._stream)))
                        if text:
                            logger.info(f"ASR: {text}")
                            callback(text)
                        self._stream = self._rec.create_stream()
        except KeyboardInterrupt:
            logger.info("ASR listen loop interrupted by Ctrl+C")

        self._running = False

    def stop(self):
        self._running = False


# Tôi muốn nói vào micro để kiểm tra
# ASR có thể nhận diện đúng câu tôi nói hay không

# if __name__ == "__main__":
#     import time

#     def on_text(text):
#         print(f"Recognized: {text}")

#     bridge = ASRBridge().init()
#     stop_evt = threading.Event()

#     t = threading.Thread(target=bridge.listen_loop, args=(on_text, stop_evt))
#     t.start()

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         stop_evt.set()
#         t.join()
