"""tests/test_asr_bridge.py - Unit tests for ASRBridge."""
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.asr_bridge import ASRBridge


class _FakeStream:
    def __init__(self):
        self.accepted = 0

    def accept_waveform(self, sample_rate, audio):
        self.accepted += 1


class _FakeRecognizer:
    def __init__(self):
        self.stream = _FakeStream()
        self.endpoint_ready = False

    def create_stream(self):
        return self.stream

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        return None

    def is_endpoint(self, stream):
        if self.endpoint_ready:
            self.endpoint_ready = False
            return True
        return False

    def get_result(self, stream):
        return "xin chao"


def test_asr_init_discovers_model_files(tmp_path, monkeypatch):
    (tmp_path / "encoder-epoch-1.onnx").write_text("x", encoding="utf-8")
    (tmp_path / "decoder-epoch-1.onnx").write_text("x", encoding="utf-8")
    (tmp_path / "joiner-epoch-1.onnx").write_text("x", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    called = {}
    fake_rec = _FakeRecognizer()

    class _FakeOnlineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            called.update(kwargs)
            return fake_rec

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OnlineRecognizer=_FakeOnlineRecognizer),
    )

    bridge = ASRBridge(str(tmp_path)).init()

    assert bridge._rec is fake_rec
    assert called["encoder"].endswith("encoder-epoch-1.onnx")
    assert called["decoder"].endswith("decoder-epoch-1.onnx")
    assert called["joiner"].endswith("joiner-epoch-1.onnx")
    assert called["tokens"].endswith("config.json")


def test_asr_listen_loop_emits_text(monkeypatch):
    fake_rec = _FakeRecognizer()

    class _FakeInputStream:
        def __init__(self, **kwargs):
            self._cb = kwargs["callback"]

        def __enter__(self):
            audio = np.ones((256, 1), dtype=np.float32)
            self._cb(audio, 256, None, None)
            fake_rec.endpoint_ready = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=_FakeInputStream),
    )

    bridge = ASRBridge("dummy")
    bridge._rec = fake_rec
    bridge._stream = fake_rec.create_stream()

    got = []
    stop_evt = threading.Event()

    def _on_text(text):
        got.append(text)
        stop_evt.set()

    bridge.listen_loop(_on_text, stop_evt)

    assert got == ["xin chao"]


def test_asr_stop_sets_running_false():
    bridge = ASRBridge("dummy")
    bridge._running = True
    bridge.stop()
    assert bridge._running is False
