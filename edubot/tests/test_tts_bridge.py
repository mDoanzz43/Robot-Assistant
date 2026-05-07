"""tests/test_tts_bridge.py - Unit tests for TTSBridge."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.tts_bridge import TTSBridge


class _FakeVietnameseTTS:
    def __init__(self, model_path, enable_transliteration=True):
        self.model_path = model_path
        self.enable_transliteration = enable_transliteration

    def speak(self, text, output_path=None, **kwargs):
        return np.zeros(1600, dtype=np.float32), 16000


class _FakeSD:
    def __init__(self):
        self.play_calls = 0
        self.wait_calls = 0

    def play(self, audio, sample_rate):
        self.play_calls += 1

    def wait(self):
        self.wait_calls += 1


def test_tts_resolve_model_path_from_directory(tmp_path):
    onnx = tmp_path / "voice.onnx"
    onnx.write_text("x", encoding="utf-8")

    bridge = TTSBridge.__new__(TTSBridge)
    resolved = bridge._resolve_model_path(str(tmp_path))

    assert resolved.endswith("voice.onnx")


def test_tts_speak_blocking(monkeypatch, tmp_path):
    onnx = tmp_path / "voice.onnx"
    onnx.write_text("x", encoding="utf-8")

    fake_sd = _FakeSD()

    monkeypatch.setitem(sys.modules, "tts", SimpleNamespace(VietnameseTTS=_FakeVietnameseTTS))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    bridge = TTSBridge(model=str(onnx))

    ok = bridge.speak("xin chao", blocking=True)

    assert ok is True
    assert fake_sd.play_calls == 1
    assert fake_sd.wait_calls == 1


def test_tts_streaming_batches_sentences(monkeypatch, tmp_path):
    onnx = tmp_path / "voice.onnx"
    onnx.write_text("x", encoding="utf-8")

    fake_sd = _FakeSD()

    monkeypatch.setitem(sys.modules, "tts", SimpleNamespace(VietnameseTTS=_FakeVietnameseTTS))
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    bridge = TTSBridge(model=str(onnx))

    got = []

    def _fake_speak_async(text):
        got.append(text)

    bridge.speak_async = _fake_speak_async
    bridge.speak_streaming(iter(["Xin", " chao.", " Toi", " la", " robot!"]))

    assert got == ["Xin chao.", "Toi la robot!"]
