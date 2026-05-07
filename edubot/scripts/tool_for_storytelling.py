"""
scripts/tool.py — Generate narrated WAV files for story_telling assets.

Mục tiêu:
- Đọc fairy_tales.json hoặc một file story JSON tương tự.
- Tiền xử lý theo câu / mệnh đề để thêm nhịp nghỉ tự nhiên.
- Xuất mỗi truyện ra một file .wav riêng để Jetson phát nhanh.
- Xuất sidecar JSON và script text để kiểm tra cách chia câu / pause.

Usage:
    python edubot/scripts/tool.py \
        --input edubot/data/documents/story_telling/fairy_tales.json \
        --output-dir edubot/data/documents/story_telling/audio
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import unicodedata
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EDUBOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EDUBOT_ROOT.parent

for p in (str(EDUBOT_ROOT), str(REPO_ROOT / "nghitts" / "python_tts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import PIPER_MODEL  # noqa: E402


def _load_vietnamese_tts() -> Any:
    module = importlib.import_module("tts")
    return module.VietnameseTTS


VietnameseTTS = _load_vietnamese_tts()


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


@dataclass
class StorySegment:
    text: str
    punctuation: str
    pause_ms: int
    length_scale: float
    paragraph_end: bool = False


def slugify(text: str) -> str:
    text = strip_accents((text or "").strip()).lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def story_filename(story_id: str, title: str) -> str:
    sid = re.sub(r"\D", "", story_id or "")
    sid = sid.zfill(2) if sid else "00"
    return f"{sid}_{slugify(title)}.wav"


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_story_records(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("stories") or raw.get("records") or raw.get("items") or []
    else:
        records = []

    result: List[dict] = []
    for idx, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or f"Truyện {idx}").strip()
        story_id = str(item.get("id") or idx).strip()
        paragraphs = item.get("paragraphs")
        if isinstance(paragraphs, list):
            text_parts = [str(p).strip() for p in paragraphs if str(p).strip()]
            story_text = "\n\n".join(text_parts)
        else:
            content = item.get("text") or item.get("content") or ""
            story_text = str(content).strip()

        if story_text:
            result.append({"id": story_id, "title": title, "text": story_text})

    return result


def _terminal_punctuation(text: str) -> str:
    stripped = (text or "").strip()
    match = re.search(r"([.!?…,:;])(?:[\"'”’)\]]*)$", stripped)
    return match.group(1) if match else "."


def _pause_ms_for_punctuation(punct: str, paragraph_end: bool) -> int:
    base = {
        ",": 120,
        ";": 180,
        ":": 200,
        ".": 260,
        "?": 280,
        "!": 240,
        "…": 420,
    }.get(punct, 220)
    return base + (180 if paragraph_end else 0)


def _length_scale_for_punctuation(punct: str, text: str, base_scale: float) -> float:
    scale = {
        ",": 0.92,
        ";": 0.95,
        ":": 0.96,
        ".": 1.00,
        "?": 0.98,
        "!": 0.94,
        "…": 1.05,
    }.get(punct, 0.98)

    if len(text.strip()) <= 18:
        scale *= 0.97
    if len(text.strip()) >= 120:
        scale *= 1.02

    scale *= base_scale
    return max(0.85, min(1.15, round(scale, 3)))


def _split_long_clause(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if len(words) <= 1:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def segment_story_text(text: str, max_clause_chars: int, base_scale: float) -> List[StorySegment]:
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    segments: List[StorySegment] = []

    for para_idx, paragraph in enumerate(paragraphs):
        if paragraph and paragraph[-1] not in ".!?…":
            paragraph = paragraph + "."

        sentence_parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
        if not sentence_parts:
            sentence_parts = [paragraph]

        for sent_idx, sentence in enumerate(sentence_parts):
            if len(sentence) <= max_clause_chars:
                pieces = [sentence]
            else:
                clause_parts = [c.strip() for c in CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
                if len(clause_parts) > 1:
                    pieces = []
                    for clause in clause_parts:
                        if len(clause) > max_clause_chars:
                            pieces.extend(_split_long_clause(clause, max_clause_chars))
                        else:
                            pieces.append(clause)
                else:
                    pieces = _split_long_clause(sentence, max_clause_chars)

            for piece_idx, piece in enumerate(pieces):
                punct = _terminal_punctuation(piece)
                paragraph_end = (
                    para_idx == len(paragraphs) - 1
                    and sent_idx == len(sentence_parts) - 1
                    and piece_idx == len(pieces) - 1
                )
                pause_ms = _pause_ms_for_punctuation(punct, paragraph_end)
                length_scale = _length_scale_for_punctuation(punct, piece, base_scale)
                segments.append(
                    StorySegment(
                        text=piece.strip(),
                        punctuation=punct,
                        pause_ms=pause_ms,
                        length_scale=length_scale,
                        paragraph_end=paragraph_end,
                    )
                )

    return segments


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)

    if np.max(np.abs(audio)) <= 1.5:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    else:
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio


def _silence(sample_rate: int, pause_ms: int) -> np.ndarray:
    if pause_ms <= 0:
        return np.zeros(0, dtype=np.int16)
    n_samples = int(sample_rate * pause_ms / 1000)
    return np.zeros(max(0, n_samples), dtype=np.int16)


def synthesize_story_audio(
    tts: Any,
    segments: Iterable[StorySegment],
    output_path: Path,
) -> dict:
    audio_chunks: List[np.ndarray] = []
    segment_meta: List[dict] = []
    sample_rate: Optional[int] = None
    total_samples = 0

    for segment in segments:
        audio, sr = tts.speak(
            segment.text,
            output_path=None,
            length_scale=segment.length_scale,
            preprocess=True,
        )
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"Sample rate mismatch: expected {sample_rate}, got {sr}")

        pcm = _to_pcm16(np.asarray(audio))
        audio_chunks.append(pcm)
        total_samples += int(pcm.size)

        pause = _silence(sample_rate, segment.pause_ms)
        if pause.size:
            audio_chunks.append(pause)
            total_samples += int(pause.size)

        segment_meta.append(asdict(segment))

    if sample_rate is None:
        raise RuntimeError("No audio was generated for this story")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(full_audio.astype(np.int16, copy=False).tobytes())

    return {
        "output_wav": str(output_path),
        "sample_rate": sample_rate,
        "duration_seconds": round(total_samples / sample_rate, 3),
        "segments": segment_meta,
    }


def build_story_audio_bundle(
    input_path: Path,
    output_dir: Path,
    model_path: str,
    max_clause_chars: int,
    base_scale: float,
    overwrite: bool,
) -> dict:
    stories = load_story_records(input_path)
    if not stories:
        raise FileNotFoundError(f"No story records found in {input_path}")

    tts = VietnameseTTS(model_path=model_path, enable_transliteration=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_manifest = {
        "input": str(input_path),
        "model_path": model_path,
        "max_clause_chars": max_clause_chars,
        "base_length_scale": base_scale,
        "stories": [],
    }

    for story in stories:
        story_id = story["id"]
        title = story["title"]
        story_text = story["text"]
        segments = segment_story_text(story_text, max_clause_chars=max_clause_chars, base_scale=base_scale)
        if not segments:
            continue

        wav_name = story_filename(story_id, title)
        stem = wav_name[:-4]
        wav_path = output_dir / wav_name
        meta_path = output_dir / f"{stem}.segments.json"
        script_path = output_dir / f"{stem}.script.txt"

        if wav_path.exists() and not overwrite:
            continue

        result = synthesize_story_audio(tts, segments, wav_path)
        meta = {
            "id": story_id,
            "title": title,
            "source_text_chars": len(story_text),
            **result,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        script_lines: List[str] = []
        for seg in result["segments"]:
            script_lines.append(seg["text"])
            script_lines.append(f"[pause={seg['pause_ms']}ms, scale={seg['length_scale']}]")
        script_path.write_text("\n".join(script_lines).strip() + "\n", encoding="utf-8")

        bundle_manifest["stories"].append(meta)
        print(f"✓ {title} -> {wav_path.name} ({result['duration_seconds']}s)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV story audio with pause-aware preprocessing")
    parser.add_argument(
        "--input",
        default=str(EDUBOT_ROOT / "data" / "documents" / "story_telling" / "fairy_tales.json"),
        help="Story JSON file to convert into audio",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EDUBOT_ROOT / "data" / "documents" / "story_telling" / "audio"),
        help="Folder for generated WAV files",
    )
    parser.add_argument("--model", default=PIPER_MODEL, help="Piper ONNX model path")
    parser.add_argument("--max-clause-chars", type=int, default=84, help="Split long story clauses above this length")
    parser.add_argument("--base-length-scale", type=float, default=0.97, help="Global TTS speed multiplier for narration")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    build_story_audio_bundle(
        input_path=input_path,
        output_dir=output_dir,
        model_path=args.model,
        max_clause_chars=args.max_clause_chars,
        base_scale=args.base_length_scale,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()