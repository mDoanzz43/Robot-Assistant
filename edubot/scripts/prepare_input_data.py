"""
prepare_input_data.py — Normalize internet-sourced material into lesson bundles.

Input expectations:
  - raw .txt / .md / .csv exports from agents or manual curation
  - optional .json / .jsonl files that already contain structured lesson data

Output:
  - one JSON bundle per source file
  - a manifest for incremental rebuilds

This script does not crawl the web. It is the normalization step after
an agent has collected and cleaned source material from internet sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.offline_ollama_build import infer_domain_from_topic, infer_topic_from_path  # noqa: E402


INPUT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".jsonl"}


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return hash_text(read_text_file(path))


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def read_csv_file(path: Path) -> str:
    rows: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parts = []
            for key in ("title", "topic", "heading", "content", "summary", "description", "text", "body"):
                value = (row.get(key) or "").strip()
                if value:
                    parts.append(value)
            if parts:
                rows.append(" | ".join(parts))
    return "\n".join(rows).strip()


def load_structured_json(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"records": data}
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return {"records": records}
    return {}


def flatten_records(records: Iterable) -> str:
    parts: List[str] = []
    for item in records:
        if isinstance(item, dict):
            for key in ("title", "topic", "heading", "content", "summary", "description", "text", "body"):
                value = str(item.get(key, "")).strip()
                if value:
                    parts.append(value)
        else:
            value = str(item).strip()
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def normalize_bundle(path: Path, topic: str | None, lesson_level: str, age: int) -> dict:
    suffix = path.suffix.lower()
    source = {"path": str(path), "type": suffix.lstrip(".") or "txt"}

    if suffix in {".json", ".jsonl"}:
        raw = load_structured_json(path)
    elif suffix == ".csv":
        raw = {"text": read_csv_file(path)}
    else:
        raw = {"text": read_text_file(path)}

    raw_text = (raw.get("text") or raw.get("summary") or raw.get("content") or "").strip()
    if not raw_text and raw.get("records"):
        raw_text = flatten_records(raw.get("records"))

    inferred_topic = topic or raw.get("topic") or infer_topic_from_path(path)
    inferred_domain = raw.get("domain") or infer_domain_from_topic(inferred_topic)

    concepts = raw.get("concepts") or raw.get("entities") or []
    relations = raw.get("relations") or raw.get("edges") or []
    qa_bank = raw.get("qa_bank") or raw.get("qas") or []

    if not isinstance(concepts, list):
        concepts = []
    if not isinstance(relations, list):
        relations = []
    if not isinstance(qa_bank, list):
        qa_bank = []

    bundle = {
        "doc_id": raw.get("doc_id") or f"doc_{inferred_domain}_{slugify(path.stem)}",
        "topic": inferred_topic,
        "domain": inferred_domain,
        "lesson_level": raw.get("lesson_level") or raw.get("level") or lesson_level,
        "age": raw.get("age", age),
        "source": raw.get("source") or source,
        "text": raw_text,
        "concepts": concepts,
        "relations": relations,
        "qa_bank": qa_bank,
        "structured": bool(concepts or relations or qa_bank),
        "source_hash": hash_file(path),
    }
    return bundle


def iter_input_files(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
    )


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "files": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize source material into EduBot lesson bundles")
    parser.add_argument("--input", required=True, help="File or folder of source material")
    parser.add_argument("--output", default="input_data", help="Output folder for normalized bundles")
    parser.add_argument("--topic", default=None, help="Force a topic for all sources")
    parser.add_argument("--lesson-level", default="hieu", help="Default lesson level: nhan_biet|hieu|van_dung|van_dung_cao")
    parser.add_argument("--age", type=int, default=6, help="Recommended age for the content")
    parser.add_argument("--recursive", action="store_true", help="Scan folders recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized bundles")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    bundles_dir = output_dir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    files = iter_input_files(input_path, recursive=args.recursive)
    if not files:
        raise FileNotFoundError(f"No supported source files found in {input_path}")

    processed = 0
    skipped = 0
    for path in files:
        bundle = normalize_bundle(path, args.topic, args.lesson_level, args.age)
        key = str(path.resolve())
        current_hash = bundle["source_hash"]
        old = manifest.get("files", {}).get(key, {})

        if not args.overwrite and old.get("source_hash") == current_hash:
            skipped += 1
            continue

        out_path = bundles_dir / bundle["domain"] / slugify(bundle["topic"]) / f"{slugify(path.stem)}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest.setdefault("files", {})[key] = {
            "source_hash": current_hash,
            "output": str(out_path),
            "topic": bundle["topic"],
            "domain": bundle["domain"],
            "updated_at": int(time.time()),
        }
        processed += 1

    save_manifest(manifest_path, manifest)
    print(f"processed={processed} skipped={skipped} output={bundles_dir}")


if __name__ == "__main__":
    main()