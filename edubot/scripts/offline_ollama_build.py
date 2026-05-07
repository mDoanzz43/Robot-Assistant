"""
scripts/offline_build_ollama.py — Build Knowledge Base dùng Ollama (local, free)

Dùng cho: PC với RTX 3060 12GB (hoặc tương đương)
Model khuyến nghị: qwen2.5:14b (Q4_K_M, ~9GB VRAM)

Setup trước khi chạy:
    # 1. Cài Ollama
    curl -fsSL https://ollama.ai/install.sh | sh

    # 2. Pull model (1 lần duy nhất, ~9GB)
    ollama pull qwen2.5:14b

    # 3. Chạy build
    python scripts/offline_build_ollama.py \\
        --input docs/dong_vat.txt \\
        --topic "động vật" \\
        --age 6 \\
        --output data/

    # 4. Copy sang Jetson
    rsync -avz data/ user@jetson:~/edubot/data/

Lưu ý:
    - offline_build_ollama.py KHÔNG chạy trên Jetson (Ollama không hỗ trợ Jetson Nano)
    - Chỉ chạy trên PC có GPU (RTX 3060 hoặc tương đương)
"""

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Config ────────────────────────────────────────────────
DEFAULT_MODEL    = "qwen2.5:14b"
OLLAMA_BASE_URL  = "http://localhost:11434"
MANIFEST_NAME    = "build_manifest.json"
DOMAIN_ANIMAL    = "animal"
DOMAIN_SPACE     = "space"
DOMAIN_GENERAL   = "general"
ALLOWED_RELATIONS = {
    "IS_A", "HAS", "CAN", "LIVES_IN", "OPPOSITE_OF", "RELATED_TO",
    "PART_OF", "PREREQUISITE_OF", "EXAMPLE_OF", "CAUSES"
}
INPUT_EXTENSIONS = {".txt", ".pdf", ".md", ".csv", ".json", ".jsonl"}
# Nếu muốn dùng model khác (nhanh hơn nhưng kém hơn):
#   FALLBACK_MODEL = "qwen2.5:7b"


# ══════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ══════════════════════════════════════════════════════════════
class OllamaClient:
    """
    Wrapper nhẹ cho Ollama REST API.
    Dùng requests thay vì ollama-python để tránh version conflict.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 base_url: str = OLLAMA_BASE_URL):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        """Kiểm tra Ollama đang chạy và model đã pull chưa."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check model available (tên có thể có :latest suffix)
            model_base = self.model.split(":")[0]
            found = any(model_base in m for m in models)
            if not found:
                print(f"\n  [!] Model '{self.model}' chưa được pull!")
                print(f"      Chạy: ollama pull {self.model}")
                print(f"      Models hiện có: {models[:5]}")
                sys.exit(1)
            print(f"  ✓ Ollama connected | model: {self.model}")
        except requests.ConnectionError:
            print(f"\n  [!] Không kết nối được Ollama tại {self.base_url}")
            print("      Kiểm tra: ollama serve (hoặc ollama app đang chạy?)")
            sys.exit(1)

    def generate_json(self, prompt: str,
                      system: str = "",
                      temperature: float = 0.1,
                      max_retries: int = 3) -> dict:
        """
        Gọi Ollama với format=json để đảm bảo output là valid JSON.
        Retry tự động nếu parse thất bại.
        """
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self.model,
            "messages": messages,
            "format":  "json",          # ← Ollama hard JSON mode
            "stream":  False,
            "options": {
                "temperature": temperature,
                "top_p":       0.9,
                "num_ctx":     8192,    # Context window đủ lớn cho tài liệu
                "num_predict": 4096,    # Max output tokens
            }
        }

        for attempt in range(1, max_retries + 1):
            try:
                t0   = time.perf_counter()
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload, timeout=300
                )
                resp.raise_for_status()
                content  = resp.json()["message"]["content"]
                elapsed  = time.perf_counter() - t0
                tokens   = resp.json().get("eval_count", 0)
                tok_s    = round(tokens / elapsed, 1) if elapsed > 0 else 0
                print(f"      [{elapsed:.1f}s | {tok_s} tok/s]", end="")

                # Parse JSON — format=json đảm bảo valid nhưng vẫn strip để chắc
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned[:cleaned.rfind("```")] if "```" in cleaned else cleaned
                cleaned = cleaned.strip()

                return json.loads(cleaned)

            except json.JSONDecodeError as e:
                print(f"\n  [!] JSON parse error (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    print(f"  Raw output snippet: {content[:300]}")
                    return {}
                time.sleep(2)
            except Exception as e:
                print(f"\n  [!] Ollama error (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return {}
                time.sleep(3)

        return {}


# ══════════════════════════════════════════════════════════════
# DOCUMENT LOADER
# ══════════════════════════════════════════════════════════════
def load_doc(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".pdf":
        try:
            import importlib
            pdf_module = importlib.import_module("pypdf")
            pages = pdf_module.PdfReader(path).pages
            return "\n".join(pg.extract_text() or "" for pg in pages)
        except ImportError:
            raise RuntimeError("pip install pypdf")
    return p.read_text(encoding="utf-8")


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "doc"


def infer_topic_from_path(p: Path, default_topic: str = "khoa học") -> str:
    parent = p.parent.name.lower()
    if parent in {"solor_system", "solar_system", "space", "astronomy", "vutru", "vu_tru"}:
        return "hệ mặt trời và khoa học vũ trụ"
    if parent in {"animal", "animals", "dongvat", "dong_vat"}:
        return "thế giới động vật"
    stem = p.stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        return default_topic
    return stem


def infer_domain_from_topic(topic: str) -> str:
    t = (topic or "").lower()
    if any(k in t for k in ("động vật", "dong vat", "animal", "animals")):
        return DOMAIN_ANIMAL
    if any(k in t for k in ("vũ trụ", "vu tru", "hệ mặt trời", "he mat troi", "space", "astronomy")):
        return DOMAIN_SPACE
    return DOMAIN_GENERAL


def load_source_bundle(path: Path) -> dict:
    """Load raw text or a normalized lesson bundle from disk."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"records": data}
        except Exception:
            pass
    elif suffix == ".jsonl":
        records = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        except Exception:
            records = []
        return {"records": records}

    return {"text": load_doc(str(path))}


def normalize_bundle_fields(bundle: dict, path: Path, fallback_topic: str, age: int) -> dict:
    """Normalize a lesson bundle into the internal build schema."""
    text = (bundle.get("text") or bundle.get("content") or bundle.get("summary") or "").strip()
    records = bundle.get("records") if isinstance(bundle.get("records"), list) else []
    if not text and records:
        parts = []
        for item in records:
            if isinstance(item, dict):
                parts.extend([
                    str(item.get("title", "")).strip(),
                    str(item.get("content", "")).strip(),
                    str(item.get("summary", "")).strip(),
                    str(item.get("text", "")).strip(),
                ])
            else:
                parts.append(str(item).strip())
        text = "\n".join(part for part in parts if part)
    if not text:
        text = load_doc(str(path))

    topic = bundle.get("topic") or fallback_topic or infer_topic_from_path(path)
    domain = bundle.get("domain") or infer_domain_from_topic(topic)

    concepts = bundle.get("concepts") or bundle.get("entities") or []
    if not isinstance(concepts, list):
        concepts = []

    relations = bundle.get("relations") or bundle.get("edges") or []
    if not isinstance(relations, list):
        relations = []

    qa_bank = bundle.get("qa_bank") or bundle.get("qas") or []
    if not isinstance(qa_bank, list):
        qa_bank = []

    return {
        "text": text,
        "topic": topic,
        "domain": domain,
        "age": bundle.get("age", age),
        "lesson_level": bundle.get("lesson_level") or bundle.get("level") or "hieu",
        "doc_id": bundle.get("doc_id"),
        "concepts": concepts,
        "relations": relations,
        "qa_bank": qa_bank,
        "source": bundle.get("source") or {
            "path": str(path),
            "type": path.suffix.lower().lstrip(".") or "txt",
        },
        "structured": bool(concepts or relations or qa_bank),
    }


def postprocess_entities(data: dict, topic: str) -> dict:
    """Normalize IDs/relations and enrich structure to avoid flat graph."""
    concepts = data.get("concepts", []) or []
    relations = data.get("relations", []) or []

    # Deduplicate concepts by normalized name.
    seen_name = {}
    fixed_concepts = []
    id_map = {}
    for c in concepts:
        old_id = str(c.get("id") or f"c_{uuid.uuid4().hex[:8]}")
        name = (c.get("name") or "").strip()
        key = slugify(name)
        if key and key in seen_name:
            id_map[old_id] = seen_name[key]
            continue

        new_id = old_id if old_id.startswith("c_") else f"c_{uuid.uuid4().hex[:8]}"
        id_map[old_id] = new_id
        seen_name[key] = new_id
        c["id"] = new_id
        c["difficulty"] = max(1, min(5, int(c.get("difficulty", 2))))
        fixed_concepts.append(c)

    concept_ids = {c["id"] for c in fixed_concepts}
    fixed_relations = []
    rel_seen = set()
    for r in relations:
        frm = id_map.get(r.get("from_id", ""), r.get("from_id", ""))
        to = id_map.get(r.get("to_id", ""), r.get("to_id", ""))
        if not frm or not to or frm not in concept_ids or to not in concept_ids or frm == to:
            continue
        rel = str(r.get("relation", "RELATED_TO")).upper()
        if rel not in ALLOWED_RELATIONS:
            rel = "RELATED_TO"
        w = float(r.get("weight", 0.8))
        w = max(0.5, min(1.0, w))
        k = (frm, to, rel)
        if k in rel_seen:
            continue
        rel_seen.add(k)
        fixed_relations.append({"from_id": frm, "to_id": to, "relation": rel, "weight": w})

    # Enrich parent-child structure with explicit IS_A edges.
    for c in fixed_concepts:
        parent = c.get("parent_id")
        if not parent:
            continue
        p = id_map.get(parent, parent)
        if p in concept_ids and p != c["id"]:
            c["parent_id"] = p
            k = (c["id"], p, "IS_A")
            if k not in rel_seen:
                rel_seen.add(k)
                fixed_relations.append({
                    "from_id": c["id"], "to_id": p, "relation": "IS_A", "weight": 0.95
                })
        else:
            c["parent_id"] = None

    if not fixed_concepts:
        root_id = f"c_{uuid.uuid4().hex[:8]}"
        fixed_concepts = [{
            "id": root_id,
            "name": topic,
            "description": f"Chủ đề về {topic}",
            "difficulty": 2,
            "parent_id": None,
        }]

    return {"concepts": fixed_concepts, "relations": fixed_relations}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return content_hash(load_doc(str(path)))


def load_manifest(out_dir: Path) -> dict:
    mf = out_dir / MANIFEST_NAME
    if not mf.exists():
        return {"version": 1, "files": {}}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "files": {}}


def save_manifest(out_dir: Path, manifest: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_input_files(input_path: Path, recursive: bool = True) -> list:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
    ]
    return sorted(files)


# ══════════════════════════════════════════════════════════════
# STEP 1: ENTITY + RELATION EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_entities(client: OllamaClient, text: str,
                     topic: str, age: int) -> dict:
    """
    Trích xuất khái niệm và quan hệ từ tài liệu.
    Dùng Ollama format=json để đảm bảo output valid.
    """
    SYSTEM = (
        "Bạn là chuyên gia phân tích nội dung giáo dục. "
        "Nhiệm vụ: trích xuất khái niệm và quan hệ từ văn bản. "
        "LUÔN trả về JSON hợp lệ theo đúng schema được yêu cầu. "
        "Không thêm giải thích hay text ngoài JSON."
    )

    # Chunk text nếu quá dài (Ollama có giới hạn context)
    doc_snippet = text[:4000] if len(text) > 4000 else text

    PROMPT = f"""Phân tích văn bản về "{topic}" dành cho trẻ {age} tuổi.

Văn bản:
{doc_snippet}

Trả về JSON với schema SAU (chỉ JSON, không giải thích):
{{
  "concepts": [
    {{
      "id": "c_XXXXXXXX",
      "name": "tên khái niệm ngắn gọn",
      "description": "mô tả 1 câu, phù hợp trẻ {age} tuổi",
      "difficulty": 2,
      "parent_id": null
    }}
  ],
  "relations": [
    {{
      "from_id": "c_XXXXXXXX",
      "to_id": "c_YYYYYYYY",
      "relation": "IS_A",
      "weight": 1.0
    }}
  ]
}}

Quy tắc:
- id phải duy nhất, dạng c_ + 8 ký tự chữ thường/số
- difficulty: 1=rất dễ, 2=dễ, 3=trung bình, 4=khó, 5=rất khó
- parent_id: null nếu là concept gốc, hoặc id của concept cha
- relation chỉ dùng 1 trong: IS_A, HAS, CAN, LIVES_IN, OPPOSITE_OF, RELATED_TO, PART_OF, PREREQUISITE_OF, EXAMPLE_OF, CAUSES
- weight: 0.5–1.0, quan hệ chính xác hơn thì weight cao hơn
- Tạo 12–20 concepts và 20–40 relations
- Tên concepts bằng tiếng Việt, ngắn gọn"""

    print("\n  → [1/2] Extracting entities & relations...")
    result = client.generate_json(PROMPT, system=SYSTEM, temperature=0.05)

    n_c = len(result.get("concepts", []))
    n_r = len(result.get("relations", []))
    print(f" → {n_c} concepts, {n_r} relations")

    # Fallback: tạo concept đơn giản nếu extraction thất bại
    if n_c == 0:
        print("  [!] Extraction thất bại, dùng fallback minimal concept")
        concept_id = f"c_{uuid.uuid4().hex[:8]}"
        result = {
            "concepts": [{
                "id": concept_id,
                "name": topic,
                "description": f"Chủ đề về {topic}",
                "difficulty": 2,
                "parent_id": None
            }],
            "relations": []
        }

    return postprocess_entities(result, topic)


# ══════════════════════════════════════════════════════════════
# STEP 2: Q&A BANK GENERATION
# ══════════════════════════════════════════════════════════════
def generate_qa_bank(client: OllamaClient, text: str,
                     concepts: list, topic: str, age: int) -> dict:
    """
    Tạo câu hỏi kiểm tra cho từng concept.
    Chia nhỏ thành nhiều batch nếu concepts nhiều để tránh timeout.
    """
    SYSTEM = (
        "Bạn là giáo viên tạo câu hỏi kiểm tra cho trẻ em. "
        "Câu hỏi phải chính xác, dựa hoàn toàn vào văn bản được cung cấp. "
        "KHÔNG bịa thêm thông tin không có trong văn bản. "
        "Trả về JSON hợp lệ theo đúng schema. Không thêm text ngoài JSON."
    )

    doc_snippet = text[:3000] if len(text) > 3000 else text

    # Danh sách concepts để model biết mapping concept_id
    concept_list = "\n".join(
        f"- {c['id']}: {c['name']} — {c.get('description', '')}"
        for c in concepts[:12]
    )

    target_qa = max(18, min(36, len(concepts) * 2))
    PROMPT = f"""Tạo câu hỏi kiểm tra về "{topic}" cho trẻ {age} tuổi.

Danh sách concepts (dùng đúng id):
{concept_list}

Văn bản tham khảo (chỉ dùng thông tin từ đây):
{doc_snippet}

Tạo khoảng {target_qa} câu hỏi đa dạng. Trả về JSON:
{{
  "qa_bank": [
    {{
      "id": "qa_XXXXXXXX",
      "concept_id": "c_XXXXXXXX",
      "question": "Câu hỏi rõ ràng cho trẻ {age} tuổi?",
      "answer": "Đáp án chính xác, ngắn gọn, lấy từ văn bản",
      "type": "open",
      "options": [],
      "difficulty": 2
    }}
  ]
}}

Quy tắc:
- id: dạng qa_ + 8 ký tự
- concept_id: PHẢI khớp với một trong các id ở trên
- type: "open" (câu tự luận), "yesno" (có/không), "fill" (điền vào chỗ trống)
- difficulty: 1-5 (phân bố đều, không phải toàn 3)
- options: [] cho tất cả (không làm MCQ để tránh bịa đặt)
- answer: trả lời bằng tiếng Việt, lấy trực tiếp từ văn bản
- Đừng bịa đặt câu trả lời — nếu không có trong văn bản, bỏ qua câu đó
- difficulty 1-2: câu dễ (có/không, điền 1 từ)
- difficulty 3: câu trung bình  
- difficulty 4-5: câu yêu cầu giải thích ngắn"""

    print("  → [2/2] Generating Q&A bank...")
    result = client.generate_json(PROMPT, system=SYSTEM, temperature=0.1)

    n_qa = len(result.get("qa_bank", []))
    print(f" → {n_qa} Q&A items")

    # Validate và clean Q&A
    valid_concept_ids = {c["id"] for c in concepts}
    valid_qa = []
    for qa in result.get("qa_bank", []):
        # Đảm bảo có đủ fields
        if not qa.get("question") or not qa.get("answer"):
            continue
        # Gán concept_id hợp lệ nếu sai
        if qa.get("concept_id") not in valid_concept_ids:
            if concepts:
                qa["concept_id"] = concepts[0]["id"]
            else:
                continue
        # Ensure id unique
        if not qa.get("id"):
            qa["id"] = f"qa_{uuid.uuid4().hex[:8]}"
        qa["type"]    = qa.get("type", "open")
        qa["options"] = qa.get("options", [])
        qa["difficulty"] = max(1, min(5, int(qa.get("difficulty", 3))))
        valid_qa.append(qa)

    if len(valid_qa) < max(6, len(concepts)):
        for c in concepts[: max(0, len(concepts) - len(valid_qa))]:
            qid = f"qa_{uuid.uuid4().hex[:8]}"
            q = f"{c.get('name', 'Khái niệm này')} là gì?"
            a = c.get("description") or f"Đây là kiến thức thuộc chủ đề {topic}."
            valid_qa.append({
                "id": qid,
                "concept_id": c.get("id"),
                "question": q,
                "answer": a,
                "type": "open",
                "options": [],
                "difficulty": max(1, min(5, int(c.get("difficulty", 2))))
            })

    return {"qa_bank": valid_qa}


# ══════════════════════════════════════════════════════════════
# STEP 3: BUILD & INDEX
# ══════════════════════════════════════════════════════════════
def build_and_index(doc_text: str, doc_id: str,
                    data: dict, out_dir: Path,
                    topic: str):
    """Build/append graph + ChromaDB + SQLite into a unified store."""
    from core.database   import DB, ConceptDAO, QADAO
    from knowledge.graph  import KnowledgeGraph
    from knowledge.vector import VectorStore

    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    import config as cfg
    cfg.DB_PATH    = out_dir / "edubot.db"
    cfg.GRAPH_PATH = out_dir / "graph" / "knowledge_graph.json"
    cfg.CHROMA_DIR = out_dir / "chroma"
    (out_dir / "graph").mkdir(exist_ok=True)
    (out_dir / "chroma").mkdir(exist_ok=True)

    # Reset DB singleton
    from core.database import DB as _DB
    _DB._instance = None
    db = _DB.get()

    c_dao  = ConceptDAO(db)
    qa_dao = QADAO(db)

    # Insert document
    db.execute(
        "INSERT OR REPLACE INTO documents(id,title) VALUES(?,?)",
        (doc_id, topic)
    )
    db.commit()

    # Normalize IDs to avoid collisions across files and within one generation batch.
    raw_concepts = data.get("concepts", [])
    concept_map = {}
    concepts = []
    used_concept_ids = set()
    seen_old_ids = set()
    for raw in raw_concepts:
        old_id = str(raw.get("id") or f"c_{uuid.uuid4().hex[:8]}")
        if old_id in seen_old_ids:
            # LLM sometimes emits duplicate concept IDs in one document.
            continue
        seen_old_ids.add(old_id)

        base = f"{doc_id}_{slugify(old_id)}"
        stable = base
        suffix = 2
        while stable in used_concept_ids:
            stable = f"{base}_{suffix}"
            suffix += 1
        used_concept_ids.add(stable)

        c = dict(raw)
        c["id"] = stable
        concept_map[old_id] = stable
        concepts.append(c)

    for c in concepts:
        parent = c.get("parent_id")
        if parent:
            c["parent_id"] = concept_map.get(parent, f"{doc_id}_{slugify(str(parent))}")

    for r in data.get("relations", []):
        r["from_id"] = concept_map.get(r.get("from_id", ""), "")
        r["to_id"] = concept_map.get(r.get("to_id", ""), "")

    used_qa_ids = set()
    for qa in data.get("qa_bank", []):
        base_qid = f"{doc_id}_{slugify(qa.get('id', uuid.uuid4().hex[:8]))}"
        qid = base_qid
        suffix = 2
        while qid in used_qa_ids:
            qid = f"{base_qid}_{suffix}"
            suffix += 1
        used_qa_ids.add(qid)
        qa["id"] = qid
        q_cid = qa.get("concept_id", "")
        qa["concept_id"] = concept_map.get(q_cid, concepts[0]["id"] if concepts else q_cid)

    # Insert concepts
    for c in concepts:
        parent = c.get("parent_id")
        # Chỉ set parent_id nếu parent đã tồn tại
        if parent and not c_dao.get(parent):
            parent = None
        c_dao.upsert(
            id=c["id"], name=c["name"],
            desc=c.get("description", ""),
            parent_id=parent,
            doc_id=doc_id,
            difficulty=c.get("difficulty", 3),
            verified=1
        )

    # Insert relations
    for r in data.get("relations", []):
        try:
            if not r.get("from_id") or not r.get("to_id"):
                continue
            c_dao.add_relation(
                r["from_id"], r["to_id"],
                r.get("relation", "RELATED_TO"),
                float(r.get("weight", 1.0))
            )
        except Exception:
            pass  # Skip invalid relations

    # Insert Q&A
    qa_dao.bulk_load(data.get("qa_bank", []))

    print(f"  ✓ SQLite: {len(concepts)} concepts, "
          f"{len(data.get('qa_bank', []))} Q&As → {cfg.DB_PATH.name}")

    # Knowledge Graph (load existing first, then append)
    graph = KnowledgeGraph(cfg.GRAPH_PATH).load()
    for c in concepts:
        graph.add_node(
            c["id"], c["name"],
            c.get("description", ""),
            c.get("difficulty", 3), True
        )
    for r in data.get("relations", []):
        try:
            graph.add_edge(r["from_id"], r["to_id"],
                           r.get("relation", "RELATED_TO"),
                           float(r.get("weight", 1.0)))
        except Exception:
            pass
    graph.save()
    print(f"  ✓ Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")

    # Vector Store
    vs = VectorStore(cfg.CHROMA_DIR).init()

    # Chunk và index full text
    main_concept_id = concepts[0]["id"] if concepts else ""
    n_chunks = vs.chunk_and_add(doc_text, doc_id, main_concept_id, doc_id)

    # Index riêng từng concept description
    desc_docs = []
    seen_desc_ids = set()
    for c in concepts:
        if not c.get("description"):
            continue
        did = f"desc_{c['id']}"
        if did in seen_desc_ids:
            continue
        seen_desc_ids.add(did)
        desc_docs.append({
            "id": did,
            "text": f"{c['name']}: {c.get('description', '')}",
            "concept_id": c["id"],
            "source": doc_id,
            "verified": True,
        })
    if desc_docs:
        vs.add_docs(desc_docs)

    print(f"  ✓ ChromaDB: {n_chunks} chunks + {len(desc_docs)} concept descriptions")

    # Export merged Q&A JSON backup
    qa_json_path = out_dir / "qa_bank.json"
    merged = []
    if qa_json_path.exists():
        try:
            merged = json.loads(qa_json_path.read_text(encoding="utf-8"))
        except Exception:
            merged = []
    merged.extend(data.get("qa_bank", []))
    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Q&A JSON merged: {qa_json_path.name}")

    db.close()
    return {
        "concepts":  len(concepts),
        "qa":        len(data.get("qa_bank", [])),
        "chunks":    n_chunks,
        "graph":     str(cfg.GRAPH_PATH),
        "db":        str(cfg.DB_PATH),
    }


# ══════════════════════════════════════════════════════════════
# QUALITY CHECK
# ══════════════════════════════════════════════════════════════
def quality_check(data: dict, topic: str) -> list:
    """Kiểm tra chất lượng extraction và in warning."""
    warnings = []
    concepts = data.get("concepts", [])
    qa       = data.get("qa_bank", [])
    relations = data.get("relations", [])

    if len(concepts) < 3:
        warnings.append(f"Ít concepts ({len(concepts)}). Tài liệu có thể quá ngắn.")
    if len(qa) < 5:
        warnings.append(f"Ít Q&A ({len(qa)}). Robot sẽ thiếu câu hỏi trong quiz.")
    if concepts:
        density = len(relations) / max(1, len(concepts))
        if density < 1.3:
            warnings.append(f"Graph phẳng: edges/concepts={density:.2f} (<1.30).")

    rel_types = {str(r.get("relation", "")).upper() for r in relations}
    if len(rel_types) < 3 and relations:
        warnings.append("Đa dạng relation thấp (<3 loại), khả năng suy luận graph yếu.")

    # Check duplicate ids
    concept_ids = [c["id"] for c in concepts]
    if len(set(concept_ids)) < len(concept_ids):
        warnings.append("Có concept_id bị trùng — đã tự động deduplicate.")

    # Check Q&A có concept_id hợp lệ
    valid_ids = set(concept_ids)
    invalid_qa = [q for q in qa if q.get("concept_id") not in valid_ids]
    if invalid_qa:
        warnings.append(f"{len(invalid_qa)} Q&A có concept_id không hợp lệ — đã bỏ qua.")

    return warnings


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="EduRobot — Offline Knowledge Build với Ollama (local, free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/offline_build_ollama.py --input docs/bai1.txt --topic "động vật" --age 6
  python scripts/offline_build_ollama.py --input docs/bai2.pdf --topic "cây cối" --age 8 --model qwen2.5:14b

Setup:
  ollama pull qwen2.5:14b   # chỉ cần pull 1 lần (~9GB)
  ollama serve               # nếu Ollama chưa chạy
        """
    )
    parser.add_argument("--input", required=True, action="append",
                        help="File hoặc thư mục tài liệu (.txt/.pdf). Có thể truyền nhiều lần.")
    parser.add_argument("--topic", default=None,
                        help="Chủ đề chung (optional). Nếu bỏ trống sẽ infer theo tên file.")
    parser.add_argument("--age",     type=int, default=6, help="Tuổi tối thiểu (default: 6)")
    parser.add_argument("--output",  default="data", help="Thư mục knowledge chung (default: data/)")
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--doc-id",  default=None, help="Document ID (chỉ dùng khi input là 1 file)")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL,
                        help=f"Ollama URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--recursive", action="store_true",
                        help="Khi input là thư mục, quét đệ quy.")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Bỏ qua manifest, build lại toàn bộ file.")

    args = parser.parse_args()

    out_dir = Path(args.output)
    files = []
    for inp in args.input:
        input_path = Path(inp)
        files.extend(list_input_files(input_path, recursive=args.recursive))
    files = sorted({str(f.resolve()): f for f in files}.values(), key=lambda p: str(p))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .txt/.pdf trong inputs: {args.input}")

    manifest = load_manifest(out_dir)

    print(f"\n{'='*56}")
    print(f"  EduRobot — Offline Knowledge Build (Ollama)")
    print(f"{'='*56}")
    print(f"  Input:   {args.input}")
    print(f"  Files:   {len(files)}")
    print(f"  Topic:   {args.topic or '[infer từ tên file]'}  (age {args.age}+)")
    print(f"  Model:   {args.model}")
    print(f"  Doc ID:  [auto per file]")
    print(f"  Output:  {out_dir}/")
    print(f"{'='*56}\n")

    # ── [0] Khởi tạo client ────────────────────────────────
    client = OllamaClient(model=args.model, base_url=args.ollama_url)

    total_concepts = total_qa = total_chunks = 0
    all_warnings = []
    processed = skipped = 0

    for idx, fp in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {fp}")
        raw_bundle = load_source_bundle(fp)
        bundle = normalize_bundle_fields(raw_bundle, fp, args.topic, args.age)
        doc_text = bundle["text"]
        h = file_hash(fp)
        key = str(fp.resolve())

        old = manifest.get("files", {}).get(key, {})
        if (not args.full_rebuild) and old.get("hash") == h:
            print("  - skip (unchanged)")
            skipped += 1
            continue

        topic = args.topic or bundle["topic"]
        domain = bundle["domain"]
        doc_id = args.doc_id if (args.doc_id and len(files) == 1) else bundle.get("doc_id") or f"doc_{domain}_{slugify(fp.stem)}"

        if bundle["structured"] and bundle.get("concepts"):
            entity_data = {
                "concepts": bundle.get("concepts", []),
                "relations": bundle.get("relations", []),
            }
        else:
            entity_data = extract_entities(client, doc_text, topic, args.age)

        if bundle["structured"] and bundle.get("qa_bank"):
            qa_data = {"qa_bank": bundle.get("qa_bank", [])}
        else:
            qa_data = generate_qa_bank(
                client, doc_text,
                entity_data.get("concepts", []),
                topic, args.age
            )
        all_data = {**entity_data, **qa_data}
        all_warnings.extend(quality_check(all_data, topic))

        result = build_and_index(doc_text, doc_id, all_data, out_dir, topic)
        total_concepts += result["concepts"]
        total_qa += result["qa"]
        total_chunks += result["chunks"]
        processed += 1

        manifest.setdefault("files", {})[key] = {
            "hash": h,
            "doc_id": doc_id,
            "topic": topic,
            "updated_at": int(time.time()),
        }
        save_manifest(out_dir, manifest)

    # ── Summary ────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  ✅  Build complete!")
    print(f"     Processed files : {processed}")
    print(f"     Skipped files   : {skipped}")
    print(f"     Concepts added  : {total_concepts}")
    print(f"     Q&A items added : {total_qa}")
    print(f"     Chunks added    : {total_chunks}")

    if all_warnings:
        print(f"\n  Warnings ({len(all_warnings)}):")
        for w in all_warnings[:20]:
            print(f"    [!] {w}")

    print(f"\n  Sync sang Jetson:")
    print(f"    rsync -avz {out_dir}/ user@jetson:~/edubot/data/")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()