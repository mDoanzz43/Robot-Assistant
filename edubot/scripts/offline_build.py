"""
scripts/offline_build.py — Build Knowledge Graph + Q&A Bank
Chạy trên PC mạnh / cloud (KHÔNG chạy trên Jetson).
Input:  tài liệu (PDF / txt)
Output: graph.json + ChromaDB + qa_bank SQLite + qa_bank.json

Usage:
    # Trên PC, trong môi trường có google-generativeai
    export GEMINI_API_KEY=your_key
    python scripts/offline_build.py \\
        --input  docs/bai1.txt \\
        --topic  "động vật" \\
        --age    6 \\
        --output data/

    # Sau đó copy sang Jetson:
    rsync -avz data/ user@jetson:/home/user/edubot/data/
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════
# DOCUMENT LOADER
# ══════════════════════════════════════════════════════════════
def load_doc(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            pages = PdfReader(path).pages
            return "\n".join(pg.extract_text() or "" for pg in pages)
        except ImportError:
            raise RuntimeError("pip install pypdf")
    return p.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# GEMINI EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_with_gemini(text: str, topic: str, age: int,
                        api_key: str,
                        model: str = "gemini-2.0-flash") -> dict:
    """
    Gọi Gemini Flash để:
      1. Extract entities + relations → knowledge graph
      2. Generate Q&A bank (15–20 câu, có difficulty levels)
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gem = genai.GenerativeModel(model)
    except ImportError:
        raise RuntimeError("pip install google-generativeai")

    doc_snippet = text[:3500]

    # ── Step 1: Entity + Relation extraction ──────────────────
    entity_prompt = f"""Phân tích văn bản về "{topic}" cho trẻ {age} tuổi.

Trả về JSON DUY NHẤT (không có text ngoài JSON):
{{
  "concepts": [
    {{
      "id": "c_<8_char_id>",
      "name": "tên khái niệm tiếng Việt",
      "description": "mô tả 1 câu ngắn, phù hợp trẻ {age} tuổi",
      "difficulty": 1-5,
      "parent_id": "c_xxx hoặc null"
    }}
  ],
  "relations": [
    {{
      "from_id": "c_xxx",
      "to_id":   "c_yyy",
      "relation": "IS_A|HAS|CAN|LIVES_IN|OPPOSITE_OF|RELATED_TO",
      "weight":   0.5-1.0
    }}
  ]
}}

Văn bản:
{doc_snippet}"""

    print("  → [1/2] Extracting entities & relations...")
    t0 = time.time()
    resp1 = gem.generate_content(entity_prompt)
    raw1  = resp1.text.strip()
    # Clean markdown code blocks if present
    if "```" in raw1:
        raw1 = raw1.split("```")[1].lstrip("json").strip()

    try:
        ent_data = json.loads(raw1)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error (entities): {e}")
        print(f"  Raw: {raw1[:300]}")
        ent_data = {"concepts": [], "relations": []}

    n_c = len(ent_data.get("concepts", []))
    n_r = len(ent_data.get("relations", []))
    print(f"  ✓ {n_c} concepts, {n_r} relations ({time.time()-t0:.1f}s)")

    # ── Step 2: Q&A bank generation ───────────────────────────
    concept_list = "\n".join(
        f"- {c['id']}: {c['name']} — {c.get('description','')}"
        for c in ent_data.get("concepts", [])[:15]
    )

    qa_prompt = f"""Tạo 18 câu hỏi kiểm tra về "{topic}" cho trẻ {age} tuổi.

Danh sách khái niệm (dùng đúng concept_id):
{concept_list}

Văn bản tham khảo:
{doc_snippet[:2000]}

Trả về JSON DUY NHẤT:
{{
  "qa_bank": [
    {{
      "id":         "qa_<8_char_id>",
      "concept_id": "c_xxx",
      "question":   "câu hỏi rõ ràng cho trẻ {age} tuổi",
      "answer":     "đáp án CHÍNH XÁC, ngắn gọn (1-2 câu)",
      "type":       "open|mcq|fill|yesno",
      "options":    ["A","B","C","D"],
      "difficulty": 1-5
    }}
  ]
}}

Quy tắc:
- difficulty 1-2: câu dễ (yes/no, điền từ 1 chữ)
- difficulty 3:   câu trung bình
- difficulty 4-5: câu khó (giải thích, so sánh)
- options: chỉ cần cho type=mcq, để [] cho loại khác
- Đảm bảo đáp án ĐÚNG với văn bản gốc, không suy diễn thêm"""

    print("  → [2/2] Generating Q&A bank...")
    t0 = time.time()
    resp2 = gem.generate_content(qa_prompt)
    raw2  = resp2.text.strip()
    if "```" in raw2:
        raw2 = raw2.split("```")[1].lstrip("json").strip()

    try:
        qa_data = json.loads(raw2)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error (QA): {e}")
        qa_data = {"qa_bank": []}

    n_qa = len(qa_data.get("qa_bank", []))
    print(f"  ✓ {n_qa} Q&A items ({time.time()-t0:.1f}s)")

    return {**ent_data, **qa_data}


# ══════════════════════════════════════════════════════════════
# BUILD & INDEX
# ══════════════════════════════════════════════════════════════
def build_and_index(doc_text: str, doc_id: str,
                    data: dict, out_dir: Path,
                    topic: str, age_min: int, age_max: int):
    from core.database   import DB, ConceptDAO, QADAO, SessionDAO
    from knowledge.graph  import KnowledgeGraph
    from knowledge.vector import VectorStore

    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch config paths to point to out_dir
    import config as cfg
    cfg.DB_PATH    = out_dir / "edubot.db"
    cfg.GRAPH_PATH = out_dir / "graph" / "knowledge_graph.json"
    cfg.CHROMA_DIR = out_dir / "chroma"
    (out_dir / "graph").mkdir(exist_ok=True)
    (out_dir / "chroma").mkdir(exist_ok=True)

    # DB
    from core.database import DB as _DB
    _DB._instance = None   # reset singleton
    db = _DB.get()
    c_dao  = ConceptDAO(db)
    qa_dao = QADAO(db)

    # Insert document
    db.execute("INSERT OR REPLACE INTO documents(id,title) VALUES(?,?)",
               (doc_id, topic))
    db.commit()

    # Insert concepts
    concepts = data.get("concepts", [])
    for c in concepts:
        c_dao.upsert(
            id=c["id"], name=c["name"],
            desc=c.get("description",""),
            parent_id=c.get("parent_id"),
            doc_id=doc_id,
            difficulty=c.get("difficulty",3),
            verified=1
        )

    # Insert relations
    for r in data.get("relations", []):
        c_dao.add_relation(r["from_id"], r["to_id"],
                           r.get("relation","RELATED_TO"),
                           r.get("weight",1.0))

    # Insert Q&A
    qa_dao.bulk_load(data.get("qa_bank", []))
    print(f"  ✓ DB: {len(concepts)} concepts, {len(data.get('qa_bank',[]))} QAs")

    # Graph
    graph = KnowledgeGraph(cfg.GRAPH_PATH)
    for c in concepts:
        graph.add_node(c["id"], c["name"],
                       c.get("description",""),
                       c.get("difficulty",3), True)
    for r in data.get("relations",[]):
        graph.add_edge(r["from_id"], r["to_id"],
                       r.get("relation","RELATED_TO"),
                       r.get("weight",1.0))
    graph.save()
    print(f"  ✓ Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")

    # Vector store
    vs = VectorStore(cfg.CHROMA_DIR).init()
    n  = vs.chunk_and_add(doc_text, doc_id,
                          concepts[0]["id"] if concepts else "",
                          doc_id)
    # Index concept descriptions
    desc_docs = [
        {"id": f"desc_{c['id']}", "text": f"{c['name']}: {c.get('description','')}",
         "concept_id": c["id"], "source": doc_id, "verified": True}
        for c in concepts if c.get("description")
    ]
    if desc_docs:
        vs.add_docs(desc_docs)
    print(f"  ✓ Vectors: {n} chunks + {len(desc_docs)} concept descs")

    # Export Q&A JSON (fast load)
    qa_json = out_dir / "qa_bank.json"
    with open(qa_json, "w", encoding="utf-8") as f:
        json.dump(data.get("qa_bank",[]), f, ensure_ascii=False, indent=2)

    db.close()
    return {
        "concepts": len(concepts),
        "qa":       len(data.get("qa_bank",[])),
        "graph":    str(cfg.GRAPH_PATH),
        "db":       str(cfg.DB_PATH),
        "qa_json":  str(qa_json),
    }


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def main():
    import os
    parser = argparse.ArgumentParser(description="EduRobot Offline Build")
    parser.add_argument("--input",   required=True, help="PDF hoặc text file")
    parser.add_argument("--topic",   required=True, help="Chủ đề (vd: 'động vật')")
    parser.add_argument("--age",     type=int, default=6,  help="Tuổi tối thiểu")
    parser.add_argument("--age-max", type=int, default=12, help="Tuổi tối đa")
    parser.add_argument("--output",  default="data", help="Output directory")
    parser.add_argument("--doc-id",  default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: --api-key hoặc env GEMINI_API_KEY required")
        sys.exit(1)

    doc_id  = args.doc_id or f"doc_{uuid.uuid4().hex[:8]}"
    out_dir = Path(args.output)

    print(f"\n{'='*52}")
    print(f" EduRobot Offline Build")
    print(f"  Input:  {args.input}")
    print(f"  Topic:  {args.topic}  (age {args.age}–{args.age_max})")
    print(f"  Doc ID: {doc_id}")
    print(f"  Output: {out_dir}")
    print(f"{'='*52}\n")

    print("[1/3] Loading document...")
    doc_text = load_doc(args.input)
    print(f"  ✓ {len(doc_text):,} characters\n")

    print("[2/3] Extracting knowledge with Gemini Flash...")
    knowledge = extract_with_gemini(
        doc_text, args.topic, args.age, api_key
    )
    print()

    print("[3/3] Building graph + indexing vectors...")
    result = build_and_index(
        doc_text, doc_id, knowledge, out_dir,
        args.topic, args.age, args.age_max
    )

    print(f"\n{'='*52}")
    print(f"✅  Build complete!")
    print(f"    Concepts : {result['concepts']}")
    print(f"    Q&A items: {result['qa']}")
    print(f"\n  Sync sang Jetson:")
    print(f"    rsync -avz {out_dir}/ user@jetson:~/edubot/data/")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
