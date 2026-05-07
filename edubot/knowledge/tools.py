"""
knowledge/tools.py — Tool Registry (LlamaIndex pattern)
5 named tools, mỗi tool testable độc lập.
Router gọi theo RULE, không theo LLM.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ToolResult:
    name:       str
    ok:         bool
    data:       Any    = None
    ctx:        str    = ""      # context string sẵn sàng đưa vào prompt
    latency_ms: float  = 0.0
    error:      str    = None


class Tools:
    """
    5 tools:
      search_concepts  — vector semantic search
      traverse_graph   — graph neighbors
      get_qa           — Q&A bank lookup
      get_mastery      — BKT state
      get_episode      — session history
    + hybrid_search    — combo vector + graph
    """

    def __init__(self, vector, graph, qa_dao, mastery_mgr, session_dao):
        self.vs      = vector       # VectorStore
        self.graph   = graph        # KnowledgeGraph
        self.qa      = qa_dao       # QADAO
        self.mastery = mastery_mgr  # MasteryManager
        self.ses     = session_dao  # SessionDAO

    # ══════════════════════════════════════════════════════════
    # TOOL 1 — search_concepts  (< 50ms)
    # ══════════════════════════════════════════════════════════
    def search_concepts(self, query: str, n=3,
                        concept_id: str = None) -> ToolResult:
        t0 = time.perf_counter()
        try:
            results, conf = self.vs.search_graded(query, n=n, concept_id=concept_id)
            ctx = self.vs.context_str(results) if results else ""
            top_score = results[0]["score"] if results else 0.0
            return ToolResult(
                name="search_concepts", ok=True,
                data={"results": results, "confidence": conf,
                      "top_score": top_score},
                ctx=ctx,
                latency_ms=round((time.perf_counter()-t0)*1000, 1)
            )
        except Exception as e:
            return ToolResult("search_concepts", False, error=str(e),
                              latency_ms=(time.perf_counter()-t0)*1000)

    # ══════════════════════════════════════════════════════════
    # TOOL 2 — traverse_graph  (< 30ms)
    # ══════════════════════════════════════════════════════════
    def traverse_graph(self, concept_id: str = None,
                       name: str = None,
                       relation: str = None,
                       depth: int = 1) -> ToolResult:
        t0 = time.perf_counter()
        try:
            cid = concept_id
            if not cid and name:
                cid = self.graph.find_id(name)
            if not cid:
                return ToolResult("traverse_graph", False,
                                  error=f"Concept not found: {name}",
                                  latency_ms=(time.perf_counter()-t0)*1000)

            nbs  = self.graph.neighbors(cid, relation=relation, depth=depth)
            node = self.graph.get_node(cid)
            ctx  = self.graph.context_text(cid)
            return ToolResult(
                name="traverse_graph", ok=True,
                data={"concept_id": cid, "node": node, "neighbors": nbs},
                ctx=ctx,
                latency_ms=round((time.perf_counter()-t0)*1000, 1)
            )
        except Exception as e:
            return ToolResult("traverse_graph", False, error=str(e),
                              latency_ms=(time.perf_counter()-t0)*1000)

    # ══════════════════════════════════════════════════════════
    # TOOL 3 — get_qa  (< 5ms)
    # ══════════════════════════════════════════════════════════
    def get_qa(self, concept_id: str, difficulty: int = None,
               exclude: List[str] = None, limit=3,
               nearest_difficulty: bool = False) -> ToolResult:
        t0 = time.perf_counter()
        try:
            items = self.qa.get_for_concept(
                concept_id, difficulty=difficulty,
                exclude=exclude or [], limit=limit,
                nearest_difficulty=nearest_difficulty,
            )
            ctx = ""
            if items:
                q = items[0]
                ctx = f"Câu hỏi: {q['question']}\nĐáp án: {q['answer']}"
            return ToolResult(
                name="get_qa", ok=True,
                data={"items": items, "has": len(items) > 0},
                ctx=ctx,
                latency_ms=round((time.perf_counter()-t0)*1000, 1)
            )
        except Exception as e:
            return ToolResult("get_qa", False, error=str(e),
                              latency_ms=(time.perf_counter()-t0)*1000)

    # ══════════════════════════════════════════════════════════
    # TOOL 4 — get_mastery  (< 1ms)
    # ══════════════════════════════════════════════════════════
    def get_mastery(self, child_id: str, concept_id: str,
                    update: Optional[bool] = None) -> ToolResult:
        """update=True → correct, False → wrong, None → read only."""
        t0 = time.perf_counter()
        try:
            if update is not None:
                data = self.mastery.record(child_id, concept_id, update)
            else:
                p = self.mastery.get(child_id, concept_id)
                data = {
                    "child_id": child_id,
                    "concept_id": concept_id,
                    "p_mastery": p,
                    "mastered": self.mastery.bkt.mastered(p),
                    "next_diff": self.mastery.bkt.difficulty(p)
                }
            return ToolResult(
                name="get_mastery", ok=True, data=data,
                ctx=f"P(mastery)={data.get('p_mastery',0):.2f}",
                latency_ms=round((time.perf_counter()-t0)*1000, 1)
            )
        except Exception as e:
            return ToolResult("get_mastery", False, error=str(e),
                              latency_ms=(time.perf_counter()-t0)*1000)

    # ══════════════════════════════════════════════════════════
    # TOOL 5 — get_episode  (< 5ms)
    # ══════════════════════════════════════════════════════════
    def get_episode(self, child_id: str, n=5) -> ToolResult:
        t0 = time.perf_counter()
        try:
            taught = self.ses.recent_taught(child_id, n)
            weak   = self.mastery.dao.weak(child_id)
            ctx_parts = []
            if taught:
                concepts = [t.get("concept_id","") for t in taught if t.get("concept_id")]
                if concepts:
                    ctx_parts.append(f"Trẻ vừa nói về: {', '.join(concepts[:3])}")
            if weak:
                names = [w.get("name", w["concept_id"]) for w in weak[:2]]
                ctx_parts.append(f"Cần ôn: {', '.join(names)}")
            return ToolResult(
                name="get_episode", ok=True,
                data={"taught": taught, "weak": weak},
                ctx="\n".join(ctx_parts),
                latency_ms=round((time.perf_counter()-t0)*1000, 1)
            )
        except Exception as e:
            return ToolResult("get_episode", False, error=str(e),
                              latency_ms=(time.perf_counter()-t0)*1000)

    # ══════════════════════════════════════════════════════════
    # COMBO — hybrid_search (vector + graph)
    # ══════════════════════════════════════════════════════════
    def hybrid_search(self, query: str,
                      concept_id: str = None) -> ToolResult:
        t0 = time.perf_counter()

        vec  = self.search_concepts(query, n=3)
        grph = None

        # Resolve concept_id từ vector results hoặc graph text search
        cid = concept_id
        if not cid and vec.ok and vec.data:
            tops = vec.data.get("results", [])
            if tops and tops[0].get("concept_id"):
                cid = tops[0]["concept_id"]
        if not cid:
            matches = self.graph.text_search(query, limit=1)
            if matches:
                cid = matches[0]["id"]

        if cid:
            grph = self.traverse_graph(concept_id=cid)

        ctx_parts = []
        if vec.ctx:  ctx_parts.append(vec.ctx)
        if grph and grph.ctx: ctx_parts.append(grph.ctx)

        conf = "none"
        if vec.ok and vec.data:
            conf = vec.data.get("confidence", "none")

        return ToolResult(
            name="hybrid_search", ok=True,
            data={
                "vector":     vec.data,
                "graph":      grph.data if grph else None,
                "concept_id": cid,
                "confidence": conf
            },
            ctx="\n\n".join(ctx_parts),
            latency_ms=round((time.perf_counter()-t0)*1000, 1)
        )

    def stats(self) -> Dict:
        return {
            "vector_docs": self.vs.doc_count,
            "graph_nodes": self.graph.n_nodes,
            "graph_edges": self.graph.n_edges,
        }
