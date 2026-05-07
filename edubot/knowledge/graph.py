"""
knowledge/graph.py — NetworkX Knowledge Graph Runtime
Build offline trên PC, load nhanh trên Jetson (~50MB RAM).
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set
import networkx as nx
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAPH_PATH


class KnowledgeGraph:
    """
    DiGraph: node = concept, edge = relation.
    Relation types: IS_A, HAS, CAN, LIVES_IN, OPPOSITE_OF, RELATED_TO
    """

    RELATION_VI = {
        "IS_A":        "là một loại",
        "HAS":         "có",
        "CAN":         "có thể",
        "LIVES_IN":    "sống ở",
        "OPPOSITE_OF": "trái nghĩa với",
        "RELATED_TO":  "liên quan đến",
    }

    def __init__(self, path: Path = GRAPH_PATH):
        self.path = path
        self.G: nx.DiGraph = nx.DiGraph()
        self._name_idx: Dict[str, str] = {}   # name.lower() → id
        self._ready = False

    # ── Load / Save ───────────────────────────────────────────
    def load(self) -> "KnowledgeGraph":
        if not self.path.exists():
            logger.warning(f"Graph not found: {self.path} — empty graph")
            self._ready = True
            return self

        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        for n in data.get("nodes", []):
            nid = n.pop("id")
            self.G.add_node(nid, **n)
            if "name" in n:
                self._name_idx[n["name"].lower()] = nid

        for e in data.get("edges", []):
            self.G.add_edge(e["from"], e["to"],
                            relation=e.get("relation", "RELATED_TO"),
                            weight=e.get("weight", 1.0))

        self._ready = True
        logger.info(f"Graph: {self.G.number_of_nodes()} nodes, "
                    f"{self.G.number_of_edges()} edges")
        return self

    def save(self, path: Path = None):
        out = path or self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        nodes = [{"id": nid, **attrs}
                 for nid, attrs in self.G.nodes(data=True)]
        edges = [{"from": u, "to": v, **d}
                 for u, v, d in self.G.edges(data=True)]
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f,
                      ensure_ascii=False, indent=2)
        logger.info(f"Graph saved: {out}")

    # ── Node management ───────────────────────────────────────
    def add_node(self, id: str, name: str, description: str = "",
                 difficulty: int = 3, verified: bool = True, **kw):
        self.G.add_node(id, name=name, description=description,
                        difficulty=difficulty, verified=verified, **kw)
        self._name_idx[name.lower()] = id

    def add_edge(self, from_id: str, to_id: str,
                 relation: str, weight: float = 1.0):
        if from_id in self.G and to_id in self.G:
            self.G.add_edge(from_id, to_id, relation=relation, weight=weight)

    # ── Lookup ────────────────────────────────────────────────
    def find_id(self, name: str) -> Optional[str]:
        nl = name.lower()
        if nl in self._name_idx:
            return self._name_idx[nl]
        for k, v in self._name_idx.items():
            if nl in k or k in nl:
                return v
        return None

    def get_node(self, id: str) -> Optional[Dict]:
        if id not in self.G:
            return None
        return {"id": id, **self.G.nodes[id]}

    def neighbors(self, id: str, relation: str = None,
                  depth: int = 1) -> List[Dict]:
        if id not in self.G:
            return []
        results = []
        if depth == 1:
            for _, nb, ed in self.G.out_edges(id, data=True):
                if relation and ed.get("relation") != relation:
                    continue
                nd = self.G.nodes.get(nb, {})
                results.append({
                    "id": nb, "name": nd.get("name", nb),
                    "relation": ed.get("relation", "RELATED_TO"),
                    "weight": ed.get("weight", 1.0),
                    "description": nd.get("description", "")
                })

            # Build scripts often encode taxonomy as child -> parent via IS_A.
            # Include incoming IS_A as children of the current concept so the
            # runtime can still ask meaningful follow-up questions.
            if relation in (None, "IS_A"):
                for child, _, ed in self.G.in_edges(id, data=True):
                    if ed.get("relation") != "IS_A":
                        continue
                    nd = self.G.nodes.get(child, {})
                    results.append({
                        "id": child,
                        "name": nd.get("name", child),
                        "relation": "HAS_CHILD",
                        "weight": ed.get("weight", 1.0),
                        "description": nd.get("description", "")
                    })
        else:
            visited: Set[str] = {id}
            q = [(id, 0)]
            while q:
                cur, d = q.pop(0)
                if d >= depth:
                    continue
                for _, nb, ed in self.G.out_edges(cur, data=True):
                    if nb in visited:
                        continue
                    if relation and ed.get("relation") != relation:
                        continue
                    visited.add(nb)
                    nd = self.G.nodes.get(nb, {})
                    results.append({
                        "id": nb, "name": nd.get("name", nb),
                        "relation": ed.get("relation", "RELATED_TO"),
                        "weight": ed.get("weight", 1.0),
                        "depth": d + 1,
                        "description": nd.get("description", "")
                    })
                    q.append((nb, d + 1))

        results.sort(key=lambda x: x.get("weight", 1.0), reverse=True)
        return results

    def context_text(self, id: str) -> str:
        """Build text context từ node + neighbors → cho LLM prompt."""
        node = self.get_node(id)
        if not node:
            return ""
        lines = [f"Khái niệm: {node.get('name', '')}"]
        if node.get("description"):
            lines.append(f"Mô tả: {node['description']}")
        nbs = self.neighbors(id, depth=1)[:5]
        if nbs:
            facts = []
            for n in nbs:
                rel_vi = self.RELATION_VI.get(n["relation"], n["relation"])
                facts.append(f"- {node.get('name','')} {rel_vi} {n['name']}")
            lines.append("Quan hệ:\n" + "\n".join(facts))
        return "\n".join(lines)

    def confusion_target(self, id: str) -> Optional[Dict]:
        """Chọn sub-concept để robot 'giả ngố' hỏi."""
        nbs = self.neighbors(id, depth=1)
        if not nbs:
            nbs = self.neighbors(id, depth=2)
        return random.choice(nbs) if nbs else None

    def text_search(self, text: str, limit=5) -> List[Dict]:
        tl = text.lower()
        results = []
        for nid, attrs in self.G.nodes(data=True):
            name = attrs.get("name", "").lower()
            desc = attrs.get("description", "").lower()
            sc = 0.0
            if tl in name or name in tl:
                sc = 1.0
            elif any(w in name for w in tl.split()):
                sc = 0.5
            elif any(w in desc for w in tl.split()):
                sc = 0.3
            if sc > 0:
                results.append({"id": nid, "score": sc, **attrs})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def add_child_node(self, id: str, name: str, desc: str,
                       parent_id: str = None):
        """Thêm concept trẻ dạy (unverified)."""
        self.add_node(id, name, desc, verified=False, source="child")
        if parent_id and parent_id in self.G:
            self.add_edge(id, parent_id, "IS_A", weight=0.4)

    @property
    def ready(self): return self._ready
    @property
    def n_nodes(self): return self.G.number_of_nodes()
    @property
    def n_edges(self): return self.G.number_of_edges()
