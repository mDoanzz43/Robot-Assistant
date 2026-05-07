"""tests/test_graph.py — KnowledgeGraph tests"""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from knowledge.graph import KnowledgeGraph


@pytest.fixture
def tmp_graph(tmp_path):
    g = KnowledgeGraph(tmp_path / "graph.json")
    g._ready = True   # skip file load
    return g


@pytest.fixture
def populated_graph(tmp_path):
    g = KnowledgeGraph(tmp_path / "graph.json")
    g._ready = True
    g.add_node("c1", "động vật",   "Sinh vật sống",        difficulty=2)
    g.add_node("c2", "con mèo",    "Động vật có 4 chân",   difficulty=2)
    g.add_node("c3", "con chó",    "Vật nuôi trung thành", difficulty=2)
    g.add_node("c4", "cá",         "Sống dưới nước",       difficulty=1)
    g.add_node("c5", "thực vật",   "Sinh vật không di chuyển", difficulty=3)
    g.add_edge("c2", "c1", "IS_A",  1.0)
    g.add_edge("c3", "c1", "IS_A",  1.0)
    g.add_edge("c4", "c1", "IS_A",  0.9)
    g.add_edge("c2", "c4", "HAS",   0.8)   # mèo ăn cá
    g.add_edge("c2", "c3", "RELATED_TO", 0.5)
    return g


class TestKnowledgeGraph:
    def test_add_and_get_node(self, tmp_graph):
        tmp_graph.add_node("n1", "con thỏ", "Động vật tai dài")
        node = tmp_graph.get_node("n1")
        assert node is not None
        assert node["name"] == "con thỏ"
        assert node["verified"] is True

    def test_find_id_exact(self, populated_graph):
        cid = populated_graph.find_id("con mèo")
        assert cid == "c2"

    def test_find_id_partial(self, populated_graph):
        cid = populated_graph.find_id("mèo")
        assert cid == "c2"

    def test_find_id_not_found(self, populated_graph):
        cid = populated_graph.find_id("rồng")
        assert cid is None

    def test_neighbors_direct(self, populated_graph):
        nbs = populated_graph.neighbors("c2", depth=1)
        ids = [n["id"] for n in nbs]
        assert "c1" in ids     # IS_A động vật
        assert "c4" in ids     # HAS cá

    def test_neighbors_filter_relation(self, populated_graph):
        nbs = populated_graph.neighbors("c2", relation="IS_A")
        assert all(n["relation"] == "IS_A" for n in nbs)
        assert len(nbs) == 1

    def test_neighbors_depth2(self, populated_graph):
        # c1 → via c2 → c4 (depth 2)
        nbs = populated_graph.neighbors("c2", depth=2)
        ids = [n["id"] for n in nbs]
        assert len(nbs) >= 2

    def test_context_text(self, populated_graph):
        ctx = populated_graph.context_text("c2")
        assert "con mèo" in ctx
        assert len(ctx) > 10

    def test_text_search(self, populated_graph):
        results = populated_graph.text_search("mèo")
        assert len(results) >= 1
        assert results[0]["id"] == "c2"

    def test_text_search_description(self, populated_graph):
        results = populated_graph.text_search("dưới nước")
        ids = [r["id"] for r in results]
        assert "c4" in ids

    def test_confusion_target(self, populated_graph):
        target = populated_graph.confusion_target("c2")
        assert target is not None
        assert "id" in target

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "g.json"
        g1 = KnowledgeGraph(path); g1._ready = True
        g1.add_node("x1", "test node", "desc")
        g1.add_node("x2", "test node 2", "")
        g1.add_edge("x1", "x2", "IS_A")
        g1.save()

        g2 = KnowledgeGraph(path)
        g2.load()
        assert g2.n_nodes == 2
        assert g2.n_edges == 1
        assert g2.find_id("test node") == "x1"

    def test_add_child_node_unverified(self, tmp_graph):
        tmp_graph.add_node("p1", "cha", "")
        tmp_graph.add_child_node("k1", "con", "mô tả", parent_id="p1")
        node = tmp_graph.get_node("k1")
        assert node["verified"] is False
        assert node.get("source") == "child"

    def test_sorted_by_weight(self, populated_graph):
        nbs = populated_graph.neighbors("c2")
        weights = [n["weight"] for n in nbs]
        assert weights == sorted(weights, reverse=True)
