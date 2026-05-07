"""tests/test_router.py — Intent Classifier + Query Router tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from core.intent import IntentClassifier, Intent
from workflow.router import Router, Path as RPath


@pytest.fixture
def clf():
    return IntentClassifier()


class TestIntentClassifier:
    # ── Teaching ──────────────────────────────────────────────
    def test_teaching_basic(self, clf):
        intent, conf = clf.classify(
            "Con mèo là động vật có 4 chân và ăn cá.", "teach")
        assert intent == Intent.TEACHING

    def test_teaching_with_la(self, clf):
        intent, _ = clf.classify("Cá là sinh vật sống dưới nước.", "teach")
        assert intent == Intent.TEACHING

    def test_teaching_with_bao_gom(self, clf):
        intent, _ = clf.classify(
            "Động vật bao gồm cá, chim, thú.", "teach")
        assert intent == Intent.TEACHING

    # ── Asking ────────────────────────────────────────────────
    def test_asking_question_mark(self, clf):
        intent, conf = clf.classify("Con mèo là gì?", "teach")
        assert intent == Intent.ASKING
        assert conf >= 0.4

    def test_asking_tai_sao(self, clf):
        intent, _ = clf.classify("Tại sao cá sống dưới nước?", "teach")
        assert intent == Intent.ASKING

    def test_asking_english(self, clf):
        intent, _ = clf.classify("What is a cat?", "teach")
        assert intent == Intent.ASKING

    def test_asking_without_mark(self, clf):
        intent, _ = clf.classify("Bạn biết con cá là gì không", "teach")
        assert intent == Intent.ASKING

    # ── Answering ─────────────────────────────────────────────
    def test_answering_short_in_quiz(self, clf):
        intent, conf = clf.classify("con mèo", "quiz")
        assert intent == Intent.ANSWERING
        assert conf >= 0.8

    def test_answering_medium_in_quiz(self, clf):
        intent, _ = clf.classify("Con mèo là động vật nuôi trong nhà", "quiz")
        assert intent == Intent.ANSWERING

    def test_question_in_quiz_overrides(self, clf):
        intent, _ = clf.classify("Đây là gì?", "quiz")
        assert intent == Intent.ASKING

    # ── Greeting ──────────────────────────────────────────────
    def test_greeting_xin_chao(self, clf):
        intent, conf = clf.classify("Xin chào", "teach")
        assert intent == Intent.GREETING
        assert conf >= 0.9

    def test_greeting_hello(self, clf):
        intent, _ = clf.classify("Hello", "teach")
        assert intent == Intent.GREETING

    def test_greeting_hi_robot(self, clf):
        intent, _ = clf.classify("Hi robot!", "teach")
        assert intent == Intent.GREETING

    # ── Edge cases ────────────────────────────────────────────
    def test_empty_text(self, clf):
        intent, conf = clf.classify("", "teach")
        assert intent == Intent.UNKNOWN
        assert conf == 0.0

    def test_whitespace_only(self, clf):
        intent, _ = clf.classify("   ", "teach")
        assert intent == Intent.UNKNOWN

    def test_confidence_in_range(self, clf):
        for text, phase in [
            ("Con mèo là gì?", "teach"),
            ("Con mèo là động vật", "teach"),
            ("Đáp án là con mèo", "quiz"),
        ]:
            _, conf = clf.classify(text, phase)
            assert 0.0 <= conf <= 1.0


class TestRouter:
    def _make_tools(self, confidence="high", top_score=0.8):
        tools = MagicMock()
        tools.search_concepts.return_value = MagicMock(
            ok=True,
            data={"results": [{"concept_id": "c1", "score": top_score}],
                  "confidence": confidence, "top_score": top_score},
            ctx="context text"
        )
        tools.get_mastery.return_value = MagicMock(
            ok=True,
            data={"p_mastery": 0.4, "next_diff": "medium"}
        )
        tools.get_qa.return_value = MagicMock(
            ok=True,
            data={"items": [{"id": "q1", "question": "?", "answer": "a"}],
                  "has": True}
        )
        return tools

    def test_greeting_returns_path_a(self):
        r = Router(self._make_tools())
        route = r.route("xin chào", Intent.GREETING, "teach",
                        "child1")
        assert route["path"] == RPath.A
        assert route["reason"] == "greeting"

    def test_teach_teaching_intent_path_a(self):
        r = Router(self._make_tools())
        route = r.route("con mèo là động vật", Intent.TEACHING,
                        "teach", "child1", "c1")
        assert route["path"] == RPath.A

    def test_teach_asking_high_conf_path_b(self):
        r = Router(self._make_tools(confidence="high", top_score=0.85))
        route = r.route("con mèo là gì?", Intent.ASKING,
                        "teach", "child1")
        assert route["path"] == RPath.B
        assert "high" in route["reason"]

    def test_teach_asking_out_of_kb_path_c(self):
        tools = MagicMock()
        tools.search_concepts.return_value = MagicMock(
            ok=True,
            data={"results": [], "confidence": "none", "top_score": 0.1},
            ctx=""
        )
        r = Router(tools)
        route = r.route("lỗ đen vũ trụ là gì?", Intent.ASKING,
                        "teach", "child1")
        assert route["path"] == RPath.C

    def test_quiz_short_answer_path_a(self):
        r = Router(self._make_tools())
        route = r.route("con mèo", Intent.ANSWERING,
                        "quiz", "child1", "c1")
        assert route["path"] == RPath.A
        assert route["reason"] == "eval_short"

    def test_quiz_long_answer_path_b(self):
        r = Router(self._make_tools())
        long = "con mèo là động vật có 4 chân rất đáng yêu và hay bắt chuột"
        route = r.route(long, Intent.ANSWERING, "quiz", "child1", "c1")
        assert route["path"] == RPath.B
        assert route["reason"] == "eval_long"

    def test_route_next_question_uses_bank(self):
        tools = self._make_tools()
        r = Router(tools)
        route = r.route_next_question("c1", "child1", [])
        assert route["reason"] == "qa_bank"
        assert route["kwargs"]["qa"]["id"] == "q1"

    def test_route_next_question_llm_when_bank_empty(self):
        tools = MagicMock()
        tools.get_mastery.return_value = MagicMock(
            ok=True, data={"p_mastery": 0.5})
        tools.get_qa.return_value = MagicMock(
            ok=True, data={"items": [], "has": False})
        r = Router(tools)
        route = r.route_next_question("c1", "child1", [])
        assert route["path"] == RPath.B
        assert route["reason"] == "qa_llm"

    def test_reward_phase_returns_path_a(self):
        r = Router(self._make_tools())
        route = r.route("", Intent.UNKNOWN, "reward", "child1")
        assert route["path"] == RPath.A
        assert route["reason"] == "reward"
