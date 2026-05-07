
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assessment.bkt import BKT
from assessment.evaluator import AnswerEvaluator


class _DummyVS:
    def embed_single(self, text):
        # Deterministic unit vectors for stable tests.
        t = (text or "").lower()
        if "sao thuy" in t or "thuy" in t:
            return [1.0, 0.0, 0.0]
        if "co" in t and len(t.split()) <= 2:
            return [0.0, 1.0, 0.0]
        if "khong" in t and len(t.split()) <= 3:
            return [0.0, -1.0, 0.0]
        return [0.0, 0.0, 1.0]


class TestAnswerEvaluatorDeterministic:
    def _mk(self):
        return AnswerEvaluator(_DummyVS(), BKT())

    def test_yes_no_exact(self):
        ev = self._mk()
        r = ev.evaluate(child_answer="Có", question="?", correct_answer="có")
        assert r["correct"] is True
        assert r["label"] == 1

    def test_yes_no_negative(self):
        ev = self._mk()
        r = ev.evaluate(child_answer="không", question="?", correct_answer="có")
        assert r["correct"] is False
        assert r["label"] == 0

    def test_short_keyword_match(self):
        ev = self._mk()
        r = ev.evaluate(
            child_answer="Sao Thủy",
            question="Hành tinh nào gần mặt trời nhất?",
            correct_answer="Sao Thủy là hành tinh gần Mặt trời nhất",
        )
        assert r["correct"] is True
        assert r["label"] in (1, 2)

    def test_empty_answer_wrong(self):
        ev = self._mk()
        r = ev.evaluate(child_answer="", question="?", correct_answer="sống sót")
        assert r["correct"] is False
        assert r["label"] == 0
