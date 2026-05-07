"""tests/test_bkt.py — Bayesian Knowledge Tracing tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from assessment.bkt import BKT, MasteryManager
from unittest.mock import MagicMock


@pytest.fixture
def bkt():
    return BKT(p_init=0.3, p_transit=0.15,
               p_guess=0.25, p_slip=0.10, mastery_thr=0.70)


class TestBKT:
    def test_correct_increases_mastery(self, bkt):
        p1 = bkt.update(0.3, correct=True)
        assert p1 > 0.3

    def test_wrong_gives_valid_probability(self, bkt):
        p1 = bkt.update(0.3, correct=False)
        assert 0.0 <= p1 <= 1.0

    def test_repeated_correct_approaches_mastery(self, bkt):
        p = 0.3
        for _ in range(15):
            p = bkt.update(p, correct=True)
        assert p >= 0.70

    def test_mastery_threshold(self, bkt):
        assert not bkt.mastered(0.69)
        assert     bkt.mastered(0.70)
        assert     bkt.mastered(0.95)

    def test_difficulty_levels(self, bkt):
        assert bkt.difficulty(0.20) == 2   # easy
        assert bkt.difficulty(0.55) == 3   # medium
        assert bkt.difficulty(0.80) == 4   # hard

    def test_clamp_bounds(self, bkt):
        # Should never go below 0 or above 1
        p = bkt.update(0.001, correct=False)
        assert p >= 0.0
        p = bkt.update(0.999, correct=True)
        assert p <= 1.0

    def test_simulate(self, bkt):
        trace = bkt.simulate(0.3, [True, True, False, True, True])
        assert len(trace) == 6
        assert all(0 <= v <= 1 for v in trace)
        # Net positive → final p higher than initial
        assert trace[-1] > trace[0]

    def test_eval_sim_correct(self, bkt):
        label, ok = bkt.eval_sim(0.80)
        assert label == 1 and ok is True

    def test_eval_sim_partial(self, bkt):
        label, ok = bkt.eval_sim(0.55)
        assert label == 2 and ok is True

    def test_eval_sim_wrong(self, bkt):
        label, ok = bkt.eval_sim(0.20)
        assert label == 0 and ok is False

    def test_p_stays_positive_after_many_wrong(self, bkt):
        p = 0.3
        for _ in range(30):
            p = bkt.update(p, correct=False)
        assert p > 0.0


class TestMasteryManager:
    def test_get_returns_p_init_for_new(self):
        mock_dao = MagicMock()
        mock_dao.get.return_value = None
        mgr = MasteryManager(mock_dao)
        p = mgr.get("child1", "concept1")
        assert p == mgr.bkt.p_init

    def test_record_calls_dao_upsert(self):
        mock_dao = MagicMock()
        mock_dao.get.return_value = None
        mgr = MasteryManager(mock_dao)
        result = mgr.record("child1", "concept1", correct=True)
        assert result["correct"] is True
        assert result["p_after"] > result["p_before"]
        mock_dao.upsert.assert_called_once()

    def test_cache_used_on_second_get(self):
        mock_dao = MagicMock()
        mock_dao.get.return_value = None
        mgr = MasteryManager(mock_dao)
        _ = mgr.get("c", "x")
        _ = mgr.get("c", "x")
        # DB called only once
        assert mock_dao.get.call_count == 1

    def test_just_mastered_flag(self):
        mock_dao = MagicMock()
        mock_dao.get.return_value = None
        mgr = MasteryManager(mock_dao)
        # Push p to just below threshold, then one correct
        mgr._cache["c:x"] = 0.68
        result = mgr.record("c", "x", correct=True)
        if result["p_after"] >= mgr.bkt.thr:
            assert result["just_mastered"] is True
