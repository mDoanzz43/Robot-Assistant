"""tests/test_memory.py — Memory system tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from core.memory import (
    WorkingMemory, EpisodicMemory,
    ContextBudget, Memory,
    estimate_tokens, trim_to_tokens
)


class TestTokenEstimator:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        t = estimate_tokens("hello world test")
        assert t > 0

    def test_vietnamese_text(self):
        vi = "con mèo là động vật có bốn chân"
        en = "a cat is an animal with four legs"
        # Vietnamese should use more tokens per char (unicode heavy)
        assert estimate_tokens(vi) > 0
        assert estimate_tokens(en) > 0

    def test_trim_no_change_when_under(self):
        text = "short text"
        result = trim_to_tokens(text, 100)
        assert result == text

    def test_trim_truncates_long_text(self):
        text = "word " * 500
        result = trim_to_tokens(text, 50)
        assert estimate_tokens(result) <= 55   # slight tolerance
        assert len(result) < len(text)

    def test_trim_adds_ellipsis(self):
        result = trim_to_tokens("word " * 500, 20)
        assert result.endswith("…")


class TestWorkingMemory:
    def test_add_and_format(self):
        wm = WorkingMemory(max_turns=3)
        wm.add("user",      "Xin chào")
        wm.add("assistant", "Chào bạn!")
        s = wm.to_str()
        assert "Bé:" in s
        assert "Robot:" in s

    def test_maxlen_evicts_oldest(self):
        wm = WorkingMemory(max_turns=2)
        for i in range(10):
            wm.add("user", f"turn {i}")
        # Buffer maxlen = max_turns * 2 = 4 entries
        s = wm.to_str()
        # Should not contain turn 0 (evicted)
        assert "turn 0" not in s

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("user", "hello")
        wm.clear()
        assert wm.to_str() == ""
        assert wm.turn_count == 0


class TestContextBudget:
    def test_builds_prompt(self):
        cb = ContextBudget()
        result = cb.build(
            system    = "Bạn là robot.",
            knowledge = "Con mèo có 4 chân.",
            history   = "Bé: chào\nRobot: chào bé",
            task      = "Con mèo là gì?"
        )
        assert "prompt" in result
        assert "usage"  in result
        assert result["usage"]["total"] > 0

    def test_usage_within_budget(self):
        from config import CTX_TOTAL
        cb = ContextBudget()
        result = cb.build("sys", "ctx", "hist", "task")
        assert result["usage"]["total"] <= CTX_TOTAL + 20  # +20 tolerance

    def test_long_context_truncated(self):
        from config import CTX_KNOWLEDGE
        cb = ContextBudget()
        long_ctx = "từ " * 1000
        result = cb.build("sys", long_ctx, "", "task")
        ctx_tokens = result["usage"]["knowledge"]
        assert ctx_tokens <= CTX_KNOWLEDGE + 5

    def test_sections_present_in_prompt(self):
        cb = ContextBudget()
        result = cb.build("system_text", "knowledge_text", "history_text", "task_text")
        p = result["prompt"]
        assert "KIẾN THỨC" in p
        assert "HỘI THOẠI" in p
        assert "YÊU CẦU"   in p


class TestEpisodicMemory:
    def _make(self):
        dao = MagicMock()
        return EpisodicMemory(dao, "sid1", "child1"), dao

    def test_record_taught_calls_dao(self):
        ep, dao = self._make()
        ep.record_taught("con mèo là động vật", "c1", 0.8)
        dao.log_taught.assert_called_once()
        assert ep.taught_count == 1

    def test_recent_concepts(self):
        ep, _ = self._make()
        ep.record_taught("text1", "c1")
        ep.record_taught("text2", "c2")
        ep.record_taught("text3", "c1")   # duplicate
        recent = ep.recent_concepts(3)
        assert "c1" in recent
        assert "c2" in recent
        assert len(recent) == 2   # deduplicated

    def test_record_error(self):
        ep, _ = self._make()
        ep.record_error("c_hard")
        assert "c_hard" in ep._errors


class TestMemoryFacade:
    def test_build_prompt_integrates_working_memory(self):
        dao = MagicMock()
        mem = Memory(dao, "s1", "c1")
        mem.add_turn("user",      "Bé hỏi")
        mem.add_turn("assistant", "Robot trả lời")
        result = mem.build_prompt("sys", "knowledge", "task")
        assert "Bé" in result["prompt"] or "Robot" in result["prompt"]

    def test_track_qa_no_duplicates(self):
        dao = MagicMock()
        mem = Memory(dao, "s1", "c1")
        mem.track_qa("qa1")
        mem.track_qa("qa1")
        mem.track_qa("qa2")
        assert len(mem.asked_qa) == 2

    def test_reset_clears_working_memory(self):
        dao = MagicMock()
        mem = Memory(dao, "s1", "c1")
        mem.add_turn("user", "hello")
        mem.track_qa("qa1")
        mem.reset()
        assert mem.working.to_str() == ""
        assert len(mem.asked_qa) == 0
