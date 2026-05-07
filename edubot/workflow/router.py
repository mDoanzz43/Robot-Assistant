"""
workflow/router.py — Rule-based Query Router
Quyết định PATH_A / PATH_B / PATH_C theo rule, không theo LLM.
LlamaIndex: mỗi path gọi đúng named tool.
"""
from typing import Dict, Optional
from loguru import logger

import sys
from pathlib import Path as FPath
sys.path.insert(0, str(FPath(__file__).parent.parent))

from core.intent import Intent


class Path:
    A = "no_llm"      # < 50ms — template, graph, QA bank
    B = "with_llm"    # 2–5s  — LLM grounded
    C = "cloud"       # 5s+   — out-of-KB, cloud fallback


class Router:
    """
    Rule-based routing. 3 inputs → 1 decision.
      - text    : user utterance
      - intent  : classified intent
      - phase   : current workflow phase
    """

    def __init__(self, tools):
        self.tools = tools

    def route(self, text: str, intent: Intent, phase: str,
              child_id: str,
              concept_id: Optional[str] = None,
              asked_qa: list = None) -> Dict:
        """
        Returns:
          {"path": PATH, "reason": str, "kwargs": dict}
        """
        asked = asked_qa or []

        # ── IDLE ──────────────────────────────────────────────
        if phase == "idle":
            return self._path(Path.A, "greeting", {})

        # ── GREETING ──────────────────────────────────────────
        if intent == Intent.GREETING:
            return self._path(Path.A, "greeting", {})

        # ── TEACH phase ───────────────────────────────────────
        if phase == "teach":
            if intent in (Intent.TEACHING, Intent.UNKNOWN):
                return self._path(Path.A, "teach_ack",
                                  {"text": text, "concept_id": concept_id})
            if intent == Intent.ASKING:
                return self._route_question(text, concept_id)

        # ── CONFUSE phase ─────────────────────────────────────
        if phase == "confuse":
            return self._path(Path.A, "confuse",
                              {"concept_id": concept_id})

        # ── QUIZ phase ────────────────────────────────────────
        if phase == "quiz":
            if intent == Intent.ANSWERING:
                words = len(text.split())
                if words <= 12:
                    return self._path(Path.A, "eval_short",
                                      {"text": text, "concept_id": concept_id})
                else:
                    return self._path(Path.B, "eval_long",
                                      {"text": text, "concept_id": concept_id})
            if intent == Intent.ASKING:
                return self._route_question(text, concept_id)

        # ── REWARD phase ──────────────────────────────────────
        if phase == "reward":
            return self._path(Path.A, "reward", {})

        # ── Default ───────────────────────────────────────────
        return self._route_question(text, concept_id)

    def route_next_question(self, concept_id: str, child_id: str,
                            asked_qa: list) -> Dict:
        """Riêng cho việc chọn câu hỏi tiếp theo trong QUIZ phase."""
        # Lấy mastery → decide difficulty
        m = self.tools.get_mastery(child_id, concept_id)
        diff = None
        if m.ok and m.data:
            p = m.data.get("p_mastery", 0.3)
            from config import BKT_EASY_THR, BKT_HARD_THR
            if p < BKT_EASY_THR:    diff = 2
            elif p < BKT_HARD_THR:  diff = 3
            else:                   diff = 4

        # Check QA bank
        qa_r = self.tools.get_qa(concept_id, difficulty=diff,
                                 exclude=asked_qa, limit=1)
        if qa_r.ok and qa_r.data and qa_r.data.get("has"):
            return self._path(Path.A, "qa_bank",
                              {"qa": qa_r.data["items"][0],
                               "concept_id": concept_id})

        # QA bank empty → LLM adaptive
        return self._path(Path.B, "qa_llm",
                          {"concept_id": concept_id, "difficulty": diff})

    def _route_question(self, text: str,
                        concept_id: Optional[str]) -> Dict:
        """Route câu hỏi của trẻ → check KB trước."""
        vec = self.tools.search_concepts(text, n=1, concept_id=concept_id)
        if concept_id and (not vec.ok or not vec.data or not vec.data.get("results")):
            # Fallback to global search only when locked-topic search has no hit.
            vec = self.tools.search_concepts(text, n=1)
        if vec.ok and vec.data:
            conf = vec.data.get("confidence", "none")
            if conf in ("high", "low"):
                return self._path(Path.B, f"question_{conf}",
                                  {"text": text, "context": vec.ctx,
                                   "sim": vec.data.get("top_score", 0.5),
                                   "confidence": conf,
                                   "concept_id": concept_id})
        return self._path(Path.C, "question_oob", {"text": text})

    @staticmethod
    def _path(p: str, reason: str, kwargs: dict) -> Dict:
        return {"path": p, "reason": reason, "kwargs": kwargs}
