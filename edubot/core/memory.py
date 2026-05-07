"""
core/memory.py — 4-Tier Memory System + Context Budget Manager
Anthropic: "treat context as a precious, finite resource"
"""
from collections import deque
from typing import Dict, List, Optional
import re
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CTX_SYSTEM, CTX_KNOWLEDGE, CTX_HISTORY,
    CTX_TASK, CTX_TOTAL, MAX_HISTORY_TURNS
)


# ── Token estimator ────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """~4 chars/token EN, ~2 chars/token VI (unicode-heavy)."""
    if not text:
        return 0
    vi = sum(1 for c in text if ord(c) > 127)
    en = len(text) - vi
    return max(1, int(en / 4 + vi / 2))


def trim_to_tokens(text: str, max_tok: int) -> str:
    if estimate_tokens(text) <= max_tok:
        return text
    lo, hi = 0, len(text)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        (lo if estimate_tokens(text[:mid]) <= max_tok else hi) == mid or None
        if estimate_tokens(text[:mid]) <= max_tok:
            lo = mid
        else:
            hi = mid
    return text[:lo].rstrip() + "…"


# ══════════════════════════════════════════════════════════════
# TIER 1 — Working Memory (conversation turns, in-RAM)
# ══════════════════════════════════════════════════════════════
class WorkingMemory:
    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self._max  = max_turns
        self._buf: deque = deque(maxlen=max_turns * 2)  # (role, text)

    def add(self, role: str, text: str):
        self._buf.append((role, text))

    def to_str(self) -> str:
        lines = []
        for role, text in self._buf:
            prefix = "Bé" if role == "user" else "Robot"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def clear(self):
        self._buf.clear()

    @property
    def turn_count(self) -> int:
        return len(self._buf) // 2


# ══════════════════════════════════════════════════════════════
# TIER 2 — Episodic Memory (session events, written to DB)
# ══════════════════════════════════════════════════════════════
class EpisodicMemory:
    def __init__(self, session_dao, session_id: str, child_id: str):
        self.dao        = session_dao
        self.session_id = session_id
        self.child_id   = child_id
        self._taught:  List[Dict] = []
        self._errors:  List[str]  = []

    def record_taught(self, utterance: str,
                      concept_id: str = None, sim_score: float = 0.0):
        self._taught.append({"utterance": utterance,
                              "concept_id": concept_id})
        self.dao.log_taught(self.session_id, self.child_id,
                            utterance, concept_id, sim_score)

    def record_error(self, concept_id: str):
        self._errors.append(concept_id)

    def recent_concepts(self, n=3) -> List[str]:
        seen, out = set(), []
        for item in reversed(self._taught[-n:]):
            cid = item.get("concept_id")
            if cid and cid not in seen:
                seen.add(cid); out.append(cid)
        return out

    def recent_teachings(self, n=4) -> List[str]:
        """Return recent unique teaching utterances for end-session recap."""
        stop_pattern = re.compile(
            r"\b("
            r"mình chỉ biết thế thôi|mình chỉ biết vậy thôi|"
            r"hết rồi|mình hết ý|không còn gì nữa|"
            r"chỉ vậy thôi|vậy thôi|thế thôi"
            r")\b",
            flags=re.IGNORECASE,
        )
        seen, out = set(), []
        for item in reversed(self._taught):
            utt = (item.get("utterance") or "").strip()
            if not utt:
                continue
            if stop_pattern.search(utt):
                continue
            key = utt.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(utt)
            if len(out) >= n:
                break
        return list(reversed(out))

    @property
    def taught_count(self) -> int:
        return len(self._taught)


# ══════════════════════════════════════════════════════════════
# CONTEXT BUDGET MANAGER
# ══════════════════════════════════════════════════════════════
class ContextBudget:
    """
    Hard-limit token budget cho mỗi LLM call.
    Đảm bảo tổng input < LLM_CTX, mỗi slot không vượt budget.
    """

    def build(self, system: str, knowledge: str,
              history: str, task: str) -> Dict:
        s = trim_to_tokens(system,   CTX_SYSTEM)
        k = trim_to_tokens(knowledge, CTX_KNOWLEDGE)
        h = trim_to_tokens(history,   CTX_HISTORY)
        t = trim_to_tokens(task,      CTX_TASK)

        usage = {
            "system":    estimate_tokens(s),
            "knowledge": estimate_tokens(k),
            "history":   estimate_tokens(h),
            "task":      estimate_tokens(t),
        }
        usage["total"] = sum(usage.values())

        parts = []
        if s: parts.append(s)
        if k: parts.append(f"[KIẾN THỨC]\n{k}")
        if h: parts.append(f"[HỘI THOẠI]\n{h}")
        if t: parts.append(f"[YÊU CẦU]\n{t}")

        logger.debug(f"CtxBudget: {usage} / {CTX_TOTAL}")
        return {
            "prompt":   "\n\n".join(parts),
            "usage":    usage,
            "over":     usage["total"] > CTX_TOTAL
        }


# ══════════════════════════════════════════════════════════════
# MEMORY FACADE — interface duy nhất cho workflow engine
# ══════════════════════════════════════════════════════════════
class Memory:
    """Bọc cả 4 tầng memory vào 1 object."""

    def __init__(self, session_dao, session_id: str, child_id: str):
        self.working  = WorkingMemory()
        self.episodic = EpisodicMemory(session_dao, session_id, child_id)
        self.budget   = ContextBudget()
        self._asked_qa: List[str] = []

    # ── Shortcuts ─────────────────────────────────────────────
    def add_turn(self, role: str, text: str):
        self.working.add(role, text)

    def record_teaching(self, utterance: str,
                        concept_id: str = None, sim: float = 0.0):
        self.episodic.record_taught(utterance, concept_id, sim)

    def build_prompt(self, system: str, knowledge: str, task: str) -> Dict:
        return self.budget.build(
            system, knowledge, self.working.to_str(), task
        )

    def track_qa(self, qa_id: str):
        if qa_id not in self._asked_qa:
            self._asked_qa.append(qa_id)

    @property
    def asked_qa(self) -> List[str]:
        return self._asked_qa

    def reset(self):
        self.working.clear()
        self._asked_qa.clear()
