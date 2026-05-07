"""
core/intent.py — Rule-based Intent Classifier
Không dùng LLM. Latency < 5ms.
"""
from enum import Enum, auto
from typing import Tuple
import re


class Intent(Enum):
    TEACHING  = auto()  # trẻ đang giải thích / dạy
    ASKING    = auto()  # trẻ đặt câu hỏi
    ANSWERING = auto()  # trẻ trả lời quiz
    GREETING  = auto()  # chào hỏi
    CHITCHAT  = auto()  # chuyện phiếm
    UNKNOWN   = auto()


_GREET_VI = {"xin chào", "chào", "hello", "hi", "alo", "hey", "oke", "xin", "sin", "sin chào", "sin trào"}
_GREET_EN = {"hello", "hi", "hey", "good morning", "good afternoon"}
# Bigrams that are greetings (first two words joined)
_GREET_BIGRAMS = {"xin chào", "good morning", "good afternoon"}

_TEACH_KW = [
    "là", "gọi là", "có nghĩa là", "tức là", "đó là",
    "bao gồm", "gồm có", "ví dụ", "chẳng hạn",
    "sống ở", "ăn", "có thể", "được làm từ",
    "mình biết", "sách nói", "thầy dạy",
    "thuộc", "loại", "nhóm", "con", "cái"
    
    #english
    "mean", "means", "meaning"
]

_QUESTION_KW = [
    # vietnamese
    "là gì", "là ai", "như thế nào", "tại sao", "vì sao",
    "ở đâu", "khi nào", "bao nhiêu", "có không", "được không",
    "nghĩa là gì", "khác nhau", "giống nhau",
    "cho mình biết", "bạn biết", "robot biết", "giải thích",
    "muốn biết", "cho em hỏi", "mình hỏi", "tôi hỏi", "mình muốn hỏi",
    "mình muốn biết", "tôi muốn biết", "cho hỏi", "hỏi là",
    
    # english
    "what is", "what are", "how does", "why", "where", "when",
    "tell me", "explain"
]


class IntentClassifier:
    """
    Phân loại intent bằng keyword matching.
    Phase context: giúp quyết định khi ambiguous.
    """

    def classify(self, text: str, phase: str = "teach") -> Tuple[Intent, float]:
        if not text or not text.strip():
            return Intent.UNKNOWN, 0.0

        t = text.lower().strip()
        words = t.split()

        # ── Greeting ─────────────────────────────────────────
        first_word  = words[0] if words else ""
        first_two   = " ".join(words[:2]) if len(words) >= 2 else ""
        is_greeting = (
            (first_word in _GREET_VI | _GREET_EN and len(words) <= 4)
            or (first_two.lower() in _GREET_BIGRAMS and len(words) <= 4)
        )
        if is_greeting:
            return Intent.GREETING, 0.95

        # ── Quiz phase: mặc định là answering ────────────────
        if phase == "quiz":
            q_score = self._question_score(t)
            if q_score >= 0.6:
                return Intent.ASKING, q_score
            return Intent.ANSWERING, 0.85

        # ── Question ─────────────────────────────────────────
        q_score = self._question_score(t)
        ask_thr = 0.3 if phase in ("teach", "confuse") else 0.4
        if q_score >= ask_thr:
            return Intent.ASKING, min(q_score, 0.95)

        # ── Teaching ─────────────────────────────────────────
        t_hits = sum(1 for kw in _TEACH_KW if kw in t)
        t_score = min(t_hits * 0.3, 0.9)
        if t_score >= 0.3 and len(words) > 3:
            return Intent.TEACHING, t_score

        # ── Default by phase ──────────────────────────────────
        if phase in ("teach", "confuse"):
            return Intent.TEACHING, 0.50
        return Intent.UNKNOWN, 0.30

    def _question_score(self, text: str) -> float:
        has_mark = float("?" in text) * 0.4
        kw_hits  = sum(1 for kw in _QUESTION_KW if kw in text)
        wh_hits = len(re.findall(r"\b(ai|gì|sao|nào|đâu|bao nhiêu|khi nào|tại sao|vì sao)\b", text))
        tail_q = 0.25 if re.search(r"\b(nào|không|chưa|hả|à)\s*\??$", text.strip()) else 0.0
        return has_mark + min(kw_hits * 0.2, 0.6) + min(wh_hits * 0.15, 0.45) + tail_q
