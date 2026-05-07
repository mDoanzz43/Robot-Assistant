"""
assessment/evaluator.py — Answer Evaluator
Deterministic-first grading to avoid LLM hallucination.
"""
import re
import unicodedata
from typing import Dict, Optional, Tuple
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AnswerEvaluator:
    """
    Đánh giá câu trả lời của trẻ.
    Không import LLM ở đây — pass từ ngoài vào để tránh circular import.
    """

    SCORE_CORRECT = 10
    SCORE_PARTIAL = 5
    SCORE_WRONG   = 0

    def __init__(self, vector_store, bkt_engine, llm_engine=None):
        self.vs  = vector_store
        self.bkt = bkt_engine
        self.llm = llm_engine   # optional, set sau khi LLM loaded

    def evaluate(
        self,
        child_answer: str,
        question:     str,
        correct_answer: str,
        use_llm: bool = False
    ) -> Dict:
        """
        Returns:
            correct (bool), label (1/2/0), score_delta, feedback (str)
        """
        if correct_answer:
            return self._deterministic_eval(child_answer, correct_answer)

        if use_llm and self.llm and self.llm.ready:
            return self._llm_eval(child_answer, question, correct_answer)

        return {
            "correct": False, "label": 0,
            "score_delta": self.SCORE_WRONG,
            "feedback": None,
            "hint": ""
        }

    def _deterministic_eval(self, child_answer: str, correct_answer: str) -> Dict:
        child_norm = self._normalize_text(child_answer)
        correct_norm = self._normalize_text(correct_answer)

        if not child_norm:
            return {
                "correct": False, "label": 0,
                "score_delta": self.SCORE_WRONG,
                "feedback": None,
                "hint": correct_answer.split()[0] if correct_answer else "",
            }

        yes_no = self._eval_yes_no(child_norm, correct_norm)
        if yes_no is not None:
            label = 1 if yes_no else 0
            return {
                "correct": yes_no,
                "label": label,
                "score_delta": self.SCORE_CORRECT if yes_no else self.SCORE_WRONG,
                "feedback": None,
                "hint": "" if yes_no else (correct_answer.split()[0] if correct_answer else ""),
            }

        exact_or_contains = (
            child_norm == correct_norm
            or child_norm in correct_norm
            or correct_norm in child_norm
        )
        overlap = self._token_overlap(child_norm, correct_norm)
        sim = self._safe_similarity(child_answer, correct_answer)

        if exact_or_contains or sim >= 0.72:
            return {
                "correct": True, "label": 1,
                "score_delta": self.SCORE_CORRECT,
                "feedback": None,
                "sim": sim,
                "overlap": overlap,
            }

        if overlap >= 0.5 or sim >= 0.45:
            return {
                "correct": True, "label": 2,
                "score_delta": self.SCORE_PARTIAL,
                "feedback": None,
                "sim": sim,
                "overlap": overlap,
            }

        return {
            "correct": False, "label": 0,
            "score_delta": self.SCORE_WRONG,
            "feedback": None,
            "hint": correct_answer.split()[0] if correct_answer else "",
            "sim": sim,
            "overlap": overlap,
        }

    def _safe_similarity(self, child_answer: str, correct_answer: str) -> float:
        try:
            import numpy as np
            vecs = self.vs.embed_single(child_answer), self.vs.embed_single(correct_answer)
            return float(np.dot(vecs[0], vecs[1]))
        except Exception as e:
            logger.warning(f"Sim eval error: {e}")
            return 0.0

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        base = max(1, min(len(a_tokens), len(b_tokens)))
        return inter / base

    @staticmethod
    def _normalize_text(text: str) -> str:
        t = (text or "").strip().lower()
        t = unicodedata.normalize("NFD", t)
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _eval_yes_no(self, child_norm: str, correct_norm: str) -> Optional[bool]:
        yes_set = {"co", "c", "yes", "dung", "phai", "chinh xac"}
        no_set = {"khong", "ko", "k", "no", "sai", "khong phai", "chua"}

        target = None
        if correct_norm in yes_set:
            target = True
        elif correct_norm in no_set:
            target = False
        if target is None:
            return None

        child_yes = any(k in child_norm.split() for k in yes_set) or "dung" in child_norm
        child_no = any(k in child_norm.split() for k in no_set) or "khong" in child_norm
        if child_yes and not child_no:
            return target is True
        if child_no and not child_yes:
            return target is False
        return None

    def _sim_eval(self, child_answer: str, correct_answer: str) -> Dict:
        """Cosine similarity giữa câu trả lời và đáp án đúng."""
        try:
            import numpy as np
            vecs = self.vs.embed_single(child_answer), \
                   self.vs.embed_single(correct_answer)
            sim = float(np.dot(vecs[0], vecs[1]))   # already L2-normalized
        except Exception as e:
            logger.warning(f"Sim eval error: {e}")
            sim = 0.5

        label, bkt_ok = self.bkt.eval_sim(sim)

        if label == 1:
            return {
                "correct": True, "label": 1,
                "score_delta": self.SCORE_CORRECT,
                "feedback": None,   # template engine sẽ điền
                "sim": sim
            }
        elif label == 2:
            return {
                "correct": True, "label": 2,
                "score_delta": self.SCORE_PARTIAL,
                "feedback": None,
                "sim": sim
            }
        else:
            hint = correct_answer.split()[0] if correct_answer else ""
            return {
                "correct": False, "label": 0,
                "score_delta": self.SCORE_WRONG,
                "feedback": None,
                "hint": hint, "sim": sim
            }

    def _llm_eval(self, child_answer: str,
                  question: str, correct_answer: str) -> Dict:
        """LLM-based evaluation cho câu trả lời tự do."""
        prompt = (
            "Bạn là bộ chấm câu trả lời trẻ em.\n"
            "CHỈ trả về đúng 1 dòng theo format: LABEL|FEEDBACK\n"
            "LABEL chỉ được là: DUNG, GAN_DUNG, SAI\n"
            "FEEDBACK tối đa 1 câu ngắn, thân thiện.\n\n"
            f"CAU_HOI: {question}\n"
            f"DAP_AN_THAM_KHAO: {correct_answer}\n"
            f"TRE_TRA_LOI: {child_answer}"
        )
        text = self.llm.generate(prompt, max_tokens=80, temperature=0.1)

        label_raw, feedback = self._parse_llm_eval(text)
        tu = label_raw.upper()

        if tu == "DUNG":
            label, correct, delta = 1, True, self.SCORE_CORRECT
        elif tu == "GAN_DUNG":
            label, correct, delta = 2, True, self.SCORE_PARTIAL
        else:
            label, correct, delta = 0, False, self.SCORE_WRONG

        return {
            "correct": correct, "label": label,
            "score_delta": delta, "feedback": feedback
        }

    @staticmethod
    def _parse_llm_eval(text: str) -> Tuple[str, str]:
        raw = " ".join((text or "").replace("\n", " ").split())
        raw = raw.replace("ĐÚNG", "DUNG").replace("GẦN ĐÚNG", "GAN_DUNG").replace("SAI", "SAI")

        m = re.search(r"\b(DUNG|GAN_DUNG|SAI)\b\s*\|\s*(.+)$", raw, flags=re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            feedback = m.group(2).strip()
        else:
            u = raw.upper()
            if "GAN_DUNG" in u:
                label = "GAN_DUNG"
            elif "DUNG" in u and "SAI" not in u:
                label = "DUNG"
            else:
                label = "SAI"
            feedback = raw

        # Strip leaked internal prompt chunks if model echoes instructions.
        for bad in ("CAU_HOI:", "DAP_AN_THAM_KHAO:", "TRE_TRA_LOI:", "LABEL|"):
            if bad in feedback.upper():
                feedback = feedback.split(bad, 1)[0].strip()

        if not feedback:
            feedback = "Bạn làm tốt lắm, mình cùng thử câu tiếp theo nhé!"

        return label, feedback[:180]
