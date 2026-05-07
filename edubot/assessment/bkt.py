"""
assessment/bkt.py — Bayesian Knowledge Tracing Engine
Corbett & Anderson (1994). Thuần Python, không cần GPU, < 1ms per update.
"""
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BKT_P_INIT, BKT_P_TRANSIT, BKT_P_GUESS, BKT_P_SLIP,
    BKT_MASTERY_THR, BKT_EASY_THR, BKT_HARD_THR
)
from loguru import logger


class BKT:
    """
    4 tham số:
      p_init     P(biết trước khi bắt đầu)
      p_transit  P(học được sau 1 attempt sai)
      p_guess    P(đoán đúng dù không biết)
      p_slip     P(trả lời sai dù biết)

    Cập nhật posterior:
      P(Ln | obs) = P(obs | Ln) * P(Ln) / P(obs)
      P(Ln+1)     = P(Ln_post) + (1-P(Ln_post)) * p_transit
    """

    def __init__(self,
                 p_init=BKT_P_INIT, p_transit=BKT_P_TRANSIT,
                 p_guess=BKT_P_GUESS, p_slip=BKT_P_SLIP,
                 mastery_thr=BKT_MASTERY_THR):
        self.p_init    = p_init
        self.p_transit = p_transit
        self.p_guess   = p_guess
        self.p_slip    = p_slip
        self.thr       = mastery_thr

    def update(self, p: float, correct: bool) -> float:
        if correct:
            p_obs  = (1 - self.p_slip) * p + self.p_guess * (1 - p)
            p_post = ((1 - self.p_slip) * p) / max(p_obs, 1e-9)
        else:
            p_obs  = self.p_slip * p + (1 - self.p_guess) * (1 - p)
            p_post = (self.p_slip * p) / max(p_obs, 1e-9)

        p_new = p_post + (1 - p_post) * self.p_transit
        return round(max(0.0, min(1.0, p_new)), 4)

    def mastered(self, p: float) -> bool:
        return p >= self.thr

    def difficulty(self, p: float) -> int:
        """Quyết định độ khó câu hỏi tiếp theo."""
        if p < BKT_EASY_THR:   return 2
        if p < BKT_HARD_THR:   return 3
        return 4

    def eval_sim(self, sim: float) -> Tuple[int, bool]:
        """
        Đánh giá câu trả lời dựa trên embedding similarity.
        Returns: (label: 1=đúng,2=gần đúng,0=sai), (bkt_correct: bool)
        """
        if sim >= 0.75:   return 1, True
        if sim >= 0.45:   return 2, True   # partial → BKT tính đúng
        return 0, False

    def simulate(self, p0: float, outcomes: List[bool]) -> List[float]:
        trace = [p0]
        for c in outcomes:
            p0 = self.update(p0, c)
            trace.append(p0)
        return trace


class MasteryManager:
    """
    Quản lý BKT state cho nhiều trẻ × nhiều concepts.
    Interface với DB qua MasteryDAO.
    RAM cache để tránh DB round-trip liên tục.
    """

    def __init__(self, mastery_dao, bkt: BKT = None):
        self.dao   = mastery_dao
        self.bkt   = bkt or BKT()
        self._cache: Dict[str, float] = {}   # "child:concept" → p

    def _key(self, child_id: str, concept_id: str) -> str:
        return f"{child_id}:{concept_id}"

    def get(self, child_id: str, concept_id: str) -> float:
        k = self._key(child_id, concept_id)
        if k in self._cache:
            return self._cache[k]
        rec = self.dao.get(child_id, concept_id)
        p = rec["p_mastery"] if rec else self.bkt.p_init
        self._cache[k] = p
        return p

    def record(self, child_id: str, concept_id: str,
               correct: bool) -> Dict:
        p_before = self.get(child_id, concept_id)
        p_after  = self.bkt.update(p_before, correct)
        self.dao.upsert(child_id, concept_id, p_after, correct)
        k = self._key(child_id, concept_id)
        self._cache[k] = p_after

        result = {
            "child_id":   child_id,
            "concept_id": concept_id,
            "p_before":   p_before,
            "p_after":    p_after,
            "correct":    correct,
            "mastered":   self.bkt.mastered(p_after),
            "just_mastered": (
                self.bkt.mastered(p_after) and not self.bkt.mastered(p_before)
            ),
            "next_diff":  self.bkt.difficulty(p_after)
        }
        logger.debug(
            f"BKT [{child_id}|{concept_id}]: "
            f"{p_before:.3f}→{p_after:.3f} ({'✓' if correct else '✗'})"
        )
        return result

    def next_difficulty(self, child_id: str, concept_id: str) -> int:
        return self.bkt.difficulty(self.get(child_id, concept_id))

    def summary(self, child_id: str) -> Dict:
        all_m = self.dao.all_for_child(child_id)
        if not all_m:
            return {"total": 0, "mastered": 0, "avg": 0.0}
        mastered = [m for m in all_m if m["p_mastery"] >= self.bkt.thr]
        return {
            "total":   len(all_m),
            "mastered": len(mastered),
            "avg":     round(sum(m["p_mastery"] for m in all_m) / len(all_m), 3),
            "weak":    self.dao.weak(child_id)[:3]
        }

    def invalidate(self, child_id: str = None):
        if child_id:
            to_del = [k for k in self._cache if k.startswith(child_id+":")]
            for k in to_del: del self._cache[k]
        else:
            self._cache.clear()
