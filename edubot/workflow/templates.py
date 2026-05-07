"""
workflow/templates.py — Procedural Memory: Response Templates
Slot-fill, không cần LLM cho 80% interaction.
"""
import random
from typing import Optional


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS (per phase)
# ══════════════════════════════════════════════════════════════
SYSTEM = {
    "teach": (
        "Bạn là robot đang học bài từ một bé nhỏ. "
        "Bạn thông minh nhưng vờ chưa biết những gì bé dạy. "
        "Phản hồi tích cực, khuyến khích bé kể thêm. Tối đa 1 câu."
    ),
    "confuse": (
        "Bạn là robot học trò. Bạn vừa nghe nhưng còn chỗ chưa hiểu. "
        "Hỏi lại đúng 1 câu cụ thể. Tự nhiên, tò mò. Tối đa 1 câu."
    ),
    "quiz": (
        "Bạn là robot kiểm tra kiến thức của bé. "
        "Đặt câu hỏi rõ ràng, phù hợp lứa tuổi. Chỉ hỏi 1 câu."
    ),
    "eval": (
        "Đánh giá câu trả lời của bé. Chỉ dùng thông tin được cung cấp. "
        "Nếu đúng: khen ngắn. Nếu sai: Hãy trả lời câu hỏi đó một cách chính xác, sau đó hỏi thêm câu hỏi liên quan để gợi ý bé học lại. Tối đa 1 câu."
    ),
}


class T:
    """Template Engine — tất cả method là classmethod."""

    _cycle_state = {}

    # ── Greetings ─────────────────────────────────────────────
    _GREET = [
        "Chào cậu! Mình là Robot thông minh đây. Hôm nay chúng mình sẽ học về chủ đề gì nhỉ?",
        "Chào bạn nhỏ! Chúc cậu một ngày tốt lành! Hôm nay tớ với cậu sẽ học thêm về chủ đề nào thế nhỉ?",
        "Xin chào bạn yêu. Mình là Robot thông minh, rất vui khi được học tập với bạn. Hôm nay tớ với học về chủ đề gì thế nhỉ?",
    ]

    # ── Teaching / Listening ──────────────────────────────────
    _ACK = [
        "Kiến thức này thật thú vị, cảm ơn bạn đã chia sẻ cho mình biết thêm nhé",
        "Ồ, thế còn gì nữa không nhỉ? Bạn hãy kể tiếp đi!",
        "Uao, mình học được điều mới rồi! Còn gì nữa không nhỉ? Bạn dạy cho mình thêm đi",
        "Bạn đã nói cho mình biết thêm kiến thức thú vị! Hãy tiếp tục thôi nào!",
    ]

    # ── Confusion questions ────────────────────────────────────
    _CONFUSE = [
        "Bạn ơi, vậy {sub} là gì vậy? Mình chưa hiểu!",
        "Thế {sub} với {concept} khác nhau ở đâu vậy?",
        "Ồ bạn nói {concept} nhỉ? Tại sao {concept} lại {prop} vậy?",
        "Mình thắc mắc: {concept} và {sub} có liên quan không?",
    ]

    # ── Quiz ──────────────────────────────────────────────────
    _QUIZ_INTRO = [
        "Bây giờ mình hỏi bạn một câu để xem bạn đã nhớ chưa nhé!",
        "Đến phần kiểm tra rồi! Mình sẽ đố bạn câu này nhé:",
        "Thử thách nhỏ cho bạn này:",
        "Đến phần giải câu đố rồi! Bạn cho mình hỏi này:",
    ]

    _CORRECT = [
        "Đúng rồi! Bạn giỏi lắm! +{pts} điểm!",
        "Chính xác! Bạn nhớ rất tốt! +{pts} điểm!",
        "Tuyệt vời! Bạn hiểu rõ lắm! +{pts} điểm!",
        "Uao, đúng luôn! +{pts} điểm cho bạn!",
    ]

    _PARTIAL = [
        "Gần đúng rồi! Bạn đã hiểu phần lớn! +{pts} điểm!",
        "Ý đúng đó, nhưng có thể đầy đủ hơn! +{pts} điểm!",
    ]

    _WRONG = [
        "Chưa đúng rồi, nhưng không sao! Thử lại nhé!",
        "Câu này hơi khó! Bạn nghĩ lại xem?",
        "Hmm, chưa phải! Gợi ý: nghĩ về {hint}...",
    ]

    # ── Reward / End ──────────────────────────────────────────
    _REWARD = [
        "Buổi học hôm nay thật tuyệt! Bạn đã dạy mình {n} kiến thức và trả lời đúng {ok}/{total}! Tổng: {score} điểm!",
        "Bạn học rất tốt! {ok}/{total} câu đúng! Bạn đạt {score} điểm!",
        "Uao, {ok}/{total} câu đúng hôm nay! Bạn thật giỏi! Tổng: {score} điểm!",
    ]

    _LEVEL_UP = [
        "Chúc mừng! Bạn đã giỏi hơn rồi đấy! Câu hỏi tiếp theo cho bạn này:",
        "Tuyệt vời! Bạn thông minh quá! Sang câu hỏi tiếp theo nào!",
    ]

    _UNKNOWN = [
        "Bạn vừa nói gì vậy? Mình chưa nghe rõ. Nói lại được không?",
        "Hmm, mình chưa hiểu. Bạn nói chậm hơn được không?",
    ]

    @classmethod
    def _p(cls, lst):
        # Avoid repeating the same response until one full cycle is consumed.
        key = id(lst)
        state = cls._cycle_state.get(key)
        if not state or state.get("idx", 0) >= len(state.get("order", [])):
            order = list(range(len(lst)))
            random.shuffle(order)
            state = {"order": order, "idx": 0}
            cls._cycle_state[key] = state

        choice_idx = state["order"][state["idx"]]
        state["idx"] += 1
        return lst[choice_idx]

    # ── Public API ────────────────────────────────────────────
    @classmethod
    def greeting(cls) -> str:
        return cls._p(cls._GREET)

    @classmethod
    def ack(cls, concept: str = "") -> str:
        return cls._p(cls._ACK).format(concept=concept or "điều bạn vừa nói")

    @classmethod
    def confuse(cls, concept: str, sub: str = "", prop: str = "") -> str:
        return cls._p(cls._CONFUSE).format(
            concept=concept, sub=sub or "thứ đó", prop=prop or "như vậy"
        )

    @classmethod
    def quiz_intro(cls) -> str:
        return cls._p(cls._QUIZ_INTRO)

    @classmethod
    def correct(cls, pts: int = 10) -> str:
        return cls._p(cls._CORRECT).format(pts=pts)

    @classmethod
    def partial(cls, pts: int = 5) -> str:
        return cls._p(cls._PARTIAL).format(pts=pts)

    @classmethod
    def wrong(cls, hint: str = "") -> str:
        return cls._p(cls._WRONG).format(hint=hint or "bài học")

    @classmethod
    def reward(cls, n: int, ok: int, total: int, score: int) -> str:
        return cls._p(cls._REWARD).format(n=n, ok=ok, total=total, score=score)

    @classmethod
    def level_up(cls, concept: str) -> str:
        return cls._p(cls._LEVEL_UP).format(concept=concept)

    @classmethod
    def unknown(cls) -> str:
        return cls._p(cls._UNKNOWN)

    @classmethod
    def system(cls, phase: str) -> str:
        return SYSTEM.get(phase, SYSTEM["teach"])

    @classmethod
    def feedback(cls, label: int, pts: int, hint: str = "") -> str:
        """Convert eval label → feedback text."""
        if label == 1:  return cls.correct(pts)
        if label == 2:  return cls.partial(pts)
        return cls.wrong(hint)
