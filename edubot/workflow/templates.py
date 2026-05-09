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

    # ── Greetings (after name capture) ─────────────────────────────────
    _GREET = [
        "Chào cậu! Mình là Robot thông minh đây. Vậy thì hôm nay chúng mình sẽ học về chủ đề gì nhỉ?",
        "Chào bạn nhỏ! Chúc cậu một ngày tốt lành! vậy thì hôm nay tớ với cậu sẽ học thêm về chủ đề nào thế nhỉ?",
        "Xin chào bạn yêu. Mình là Robot thông minh, rất vui khi được học tập với bạn. Vậy thì hôm nay tớ với học về chủ đề gì thế nhỉ?",
    ]

    # ── Name capture prompts (rotation) ──────────────────────────────────
    _NAME_CAPTURE = [
        "Hế lô bạn"
        "mình vừa khởi động toàn bộ hệ thống và sẵn sàng hỗ trợ rồi đây. "
        "Trước tiên. mình nên gọi bạn là gì cho ngầu nhỉ?",
        
        "Tinh tinh tinh !!! Rô bốt đã hoạt động trở lại rồi đây! "
        "Hôm nay chúng ta sẽ cùng nhau làm điều gì thú vị đây? "
        "À mà khoan. cho mình biết tên của bạn trước đã chứ",
        
        "Chào mừng bạn đã quay trở lại, "
        "mình là trợ lý thông minh, sẽ luôn có mặt để hỗ trợ bạn mọi lúc. "
        "Nhưng trước khi bắt đầu . mình có thể biết tên của bạn không?",
    ]

    # ── Teaching / Listening - 4 stages based on number of teaching turns ──
    _ACK_STAGE1 = [
        "Ồ? Đây là lần đầu mình nghe tới điều này đó! Còn cái gì thú vị hơn không? Hãy kể cho mình tiếp đi.",
        "Uầy, kiến thức mở màn thú vị hơn mình tưởng luôn đó. Tiếp tục dạy cho mình đi.",
        "Kiến thức mới này đã được gieo vào bộ não của mình thành công rồi, tiếp tục dạy mình đi cậu chủ ơi.",
        "Chờ chút. thông tin này làm mình tò mò thật đấy. Bạn còn biết thêm gì nữa không? Hãy kể cho mình thêm đi",
        "Mình vừa học được điều mới luôn đó. Kể tiếp đi, mình đang nghe rất chăm chú đây.",
    ]

    _ACK_STAGE2 = [
        "Khoan nha. mình bắt đầu thấy sự liên kết rồi đó. Tiếp tục dạy mình đi nào, nét gâu, nét gâu.",
        "Kiến thức này hay nha. Bạn giỏi thật đấy, hãy tiếp tục đi, mình thích nghe thêm nữa. nét gâu, nét gâu",
        "Mình cảm giác mọi thứ bắt đầu kết nối lại với nhau rồi đó. Còn phần tiếp theo thì sao? nét gâu, nét gâu",
        "Ơ kìa, càng nghe càng hợp lý luôn ấy. Bạn kể tiếp đi, mình bắt đầu hiểu rồi.",
    ]

    _ACK_STAGE3 = [
        "Càng nghe mình càng thấy chủ đề này thú vị cực kỳ. Vui quá, hãy dạy mình tiếp đi nào!",
        "Bạn siêu thật đấy, kiến thức của bạn thật đáng ngưỡng mộ, hãy dạy cho mình thêm điều nữa đi nào. Năn nỉ đó.",
        "Mình bắt đầu hiểu sâu hơn rồi nha. Nhưng chắc vẫn còn điều hay phía sau đúng không?, hãy tiết lộ cho mình đi mà",
        "Bạn kể chuyện có duyên ghê luôn ấy. Dạy tiếp đi, mình chưa muốn dừng đâu.",
    ]

    _ACK_STAGE4PLUS = [
        "Hôm nay bạn dạy cho mình nhiều kiến thức thật đấy. Giờ mình cảm giác bản thân hiểu hơn nhiều rồi, hay là chúng mình kiểm tra thử một chút không nhỉ?",
        "Kho dữ liệu của mình hôm nay được cập nhập mạnh thật sự luôn, hay là kiểm tra một chút nhỉ",
        "Hôm nay mình học được nhiều điều mới ghê. Thử làm một vài câu hỏi nhỏ xem có nhớ hết không đi.",
        "Càng nghe bạn dạy mình càng hiểu ra nhiều thứ hơn. Giờ tới lượt tớ hỏi bạn trả lời nhé",
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
    ]

    # ── Reward / End ──────────────────────────────────────────
    _REWARD = [
        "Buổi học hôm nay thật tuyệt! Bạn đã dạy mình {n} kiến thức và trả lời đúng {ok} trên tổng số {total}! Tổng: {score} điểm!",
        "Bạn học rất tốt! {ok} trên tổng số {total} câu đúng! Bạn đạt {score} điểm!",
        "Uao, {ok} trên tổng số {total} câu đúng hôm nay! Bạn thật giỏi! Tổng: {score} điểm!",
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
        """Random greeting after child name is captured."""
        return cls._p(cls._GREET)

    @classmethod
    def name_capture_prompt(cls) -> str:
        """Rotating prompt to ask for child's name (3 variants)."""
        return cls._p(cls._NAME_CAPTURE)

    @classmethod
    def ack(cls, stage: int = 1, concept: str = "") -> str:
        """Get acknowledgment response based on teaching stage (1-4+).
        stage 1: first teaching turn
        stage 2: second teaching turn
        stage 3: third teaching turn
        stage 4+: fourth or more teaching turns
        """
        if stage <= 1:
            ack_list = cls._ACK_STAGE1
        elif stage == 2:
            ack_list = cls._ACK_STAGE2
        elif stage == 3:
            ack_list = cls._ACK_STAGE3
        else:  # stage >= 4
            ack_list = cls._ACK_STAGE4PLUS
        return cls._p(ack_list)

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
