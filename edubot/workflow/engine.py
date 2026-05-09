"""
workflow/engine.py — Main engine for managing session state, processing ASR text, routing, and generating responses.
"""
import random
import re
import uuid
import json
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Optional, List
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEACH_TURNS_BEFORE_CONFUSE, CONFUSE_TURNS,
    QUIZ_TOTAL, QUIZ_BANK_RATIO, LOG_DIR, DATA_DIR
)
from workflow.router import Router, Path as RPath
from workflow.templates import T
from core.intent import IntentClassifier


class Phase(Enum):
    IDLE    = "idle"
    TEACH   = "teach"
    CONFUSE = "confuse"
    QUIZ    = "quiz"
    REWARD  = "reward"


class Engine:
    """
    Nhận text từ ASR → route → response text → TTS.
    Hardware events được emit qua hw_cb(event, data).
    """

    def __init__(self, tools, llm, cloud_llm, evaluator,
                 mastery_mgr, session_dao,
                 hw_cb: Callable = None):
        self.tools    = tools
        self.llm      = llm
        self.cloud    = cloud_llm
        self.evaluator = evaluator
        self.mastery  = mastery_mgr
        self.ses_dao  = session_dao
        self.hw_cb    = hw_cb or (lambda e, d: None)
        self.clf      = IntentClassifier()
        self.router   = Router(tools)

        # Session state
        self._phase       = Phase.IDLE
        self._sid:   str  = None
        self._child: str  = None
        self._doc:   str  = None
        self._memory      = None
        self._concept_id: Optional[str] = None
        self._topic_label: str = ""
        self._topic_domain: str = ""
        self._topic_doc_prefix: str = ""

        # Counters
        self._teach_turns  = 0
        self._confuse_turns = 0
        self._ask_turns    = 0
        self._quiz_offer_pending = False
        self._quiz_idx     = 0
        self._score        = 0
        self._correct      = 0
        self._current_qa   = None
        self._quiz_wait_understand = False
        self._quiz_explain_level = 0
        self._asked_quiz_questions = set()
        self._asked_qa_ids = set()
        self._quiz_concept_queue = []
        self._quiz_focus_concept: Optional[str] = None
        self._concept_quiz_correct_counts: Dict[str, int] = {}
        self._doc_title_cache: Dict[str, str] = {}
        self._stories_cache: Optional[List[Dict[str, str]]] = None
        self._fun_facts_cache: Optional[List[str]] = None
        self._last_story_id: str = ""
        self._last_fun_fact: str = ""
        self._last_response_kind: str = "normal"
        self._post_session_offer_pending: bool = False
        self._completed_cycles: int = 0
        self._story_done: bool = False
        self._terminate_requested: bool = False
        self._mimic_active: bool = False
        self._last_story_audio_path: str = ""
        self._last_story_title: str = ""

    # ══════════════════════════════════════════════════════════
    # SESSION CONTROL
    # ══════════════════════════════════════════════════════════
    def start_session(self, child_id: str, memory,
                      doc_id: str = None,
                      session_id: str = None) -> str:
        # Allow caller to pre-create session to avoid duplicate DB rows.
        self._sid    = session_id or self.ses_dao.create(child_id, doc_id)
        self._child  = child_id
        self._doc    = doc_id
        self._memory = memory
        self._phase  = Phase.TEACH
        self._teach_turns = self._confuse_turns = 0
        self._ask_turns = 0
        self._quiz_offer_pending = False
        self._quiz_idx = self._score = self._correct = 0
        self._current_qa = None
        self._concept_id = None
        self._topic_label = ""
        self._topic_domain = ""
        self._topic_doc_prefix = ""
        self._quiz_wait_understand = False
        self._quiz_explain_level = 0
        self._asked_quiz_questions = set()
        self._asked_qa_ids = set()
        self._quiz_concept_queue = []
        self._quiz_focus_concept = None
        self._concept_quiz_correct_counts = {}
        self._doc_title_cache = {}
        self._post_session_offer_pending = False
        self._completed_cycles = 0
        self._story_done = False
        self._terminate_requested = False
        self._mimic_active = False
        self._last_story_audio_path = ""
        self._last_story_title = ""

        self.ses_dao.set_phase(self._sid, "teach")
        self.hw_cb("phase", {"phase": "teach"})
        logger.info(f"Session {self._sid} started — child: {child_id}")
        self._append_history_log("system", "SESSION_STARTED")
        return self._sid

    def end_session(self) -> str:
        self._append_history_log(
            "system",
            f"SESSION_ENDED score={self._score} correct={self._correct}/{self._quiz_idx}"
        )
        self.ses_dao.end(self._sid)
        self._phase = Phase.IDLE
        logger.info(f"Session {self._sid} ended — score: {self._score}")
        return f"score={self._score}, correct={self._correct}/{self._quiz_idx}"

    # ══════════════════════════════════════════════════════════
    # MAIN PROCESS
    # ══════════════════════════════════════════════════════════
    def process(self, text: str) -> str:
        """ASR text → response text. Entry point."""
        text = text.strip()
        if not text:
            return ""
        self._last_story_audio_path = ""
        self._last_story_title = ""

        logger.info(f"[{self._phase.value}] ← {text[:80]}")

        intent, _ = self.clf.classify(text, self._phase.value)
        logger.info(f"[intent] phase={self._phase.value} intent={intent.name}")
        if self._memory:
            self._memory.add_turn("user", text)
        self._append_history_log("user", text)

        if self._is_session_end_request(text):
            if self._can_end_session_now():
                self._last_response_kind = "normal"
                bye = "Tạm biệt bạn nhé. Hẹn gặp lại ở buổi học sau!"
                if self._memory and bye:
                    self._memory.add_turn("assistant", bye)
                self._append_history_log("assistant", bye)
                self._terminate_requested = True
                self.end_session()
                logger.info(f"[{self._phase.value}] → {bye[:120]}")
                return bye
            remind = (
                "Mình và bạn chưa hoàn thành một vòng học hoặc kể chuyện xong. "
                "Mình học thêm chút nữa rồi hẵng tạm biệt nhé."
            )
            if self._memory:
                self._memory.add_turn("assistant", remind)
            self._append_history_log("assistant", remind)
            logger.info(f"[{self._phase.value}] → {remind[:120]}")
            return remind

        if self._is_mimic_stop_request(text):
            self._mimic_active = False
            self._last_response_kind = "normal"
            msg = "Ô kê, mình dừng bắt chước nhé. Bạn muốn học tiếp hay nghe kể chuyện?"
            if self._memory:
                self._memory.add_turn("assistant", msg)
            self._append_history_log("assistant", msg)
            logger.info(f"[{self._phase.value}] → {msg[:120]}")
            return msg

        gesture = self._detect_gesture_action(text)
        if gesture:
            self._last_response_kind = "normal"
            self.hw_cb("gesture", {"action": gesture})
            if gesture == "handshake":
                msg = "Bạn hãy giơ tay ra để bắt tay với tôi nào."
            else:
                msg = "Bạn xem tay tôi di chuyển này."
            if self._memory:
                self._memory.add_turn("assistant", msg)
            self._append_history_log("assistant", msg)
            logger.info(f"[{self._phase.value}] → {msg[:120]}")
            return msg

        if self._mimic_active:
            if self._is_story_request(text):
                self._mimic_active = False
            elif self._wants_new_topic(text) or self._is_topic_start_request(text):
                self._mimic_active = False
            else:
                self._last_response_kind = "mimic"
                resp = text.strip()
                if self._memory and resp:
                    self._memory.add_turn("assistant", resp)
                if resp:
                    self._append_history_log("assistant", resp)
                logger.info(f"[{self._phase.value}] → {resp[:120]}")
                return resp

        if self._is_mimic_request(text):
            self._mimic_active = True
            self._last_response_kind = "normal"
            resp = "Được rồi, bắt chước mode đã bật. Bạn nói gì mình nhại y chang nhé!"
            if self._memory and resp:
                self._memory.add_turn("assistant", resp)
            if resp:
                self._append_history_log("assistant", resp)
            logger.info(f"[{self._phase.value}] → {resp[:120]}")
            return resp

        if self._is_story_request(text):
            if self._is_lesson_in_progress():
                self._last_response_kind = "normal"
                resp = (
                    "Mình rất muốn kể chuyện cho bạn nghe lắm. "
                    "Mình và bạn học xong phần này đã nhé, rồi mình kể truyện thật hay cho bạn."
                )
            else:
                self._last_response_kind = "story"
                resp = self._tell_story()
                self._story_done = True
        elif self._is_fun_fact_request(text):
            self._last_response_kind = "fun_fact"
            resp = self._share_fun_fact()
        elif self._phase == Phase.IDLE:
            self._last_response_kind = "normal"
            if self._post_session_offer_pending:
                if self._wants_new_topic(text) or self._is_topic_start_request(text):
                    self._prepare_new_learning_round()
                    resp = self._teach(text, intent)
                else:
                    resp = (
                        "Bạn muốn học tiếp chủ đề mới, nghe kể chuyện hay chuyển sang bắt chước? "
                        "Nếu muốn kết thúc thì nói: chào tạm biệt."
                    )
            else:
                resp = T.greeting()
                self._phase = Phase.TEACH
        elif self._phase == Phase.TEACH:
            self._last_response_kind = "normal"
            resp = self._teach(text, intent)
        elif self._phase == Phase.CONFUSE:
            self._last_response_kind = "normal"
            resp = self._confuse(text, intent)
        elif self._phase == Phase.QUIZ:
            self._last_response_kind = "normal"
            resp = self._quiz(text, intent)
        elif self._phase == Phase.REWARD:
            self._last_response_kind = "normal"
            resp = self._reward()
        else:
            self._last_response_kind = "normal"
            resp = T.unknown()

        if self._memory and resp:
            self._memory.add_turn("assistant", resp)
        if resp:
            self._append_history_log("assistant", resp)

        logger.info(f"[{self._phase.value}] → {resp[:120]}")
        return resp

    def _tell_story(self) -> str:
        story = self._select_story()
        if not story:
            return (
                "Mình chưa có truyện để kể lúc này. "
                "Bạn giúp mình thêm truyện vào fairy_tales.json nhé!"
            )

        title = story.get("title", "một câu chuyện")
        content = story.get("text", "").strip()
        if not content:
            return "Mình chưa đọc được nội dung truyện này, bạn thử truyện khác nhé."

        story_id = str(story.get("id") or "").strip()
        self._last_story_title = title
        self._last_story_audio_path = str(
            Path(DATA_DIR)
            / "documents"
            / "story_telling"
            / "audio"
            / self._story_audio_filename(story_id, title)
        )

        wav_exists = Path(self._last_story_audio_path).exists()
        if wav_exists:
            return f"Tôi sẽ kể cho bạn câu chuyện {title}."
        return f"Tôi sẽ kể cho bạn câu chuyện {title}. {content}"

    def _share_fun_fact(self) -> str:
        fact = self._select_fun_fact()
        if not fact:
            return (
                "Hiện tại mình chưa có fun fact nào sẵn sàng. "
                "Bạn có thể thêm vào fun_facts.json rồi mình kể tiếp nhé!"
            )
        return f"Fun fact đây: {fact}"

    def _select_story(self) -> Optional[Dict[str, str]]:
        stories = self._load_stories()
        if not stories:
            return None

        if len(stories) == 1:
            picked = stories[0]
        else:
            candidates = [s for s in stories if s.get("id", "") != self._last_story_id]
            if not candidates:
                candidates = stories
            picked = random.choice(candidates)

        self._last_story_id = picked.get("id", "")
        return picked

    def _select_fun_fact(self) -> str:
        facts = self._load_fun_facts()
        if not facts:
            return ""

        if len(facts) == 1:
            picked = facts[0]
        else:
            candidates = [f for f in facts if f != self._last_fun_fact]
            if not candidates:
                candidates = facts
            picked = random.choice(candidates)

        self._last_fun_fact = picked
        return picked

    def _load_stories(self) -> List[Dict[str, str]]:
        if self._stories_cache is not None:
            return self._stories_cache

        path = Path(DATA_DIR) / "documents" / "story_telling" / "fairy_tales.json"
        stories: List[Dict[str, str]] = []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load stories from {path}: {e}")
            self._stories_cache = []
            return self._stories_cache

        if isinstance(raw, list):
            for idx, item in enumerate(raw, start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or f"Truyện {idx}").strip()
                sid = str(item.get("id") or idx).strip()
                paragraphs = item.get("paragraphs")
                if isinstance(paragraphs, list):
                    parts = [str(p).strip() for p in paragraphs if str(p).strip()]
                else:
                    content = item.get("content") or item.get("text") or ""
                    parts = [str(content).strip()] if str(content).strip() else []
                text = " ".join(parts).strip()
                if text:
                    stories.append({"id": sid, "title": title, "text": text})
        elif isinstance(raw, dict):
            records = raw.get("records")
            if isinstance(records, list):
                for idx, item in enumerate(records, start=1):
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title") or f"Truyện {idx}").strip()
                    sid = str(item.get("id") or idx).strip()
                    content = item.get("content") or item.get("text") or ""
                    text = str(content).strip()
                    if text:
                        stories.append({"id": sid, "title": title, "text": text})

        self._stories_cache = stories
        logger.info(f"Story library loaded: {len(stories)} stories")
        return stories

    def _load_fun_facts(self) -> List[str]:
        if self._fun_facts_cache is not None:
            return self._fun_facts_cache

        path = Path(DATA_DIR) / "documents" / "story_telling" / "fun_facts.json"
        facts: List[str] = []

        try:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                self._fun_facts_cache = []
                return self._fun_facts_cache
            raw = json.loads(raw_text)
        except Exception as e:
            logger.warning(f"Could not load fun facts from {path}: {e}")
            self._fun_facts_cache = []
            return self._fun_facts_cache

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    facts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("fact") or ""
                    text = str(text).strip()
                    if text:
                        facts.append(text)
        elif isinstance(raw, dict):
            if isinstance(raw.get("facts"), list):
                for item in raw["facts"]:
                    text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else ""
                    text = str(text).strip()
                    if text:
                        facts.append(text)

        self._fun_facts_cache = facts
        logger.info(f"Fun-fact library loaded: {len(facts)} items")
        return facts

    @staticmethod
    def _is_story_request(text: str) -> bool:
        t = (text or "").lower()
        triggers = [
            "kể chuyện", "ke chuyen", "kể truyện", "ke truyen",
            "đọc truyện", "doc truyen", "đọc chuyện", "doc chuyen",
            "nghe truyện", "nghe chuyện", "câu chuyện", "cau chuyen", "story",
        ]
        return any(k in t for k in triggers)

    @staticmethod
    def _is_fun_fact_request(text: str) -> bool:
        t = (text or "").lower()
        triggers = [
            "fun fact", "funfact", "sự thật thú vị", "su that thu vi",
            "fact thú vị", "fact thu vi",
        ]
        return any(k in t for k in triggers)

    @staticmethod
    def _is_mimic_request(text: str) -> bool:
        t = (text or "").lower()
        triggers = [
            "bắt chước", "bat chuoc", "bắt trước", "bat truoc",
            "nhại lại", "nhai lai", "lặp lại", "lap lai",
            "nhắc lại", "nhac lai", "repeat after me", "bắt chước đi",
        ]
        return any(k in t for k in triggers)

    @staticmethod
    def _is_mimic_stop_request(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "dừng bắt chước", "dung bat chuoc", "thoát bắt chước", "thoat bat chuoc",
            "đổi chế độ", "doi che do", "học tiếp", "hoc tiep",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_session_end_request(text: str) -> bool:
        t = (text or "").lower().strip()
        keys = [
            "chào tạm biệt", "tam biet", "tạm biệt", "bye", "bai bai", "bai",
            "kết thúc", "ket thuc", "chào bạn", "chao ban",
        ]
        return any(k in t for k in keys)

    def _can_end_session_now(self) -> bool:
        return self._completed_cycles >= 1 or self._story_done

    def _is_lesson_in_progress(self) -> bool:
        if self._phase in (Phase.CONFUSE, Phase.QUIZ, Phase.REWARD):
            return True
        if self._phase == Phase.TEACH:
            return bool(self._teach_turns > 0 or self._ask_turns > 0 or self._concept_id)
        return False

    def _prepare_new_learning_round(self):
        self._post_session_offer_pending = False
        self._phase = Phase.TEACH
        self._teach_turns = 0
        self._ask_turns = 0
        self._quiz_offer_pending = False
        self._quiz_idx = 0
        self._correct = 0
        self._score = 0
        self._current_qa = None
        self._quiz_wait_understand = False
        self._quiz_explain_level = 0
        self._asked_quiz_questions = set()
        self._asked_qa_ids = set()
        self._quiz_concept_queue = []
        self._quiz_focus_concept = None
        self._concept_quiz_correct_counts = {}
        self._concept_id = None
        self._topic_label = ""
        self._topic_domain = ""
        self._topic_doc_prefix = ""
        self.ses_dao.set_phase(self._sid, "teach")

    @staticmethod
    def _wants_new_topic(text: str) -> bool:
        t = (text or "").lower().strip()
        keys = [
            "học tiếp", "hoc tiep", "chủ đề mới", "chu de moi",
            "đổi chủ đề", "doi chu de", "học chủ đề", "hoc chu de",
            "bài mới", "bai moi",
        ]
        return any(k in t for k in keys)

    # ══════════════════════════════════════════════════════════
    # PHASE HANDLERS
    # ══════════════════════════════════════════════════════════
    def _teach(self, text: str, intent) -> str:
        from core.intent import Intent

        if self._is_topic_switch_request(text):
            self._concept_id = None
            self._topic_label = ""
            self._topic_domain = ""
            self._topic_doc_prefix = ""
            return "Được nhé, mình sẽ đổi chủ đề. Bạn muốn học chủ đề nào tiếp theo?"

        if self._quiz_offer_pending:
            if self._is_quiz_opt_in(text):
                self._quiz_offer_pending = False
                return self._goto_quiz()
            if self._is_teach_continue_request(text):
                self._quiz_offer_pending = False
                # Give learner a few more teaching turns before asking quiz again.
                self._teach_turns = max(0, TEACH_TURNS_BEFORE_CONFUSE - 3)
                return "Tuyệt vời, bạn dạy tiếp nhé. Mình đang nghe đây!"
            if self._is_uncertain(text):
                self._quiz_offer_pending = False
                return "Không sao cả. Bạn dạy mình thêm một ý ngắn nữa nhé, rồi mình hỏi lại thật dễ."
            self._quiz_offer_pending = False

        if self._is_teach_stop_signal(text):
            if self._concept_id and self._teach_turns >= max(1, TEACH_TURNS_BEFORE_CONFUSE - 1):
                self._quiz_offer_pending = True
                return (
                    "Ô kê, mình hiểu rồi. Bạn muốn mình hỏi 1 câu kiểm tra ngắn không? "
                    "Nếu muốn, nói 'ô kê' hoặc 'kiểm tra'."
                )
            return "Ô kê, cảm ơn bạn. Nếu muốn, bạn có thể dạy thêm một ý ngắn nữa nhé."

        if intent == Intent.GREETING:
            route = {"path": RPath.A, "reason": "greeting", "kwargs": {}}
            self._log_route(route, from_phase=Phase.TEACH.value)
            return self._execute_route(route, text)

        # Ưu tiên bắt chủ đề trực tiếp từ câu mở đầu để tránh map lệch concept.
        if self._is_topic_start_request(text):
            topic = self._extract_requested_topic(text)
            if topic:
                self._topic_label = topic
                self._topic_domain = self._infer_domain_from_text(topic)
            if topic:
                resolved = self._resolve_topic_concept(topic)
                if resolved:
                    self._concept_id = resolved
                    self._topic_domain = self._concept_domain(self._concept_id) or self._topic_domain
                    self._topic_doc_prefix = self._concept_doc_prefix(self._concept_id)
            concept = topic or self._concept_id_name() or "chủ đề này"
            return (
                f"Chủ đề {concept} thật thú vị! Mình rất háo hức được biết thêm các kiến thức liên quan đến {concept}. "
                "Bạn có thể dạy cho mình một vài kiến thức về chủ đề này được không? Mình sẵn sàng rồi!"
            )

        # Tìm concept liên quan
        vec = self.tools.search_concepts(text, n=1)
        if vec.ok and vec.data:
            tops = vec.data.get("results", [])
            if tops and tops[0].get("concept_id"):
                candidate = tops[0]["concept_id"]
                if self._can_adopt_candidate(candidate):
                    self._concept_id = candidate
                    self._topic_domain = self._concept_domain(candidate) or self._topic_domain
                sim = tops[0].get("score", 0.0)
            else:
                sim = 0.0
        else:
            sim = 0.0

        # Route câu hỏi của trẻ → trả lời từ KB
        if intent == Intent.ASKING:
            self._ask_turns += 1
            route = self.router._route_question(text, self._concept_id)
            self._log_route(route, from_phase=Phase.TEACH.value)
            ans = self._execute_route(route, text)
            # Keep dialog natural: occasionally invite child to continue teaching.
            if self._concept_id and (self._ask_turns % 2 == 0):
                return ans + " Bạn muốn hỏi tiếp hay dạy mình thêm 1 ý mới?"
            return ans

        if self._memory:
            self._memory.record_teaching(text, self._concept_id, sim)

        self._teach_turns += 1

        # Transition check
        if self._teach_turns >= TEACH_TURNS_BEFORE_CONFUSE:
            if not self._concept_id:
                # Keep conversation in teach mode until we have a clear concept.
                self._teach_turns = TEACH_TURNS_BEFORE_CONFUSE - 1
                return (
                    "Mình chưa rõ chủ đề chính bạn đang dạy. "
                    "Bạn nói rõ một câu kiểu 'Hệ Mặt Trời là...' hoặc 'Động vật là...' nhé!"
                )
            # Do not force confuse phase. Offer a quiz and let learner choose.
            self._teach_turns = TEACH_TURNS_BEFORE_CONFUSE - 1
            self._quiz_offer_pending = True
            return (
                "Thật là tuyệt vời! Mình nghĩ bạn dạy rất tốt rồi. Bạn muốn mình hỏi 1 câu kiểm tra ngắn không? "
                "Nếu muốn, nói 'ô kê, hoặc kiểm tra'. Nếu chưa, bạn dạy thêm cho mình."
            )

        # Đa dạng response: ack hoặc deepening
        stage = self._get_teaching_stage()
        if self._teach_turns % 3 == 0 and self._concept_id:
            g = self.tools.traverse_graph(concept_id=self._concept_id)
            if g.ok and g.data:
                node = g.data.get("node", {})
                name = node.get("name", "") if node else ""
                if name:
                    return T.ack(stage=stage, concept=name)
        route = {
            "path": RPath.A,
            "reason": "teach_ack",
            "kwargs": {"concept_id": self._concept_id},
        }
        self._log_route(route, from_phase=Phase.TEACH.value)
        return T.ack(stage=stage, concept=self._concept_id_name())

    def _start_confuse(self) -> str:
        self._phase = Phase.CONFUSE
        self._confuse_turns = 0
        self.ses_dao.set_phase(self._sid, "confuse")
        self.hw_cb("phase", {"phase": "confuse"})

        if not self._concept_id:
            return self._goto_quiz()

        g = self.tools.traverse_graph(concept_id=self._concept_id)
        if g.ok and g.data:
            node = g.data.get("node", {})
            nbs  = g.data.get("neighbors", [])
            name = node.get("name", "chủ đề này") if node else "chủ đề này"
            if nbs:
                sub = random.choice(nbs[:3])
                return T.confuse(concept=name, sub=sub.get("name", ""))

        return T.confuse(concept=self._concept_id_name())

    def _confuse(self, text: str, intent) -> str:
        from core.intent import Intent

        if intent == Intent.GREETING:
            route = {"path": RPath.A, "reason": "greeting", "kwargs": {}}
            self._log_route(route, from_phase=Phase.CONFUSE.value)
            return self._execute_route(route, text)

        if self._memory:
            self._memory.record_teaching(text, self._concept_id, 0.4)

        self._confuse_turns += 1

        if intent == Intent.ASKING:
            route = self.router._route_question(text, self._concept_id)
            self._log_route(route, from_phase=Phase.CONFUSE.value)
            return self._execute_route(route, text)

        ack = random.choice(["Ồ mình hiểu hơn rồi! ", "À, thì ra là vậy! ", "Mình cảm ơn! "])

        if self._confuse_turns >= CONFUSE_TURNS:
            # More flexible: if learner is still teaching strongly, stay a bit longer.
            if (
                CONFUSE_TURNS >= 3
                and intent == Intent.TEACHING
                and self._confuse_turns == CONFUSE_TURNS
            ):
                self._confuse_turns -= 1
                return ack + "Bạn giải thích tốt quá, kể thêm chút nữa rồi mình kiểm tra nhé!"
            return ack + self._goto_quiz()

        # Hỏi thêm sub-concept
        if self._concept_id:
            g = self.tools.traverse_graph(concept_id=self._concept_id)
            if g.ok and g.data:
                nbs = g.data.get("neighbors", [])
                if len(nbs) > 1:
                    sub = nbs[1]
                    return ack + T.confuse(
                        concept=self._concept_id_name(),
                        sub=sub.get("name", "")
                    )
        return ack + "Bạn biết thêm gì về chủ đề này không?"

    def _goto_quiz(self) -> str:
        self._phase = Phase.QUIZ
        self._quiz_idx = 0
        self._quiz_focus_concept = self._concept_id
        self._quiz_concept_queue = self._build_quiz_concept_queue()
        self.ses_dao.set_phase(self._sid, "quiz")
        self.hw_cb("phase", {"phase": "quiz"})
        intro = T.quiz_intro()
        first = self._next_question()
        return f"{intro} {first}"

    def _quiz(self, text: str, intent) -> str:
        from core.intent import Intent
        if not self._current_qa:
            return self._next_question()

        if self._quiz_wait_understand:
            return self._handle_understanding_check(text)

        if self._is_request_next_question(text):
            self._finalize_current_question(correct=False, delta=0, child_answer=text)
            return "Được nhé, giờ tớ hỏi bạn câu khác nè! " + self._next_question()

        if self._is_request_reask_question(text):
            q = self._current_qa.get("question", "")
            if q:
                return f"Mình hỏi lại nhé: {q}"
            return self._next_question()

        if intent == Intent.GREETING:
            route = {"path": RPath.A, "reason": "greeting", "kwargs": {}}
            self._log_route(route, from_phase=Phase.QUIZ.value)
            return self._execute_route(route, text)

        if intent == Intent.ASKING:
            route = self.router._route_question(text, self._concept_id)
            self._log_route(route, from_phase=Phase.QUIZ.value)
            return self._execute_route(route, text)

        # Evaluate answer
        qa    = self._current_qa
        if self._is_uncertain(text):
            return self._explain_answer_and_check_understanding(qa)

        if not (qa.get("answer") or "").strip():
            eval_r = {
                "correct": False,
                "label": 2,
                "score_delta": 0,
                "feedback": "Cảm ơn bạn đã trả lời! Mình sẽ hỏi câu khác rõ hơn nhé.",
            }
        else:
            eval_r = self.evaluator.evaluate(
                child_answer   = text,
                question       = qa.get("question", ""),
                correct_answer = qa.get("answer", "")
            )

        label      = eval_r["label"]
        delta      = eval_r["score_delta"]
        bkt_correct = eval_r["correct"]
        fb_text    = eval_r.get("feedback")

        qa_concept_id = qa.get("concept_id") or self._concept_id

        # BKT update
        newly_mastered = False
        if qa_concept_id:
            bkt_r = self.tools.get_mastery(
                self._child, qa_concept_id, update=bkt_correct
            )
            if bkt_r.ok and bkt_r.data:
                newly_mastered = bkt_r.data.get("just_mastered", False)

        # Score update
        self._score   += delta
        if bkt_correct: self._correct += 1
        if qa_concept_id and bkt_correct:
            self._concept_quiz_correct_counts[qa_concept_id] = (
                self._concept_quiz_correct_counts.get(qa_concept_id, 0) + 1
            )
        self.ses_dao.add_score(self._sid, delta, bkt_correct)
        self.ses_dao.log_quiz(
            self._sid, self._child, qa_concept_id or "",
            qa.get("question",""), text,
            1 if bkt_correct else 0, delta, qa.get("id")
        )

        self.hw_cb("quiz_result",
                   {"correct": bkt_correct, "delta": delta, "score": self._score})

        self._quiz_idx += 1
        self._current_qa = None
        self._quiz_wait_understand = False
        self._quiz_explain_level = 0

        # Build feedback
        if fb_text:
            feedback = fb_text
        else:
            hint = (qa.get("answer","").split()[0]
                    if qa.get("answer") else "")
            feedback = T.feedback(label, delta, hint)

        concept_correct_count = self._concept_quiz_correct_counts.get(qa_concept_id or "", 0)
        # Level-up only when learner shows strong understanding on this concept.
        if newly_mastered and bkt_correct and label == 1 and concept_correct_count >= 2:
            concept_name = self._concept_name(qa_concept_id)
            self.hw_cb("level_up", {"concept": concept_name})
            feedback += " " + T.level_up(concept_name)

        if self._quiz_idx >= QUIZ_TOTAL:
            self._phase = Phase.REWARD
            self.ses_dao.set_phase(self._sid, "reward")
            self.hw_cb("phase", {"phase": "reward"})
            return feedback + " " + self._reward()

        return feedback + " " + self._next_question()

    def _finalize_current_question(self, correct: bool, delta: int, child_answer: str):
        qa = self._current_qa or {}
        qa_concept_id = qa.get("concept_id") or self._concept_id

        newly_mastered = False
        if qa_concept_id:
            bkt_r = self.tools.get_mastery(
                self._child, qa_concept_id, update=correct
            )
            if bkt_r.ok and bkt_r.data:
                newly_mastered = bkt_r.data.get("just_mastered", False)

        self._score += delta
        if correct:
            self._correct += 1
        self.ses_dao.add_score(self._sid, delta, correct)
        self.ses_dao.log_quiz(
            self._sid, self._child, qa_concept_id or "",
            qa.get("question", ""), child_answer,
            1 if correct else 0, delta, qa.get("id")
        )
        self.hw_cb("quiz_result", {"correct": correct, "delta": delta, "score": self._score})

        if newly_mastered:
            concept_name = self._concept_name(qa_concept_id)
            self.hw_cb("level_up", {"concept": concept_name})

        self._quiz_idx += 1
        self._current_qa = None
        self._quiz_wait_understand = False
        self._quiz_explain_level = 0

    def _explain_answer_and_check_understanding(self, qa: Dict) -> str:
        answer = (qa.get("answer") or "").strip()
        if not answer:
            return "Không sao cả. Mình đổi sang câu khác dễ hơn nhé! " + self._next_question()

        explain = (
            f"Không sao đâu. Đáp án là: {answer}. "
            "Bạn đã hiểu ý này chưa?"
        )
        self._quiz_wait_understand = True
        self._quiz_explain_level = 1
        return explain

    def _handle_understanding_check(self, text: str) -> str:
        if self._is_affirmative(text):
            self._finalize_current_question(correct=False, delta=0, child_answer=text)
            if self._quiz_idx >= QUIZ_TOTAL:
                self._phase = Phase.REWARD
                self.ses_dao.set_phase(self._sid, "reward")
                self.hw_cb("phase", {"phase": "reward"})
                return "Tuyệt, vậy mình sang câu tiếp theo nhé! " + self._reward()
            return "Tuyệt, vậy mình sang câu tiếp theo nhé! " + self._next_question()

        if self._is_uncertain(text) or self._is_negative(text):
            answer = (self._current_qa or {}).get("answer", "").strip()
            if self._quiz_explain_level <= 1 and answer:
                self._quiz_explain_level = 2
                return (
                    "Mình giải thích kỹ hơn nhé: "
                    f"{answer}. "
                    "Bạn thử liên hệ với điều bạn vừa học để dễ nhớ hơn. "
                    "Giờ bạn đã hiểu hơn chưa?"
                )

            self._finalize_current_question(correct=False, delta=0, child_answer=text)
            if self._quiz_idx >= QUIZ_TOTAL:
                self._phase = Phase.REWARD
                self.ses_dao.set_phase(self._sid, "reward")
                self.hw_cb("phase", {"phase": "reward"})
                return "Không sao đâu, mình sẽ ôn lại sau nhé. " + self._reward()
            return "Không sao đâu, mình sẽ hỏi câu khác dễ hơn nhé! " + self._next_question()

        return "Bạn trả lời giúp mình 'rồi' hoặc 'chưa' nhé, để mình biết có cần giải thích thêm không."

    def _next_question(self) -> str:
        if not self._concept_id:
            self._phase = Phase.REWARD
            return self._reward()

        route = self.router.route_next_question(
            self._concept_id, self._child,
            self._memory.asked_qa if self._memory else []
        )
        self._log_route(route, from_phase=Phase.QUIZ.value, action="next_question")

        diff_hint = route["kwargs"].get("difficulty", 1)
        progressive_diff = min(5, 1 + self._quiz_idx)
        diff = max(progressive_diff, diff_hint or progressive_diff)
        focus_only = self._quiz_idx < max(2, QUIZ_TOTAL // 2)
        qa = self._select_qa_from_bank(diff, focus_only=focus_only)
        if qa:
            q_text = self._normalize_quiz_question(qa.get("question", ""))
            if self._is_repeated_quiz_question(q_text):
                q_text = self._fallback_rotating_question()
            qa["question"] = q_text
            self._current_qa = qa
            qa_id = qa.get("id")
            if qa_id:
                self._asked_qa_ids.add(qa_id)
            if self._memory and qa_id:
                self._memory.track_qa(qa_id)
            if qa_id:
                self.tools.qa.inc_used(qa_id)
            self._remember_quiz_question(q_text)
            return q_text

        # Deterministic fallback when QA bank is missing for current concepts.
        q_text = self._fallback_rotating_question()
        self._current_qa = {
            "question": q_text,
            "answer": self._default_quiz_answer(),
            "id": None,
            "concept_id": self._concept_id,
        }
        self._remember_quiz_question(q_text)
        return q_text

        return "Bạn hãy kể thêm về chủ đề này nhé!"

    def _reward(self) -> str:
        n = self._memory.episodic.taught_count if self._memory else self._teach_turns
        resp = T.reward(n=n, ok=self._correct,
                        total=self._quiz_idx, score=self._score)
        summary = self._session_teaching_summary()
        if summary:
            resp = f"{resp} {summary}"
        self.hw_cb("celebrate", {"score": self._score})
        self._completed_cycles += 1
        self._post_session_offer_pending = True
        self._phase = Phase.IDLE
        self.ses_dao.set_phase(self._sid, "idle")
        return (
            f"{resp} Bạn muốn học tiếp chủ đề mới, nghe kể chuyện hay chuyển sang bắt chước? "
            "Nếu muốn dừng, bạn nói: chào tạm biệt."
        )

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = (text or "").replace("đ", "d").replace("Đ", "D")
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @classmethod
    def _story_audio_filename(cls, story_id: str, title: str) -> str:
        sid = re.sub(r"\D", "", story_id or "")
        sid = sid.zfill(2) if sid else "00"
        base = cls._strip_accents(title).lower()
        base = re.sub(r"[^a-z0-9]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        return f"{sid}_{base or 'story'}.wav"

    def _session_teaching_summary(self) -> str:
        if not self._memory:
            return ""
        teach_lines = self._memory.episodic.recent_teachings(n=4)
        if teach_lines:
            bullets = " ; ".join(f"{i+1}) {t}" for i, t in enumerate(teach_lines))
            return (
                f"Tổng kết hôm nay, bạn đã dạy mình: {bullets}. "
                "Cảm ơn bạn đã dạy rất rõ ràng, bạn giỏi lắm!"
            )
        return "Cảm ơn bạn đã dạy mình những kiến thức thật bổ ích!"

    def _append_history_log(self, role: str, text: str):
        try:
            history_dir = Path(LOG_DIR) / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            log_file = history_dir / "conservation_1.log"
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            clean_text = " ".join((text or "").replace("\n", " ").split())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{ts}\tdate={datetime.now().date()}\t"
                    f"session={self._sid or '-'}\tchild={self._child or '-'}\t"
                    f"phase={self._phase.value}\t{role}: {clean_text}\n"
                )
        except Exception as e:
            logger.warning(f"History log write failed: {e}")

    # ══════════════════════════════════════════════════════════
    # ROUTE EXECUTOR
    # ══════════════════════════════════════════════════════════
    def _execute_route(self, route: Dict, text: str) -> str:
        reason = route["reason"]
        kwargs = route["kwargs"]
        pth    = route["path"]

        if pth == RPath.A:
            if reason == "greeting":
                return T.greeting()
            if reason in ("teach_ack", "confuse"):
                stage = self._get_teaching_stage() if reason == "teach_ack" else 1
                return T.ack(stage=stage, concept=self._concept_id_name())
            return T.unknown()

        if pth == RPath.B:
            ctx  = kwargs.get("context", "")
            sim  = kwargs.get("sim", 0.5)
            conf = kwargs.get("confidence", "low")
            hist = self._memory.working.to_str() if self._memory else ""
            sys_p = T.system("teach")
            result = self.llm.grounded(
                question=text, context=ctx, system=sys_p,
                history=hist, sim=sim, confidence=conf
            )
            return self._clean_llm_text(result["text"])

        if pth == RPath.C:
            # Cloud fallback (Gemini) is disabled for local-only deployment.
            return self._local_llm_fallback(text)
        return T.unknown()

    def _log_route(self, route: Dict, from_phase: str, action: str = "answer"):
        path = route.get("path", "unknown")
        reason = route.get("reason", "n/a")
        marker = "PATH_A" if path == RPath.A else "PATH_B(LLM)" if path == RPath.B else "PATH_C(CLOUD)"
        logger.info(f"[route] phase={from_phase} action={action} -> {marker} reason={reason}")

    @staticmethod
    def _is_quiz_opt_in(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "quiz", "kiểm tra", "kiem tra", "đố", "hãy kiểm tra", "hỏi đi", "hỏi đy",
            "ok", "oke", "ô kê", "ô cê", "được", "duoc", "bắt đầu", "bat dau", "Oke", "Kiểm tra", "Hỏi đi"
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_uncertain(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "không biết", "khong biet", "không nhớ", "khong nho", "chịu", "chiu",
            "hết", "het", "bí rồi", "bi roi", "con khong biet", "em khong biet",
            "không biết trả lời", "khong biet tra loi",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_teach_continue_request(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "dạy tiếp", "day tiep", "dạy thêm", "day them",
            "muốn dạy tiếp", "muon day tiep", "chưa kiểm tra", "chua kiem tra",
            "để dạy thêm", "de day them",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_teach_stop_signal(text: str) -> bool:
        t = (text or "").lower().strip()
        keys = [
            "mình chỉ biết thế thôi", "minh chi biet the thoi", "mình chỉ biết vậy thôi", "minh chi biet vay thoi",
            "hết rồi", "het roi", "mình hết ý", "minh het y", "không còn gì nữa", "khong con gi nua",
            "chỉ vậy thôi", "chi vay thoi", "vậy thôi", "thế thôi", "the thoi",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _extract_question_only(text: str) -> str:
        cleaned = " ".join((text or "").replace("\n", " ").split())
        cleaned = cleaned.replace('"', "")
        for bad in (
            "[TRẢ LỜI]",
            "Đánh giá:",
            "Trẻ trả lời",
            "Lời giải đúng",
            "Mã câu hỏi",
        ):
            cleaned = cleaned.replace(bad, " ")
        m = re.search(r"([^?]{5,200}\?)", cleaned)
        if m:
            return m.group(1).strip()
        # Fallback: first sentence as a question.
        first = re.split(r"[.!]", cleaned)[0].strip()
        if not first:
            return "Bạn có thể nhắc lại ý chính của chủ đề này không?"
        if not first.endswith("?"):
            first += "?"
        return first

    @staticmethod
    def _is_topic_start_request(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "muốn học", "muon hoc", "học về", "hoc ve", "chủ đề", "chu de",
            "hôm nay học", "hom nay hoc",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _extract_requested_topic(text: str) -> str:
        t = (text or "").strip()
        patterns = [
            r"(?:chủ đề|chu de)\s+(.+)$",
            r"(?:muốn học về|muon hoc ve|học về|hoc ve)\s+(.+)$",
            r"(?:hôm nay học|hom nay hoc)\s+(.+)$",
        ]
        topic = ""
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                topic = m.group(1).strip(" .,!?")
                break
        if not topic:
            return ""
        for tail in ("nhé", "nha", "ạ", "a", "đi", "di", "thế", "the"):
            if topic.lower().endswith(" " + tail):
                topic = topic[: -(len(tail) + 1)].strip()
        return topic

    @staticmethod
    def _is_request_reask_question(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "bạn hỏi", "ban hoi", "hỏi tớ", "hoi to", "hỏi đi", "hoi di",
            "câu hỏi", "cau hoi", "hỏi lại", "hoi lai",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_request_next_question(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "hỏi tiếp", "hoi tiep", "câu khác", "cau khac",
            "đổi câu", "doi cau", "tiếp đi", "tiep di",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        t = (text or "").lower().strip()
        keys = [
            "rồi", "roi", "hiểu rồi", "hieu roi", "mình hiểu", "em hiểu",
            "ok", "oke", "ô kê", "ô cê", "được", "duoc", "biết rồi", "biet roi",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _is_negative(text: str) -> bool:
        t = (text or "").lower().strip()
        keys = [
            "chưa", "chua", "chưa hiểu", "chua hieu", "không hiểu", "khong hieu",
            "vẫn chưa", "van chua", "khó quá", "kho qua",
        ]
        return any(k in t for k in keys)

    @staticmethod
    def _detect_gesture_action(text: str) -> str:
        t = (text or "").lower()
        if not t:
            return ""

        verb_keys = (
            "giơ", "gio", "dơ", "do", "rơ", "ro", "di chuyển", "di chuyen", "nâng", "nang",
            "vẫy", "vay", "đưa", "dua", "nhấc", "nhac", "bat tay", "bắt tay",
        )
        if not any(k in t for k in verb_keys):
            return ""

        if "bắt tay" in t or "bat tay" in t:
            return "handshake"
        if "cả hai tay" in t or "ca hai tay" in t or "hai tay" in t:
            return "both_swing"
        if "tay trái" in t or "tay trai" in t:
            return "left_wave"
        if "tay phải" in t or "tay phai" in t:
            return "right_wave"
        return ""

    def _default_quiz_question(self) -> str:
        concept = self._topic_label or self._concept_id_name() or "chủ đề vừa học"
        return f"Bạn hãy nói 1 đặc điểm đúng của {concept} nhé?"

    def _default_quiz_answer(self) -> str:
        concept = self._topic_label or self._concept_id_name() or "chủ đề này"
        g = self.tools.traverse_graph(concept_id=self._concept_id) if self._concept_id else None
        if g and g.ok and g.data:
            node = g.data.get("node", {})
            desc = (node.get("description") or "").strip() if node else ""
            if desc:
                return desc
        return f"Ý chính là nêu đúng một đặc điểm cơ bản của {concept}."

    def _build_quiz_concept_queue(self):
        queue = []
        if self._concept_id:
            queue.append(self._concept_id)
        current_domain = self._concept_domain(self._concept_id)
        g = self.tools.traverse_graph(concept_id=self._concept_id) if self._concept_id else None
        if g and g.ok and g.data:
            for nb in g.data.get("neighbors", []):
                cid = nb.get("id")
                if current_domain and self._concept_domain(cid) not in ("", current_domain):
                    continue
                if cid and cid not in queue:
                    queue.append(cid)
        return queue

    def _select_qa_from_bank(self, difficulty: int, focus_only: bool = False):
        queue = self._quiz_concept_queue or self._build_quiz_concept_queue()
        if not queue and self._concept_id:
            queue = [self._concept_id]
        self._quiz_concept_queue = queue
        if not queue:
            return None

        focus_concept = self._quiz_focus_concept or self._concept_id
        if focus_only and focus_concept:
            queue = [focus_concept]

        asked = set(self._asked_qa_ids)
        if self._memory:
            asked.update(self._memory.asked_qa)

        start = self._quiz_idx % len(queue)
        concept_order = queue[start:] + queue[:start]

        for cid in concept_order:
            qa_r = self.tools.get_qa(
                cid,
                difficulty=difficulty,
                exclude=list(asked),
                limit=1,
                nearest_difficulty=True,
            )
            if qa_r.ok and qa_r.data and qa_r.data.get("has"):
                qa = qa_r.data["items"][0]
                qa["concept_id"] = qa.get("concept_id") or cid
                return qa
        return None

    def _concept_name(self, concept_id: Optional[str]) -> str:
        if not concept_id:
            return ""
        g = self.tools.traverse_graph(concept_id=concept_id)
        if g.ok and g.data:
            node = g.data.get("node", {})
            if node:
                return node.get("name", "")
        return ""

    def _resolve_topic_concept(self, topic: str) -> Optional[str]:
        """Resolve user topic text to the closest concept by name overlap, then vector score."""
        vec_topic = self.tools.search_concepts(topic, n=12)
        if not (vec_topic.ok and vec_topic.data):
            return None

        tops = vec_topic.data.get("results", [])
        if not tops:
            return None

        target_domain = self._infer_domain_from_text(topic)
        topic_norm = self._normalize_for_match(topic)
        topic_tokens = [tok for tok in topic_norm.split() if tok]
        best_cid = None
        best_key = (-1.0, -1.0, -1.0, -1.0, -1.0, -10_000.0)

        for item in tops:
            cid = item.get("concept_id")
            if not cid:
                continue

            if target_domain:
                cid_domain = self._concept_domain(cid)
                if cid_domain and cid_domain != target_domain:
                    continue

            name = self._concept_name(cid)
            name_norm = self._normalize_for_match(name)
            overlap = self._topic_overlap(topic_norm, name_norm)
            sim = float(item.get("score", 0.0))
            source = self._normalize_for_match(str(item.get("source", "")))
            source_hit = 1.0 if topic_tokens and any(tok in source for tok in topic_tokens) else 0.0
            doc_id = self._concept_doc_prefix(cid)
            doc_title = self._normalize_for_match(self._doc_title(doc_id))
            doc_title_overlap = self._topic_overlap(topic_norm, doc_title)
            exact = 1.0 if name_norm == topic_norm else 0.0
            starts = 1.0 if name_norm.startswith(topic_norm) or topic_norm.startswith(name_norm) else 0.0
            # Prefer shorter/closer names when overlaps tie (e.g., "cá" over "cá voi").
            length_score = -abs(len(name_norm) - len(topic_norm))
            key = (exact, doc_title_overlap, source_hit, starts, overlap + sim * 0.01, length_score)
            if key > best_key:
                best_key = key
                best_cid = cid

        if best_cid:
            return best_cid
        return tops[0].get("concept_id")

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"[^a-zA-Z0-9\sÀ-ỹà-ỹ]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _topic_overlap(topic_norm: str, name_norm: str) -> float:
        if not topic_norm or not name_norm:
            return 0.0
        a = set(topic_norm.split())
        b = set(name_norm.split())
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a))

    @staticmethod
    def _clean_quiz_answer(text: str) -> str:
        ans = " ".join((text or "").replace("\n", " ").split())
        ans = ans.strip("'\" .")
        return ans

    @staticmethod
    def _normalize_quiz_question(text: str) -> str:
        q = " ".join((text or "").replace("\n", " ").split())
        q = q.strip("'\" ")
        if q and not q.endswith("?"):
            q += "?"
        return q

    def _remember_quiz_question(self, question: str):
        normalized = self._normalize_quiz_question(question).lower()
        if normalized:
            self._asked_quiz_questions.add(normalized)

    def _is_repeated_quiz_question(self, question: str) -> bool:
        normalized = self._normalize_quiz_question(question).lower()
        if not normalized:
            return False
        return normalized in self._asked_quiz_questions

    def _fallback_rotating_question(self) -> str:
        concept = self._topic_label or self._concept_id_name() or "chủ đề này"
        cands = [
            f"Bạn nêu 1 đặc điểm của {concept} nhé?",
            f"Bạn có thể kể cho mình biết về một điều mà bạn biết về {concept} được không?",
        ]
        for q in cands:
            if not self._is_repeated_quiz_question(q):
                return q
        return self._default_quiz_question()

    @staticmethod
    def _is_bad_quiz_question(q: str) -> bool:
        t = (q or "").strip().lower()
        if len(t) < 8:
            return True
        bad = [
            "trẻ trả lời",
            "đánh giá",
            "lời giải",
            "mã câu hỏi",
            "[trả lời]",
            "mình chưa biết điều này",
        ]
        return any(k in t for k in bad)

    @staticmethod
    def _clean_llm_text(text: str) -> str:
        cleaned = (text or "").strip()
        for prefix in ("Robot:", "robot:", "Assistant:", "assistant:", "Trợ lý:", "Tro ly:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        for marker in ("Robot:", "robot:", "Assistant:", "assistant:", "Trợ lý:", "Tro ly:"):
            cleaned = cleaned.replace(marker, " ")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _concept_id_name(self) -> str:
        if not self._concept_id:
            return ""
        g = self.tools.traverse_graph(concept_id=self._concept_id)
        if g.ok and g.data:
            node = g.data.get("node", {})
            if node:
                return node.get("name", "")
        return ""

    def _get_teaching_stage(self) -> int:
        """Calculate teaching stage (1-4+) based on _teach_turns.
        Stage 1: 1st teaching turn
        Stage 2: 2nd teaching turn
        Stage 3: 3rd teaching turn
        Stage 4+: 4th+ teaching turns
        """
        if self._teach_turns <= 1:
            return 1
        elif self._teach_turns == 2:
            return 2
        elif self._teach_turns == 3:
            return 3
        else:  # 4+
            return 4

    def _can_adopt_candidate(self, candidate_id: str) -> bool:
        if not candidate_id:
            return False
        if not self._concept_id:
            return True

        # When user explicitly selected a topic, keep concept within the same lesson/doc.
        if self._topic_doc_prefix:
            cand_doc = self._concept_doc_prefix(candidate_id)
            return bool(cand_doc) and cand_doc == self._topic_doc_prefix

        cur_domain = self._topic_domain or self._concept_domain(self._concept_id)
        cand_domain = self._concept_domain(candidate_id)
        if not cur_domain or not cand_domain:
            return True
        return cand_domain == cur_domain

    @staticmethod
    def _concept_doc_prefix(concept_id: str) -> str:
        cid = concept_id or ""
        if "_c_" in cid:
            return cid.rsplit("_c_", 1)[0]
        return ""

    def _doc_title(self, doc_id: str) -> str:
        if not doc_id:
            return ""
        if doc_id in self._doc_title_cache:
            return self._doc_title_cache[doc_id]

        title = ""
        try:
            row = self.ses_dao.db.one("SELECT title FROM documents WHERE id=?", (doc_id,))
            if row:
                title = str(row["title"] or "")
        except Exception:
            title = ""

        self._doc_title_cache[doc_id] = title
        return title

    @staticmethod
    def _concept_domain(concept_id: str) -> str:
        cid = (concept_id or "").lower()
        m = re.match(r"doc_([^_]+)_", cid)
        if m:
            return m.group(1)
        return ""

    @staticmethod
    def _infer_domain_from_text(text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ("động vật", "dong vat", "animal", "animals")):
            return "animal"
        if any(k in t for k in ("vũ trụ", "vu tru", "hệ mặt trời", "he mat troi", "space", "astronomy")):
            return "space"
        return ""

    @staticmethod
    def _is_topic_switch_request(text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "đổi chủ đề", "doi chu de", "chuyển chủ đề", "chuyen chu de",
            "đổi sang", "doi sang", "học chủ đề khác", "hoc chu de khac",
        ]
        return any(k in t for k in keys)

    def _local_llm_fallback(self, text: str) -> str:
        final_fallback = "Bạn hãy hỏi bố mẹ để tìm hiểu thêm về cái này nhé"
        if not (self.llm and self.llm.ready):
            return final_fallback

        prompt = (
            "Bạn là trợ lý cho trẻ em. Trả lời ngắn gọn tối đa 2 câu. "
            "Nếu không chắc chắn hoặc thiếu thông tin, PHẢI trả về đúng câu: "
            f"'{final_fallback}'.\n\n"
            f"Câu hỏi của bé: {text}"
        )
        ans = self._clean_llm_text(self.llm.generate(prompt, max_tokens=80, temperature=0.2))
        if not ans:
            return final_fallback
        uncertain_markers = ["không chắc", "khong chac", "chưa biết", "chua biet", "không có thông tin", "khong co thong tin"]
        if any(m in ans.lower() for m in uncertain_markers):
            return final_fallback
        return ans

    @property
    def phase(self) -> str: return self._phase.value

    def status(self) -> Dict:
        return {
            "session":    self._sid,
            "child":      self._child,
            "phase":      self._phase.value,
            "teach_turns": self._teach_turns,
            "quiz":       f"{self._quiz_idx}/{QUIZ_TOTAL}",
            "score":      self._score,
            "post_session_offer": self._post_session_offer_pending,
            "terminate_requested": self._terminate_requested,
            "mimic_active": self._mimic_active,
        }

    @property
    def last_response_kind(self) -> str:
        return self._last_response_kind

    @property
    def terminate_requested(self) -> bool:
        return self._terminate_requested

    @property
    def last_story_audio_path(self) -> str:
        return self._last_story_audio_path

    @property
    def last_story_title(self) -> str:
        return self._last_story_title



###
# """
# workflow/engine.py — TutorEngine: Conversational AI Tutor
# Thay thế hoàn toàn Engine 5-phase cứng nhắc.

# Thiết kế:
#   - Không có phase cứng: conversation-driven, intent-driven
#   - Quiz inject tự nhiên sau N turns teaching (không force)
#   - Trả lời câu hỏi BẤT KỲ lúc nào (kể cả giữa quiz)
#   - Filter quiz theo doc_id (không lẫn lộn chủ đề)
#   - Xử lý "Mình không biết" nhẹ nhàng (gợi ý, không trừ điểm)
#   - ASR input normalization (ALL CAPS → lowercase)
# """
# import random
# import re
# from enum import Enum
# from typing import Callable, Dict, List, Optional
# from loguru import logger

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
# from config import QUIZ_BANK_RATIO


# # ══════════════════════════════════════════════════════════════
# # TUTOR MODE (thay thế Phase cứng)
# # ══════════════════════════════════════════════════════════════
# class TutorMode(Enum):
#     IDLE      = "idle"      # Chưa bắt đầu session
#     LISTENING = "listening" # Đang lắng nghe trẻ dạy / chat
#     QUIZ_WAIT = "quiz"      # Robot vừa hỏi, chờ trả lời


# # ══════════════════════════════════════════════════════════════
# # TEXT NORMALIZER — Fix ASR all-caps + clean artifacts
# # ══════════════════════════════════════════════════════════════
# def normalize_asr(text: str) -> str:
#     """
#     Sherpa-onnx trả về all-caps, cần lowercase để intent classifier hoạt động.
#     Cũng xử lý artifacts phổ biến từ ASR.
#     """
#     if not text:
#         return ""
#     t = text.strip()
#     # Lowercase nếu all-caps (ASR output)
#     if t.isupper():
#         t = t.lower()
#     # Xóa dấu cách thừa
#     t = re.sub(r"\s+", " ", t).strip()
#     return t


# # ══════════════════════════════════════════════════════════════
# # TUTOR RESPONSES — phong phú hơn, tự nhiên hơn
# # ══════════════════════════════════════════════════════════════
# class Tutor:
#     """Response templates phù hợp với conversational tutor."""

#     # Khi trẻ dạy điều gì đó
#     TEACH_ACKS = [
#         "Ồ thú vị đó! Bạn biết thêm gì nữa không?",
#         "Mình hiểu rồi! Còn gì nữa về {topic} không?",
#         "Hay quá! {statement} — Bạn học ở đâu vậy?",
#         "Mình ghi nhớ rồi! Bạn có thể giải thích thêm không?",
#         "Wao, mình chưa biết điều đó! Cảm ơn bạn đã dạy!",
#         "À ra vậy! Vậy thì {topic} còn có điều gì thú vị nữa?",
#         "Bạn giỏi quá! Tiếp tục dạy mình đi!",
#     ]

#     # Khi inject quiz tự nhiên
#     QUIZ_INJECT = [
#         "Ồ, bạn đã dạy mình nhiều về {topic} rồi! Mình thử hỏi lại bạn một câu nhé: {question}",
#         "Hay lắm! Để mình xem mình học được gì từ bạn — {question}",
#         "Bạn biết nhiều quá! Mình muốn kiểm tra lại: {question}",
#         "Tiện đây mình hỏi một câu về {topic} nha: {question}",
#     ]

#     # Khi trả lời đúng
#     CORRECT = [
#         "Đúng rồi! Bạn giỏi lắm! ✓",
#         "Chính xác! Bạn nhớ rất tốt!",
#         "Tuyệt vời! Đúng luôn!",
#         "Xuất sắc! Bạn học bài rất kỹ!",
#     ]

#     # Khi gần đúng
#     PARTIAL = [
#         "Gần đúng rồi! Ý bạn đúng nhưng có thể đầy đủ hơn!",
#         "Khá tốt! Câu trả lời đúng thêm một chút nữa là hoàn hảo!",
#         "Đúng một phần! Bạn thử bổ sung thêm không?",
#     ]

#     # Khi sai — không trừ điểm nặng
#     WRONG_GENTLE = [
#         "Chưa đúng lắm, nhưng không sao! Đáp án là: {hint}",
#         "Hmmm, thử lại nhé! Gợi ý: {hint}",
#         "Câu này hơi khó! Thực ra đáp án là: {hint}",
#     ]

#     # Khi trẻ nói "không biết"
#     DONT_KNOW = [
#         "Không sao! Mình nhắc nhé: {hint}. Bạn nhớ chưa?",
#         "Câu này hơi khó! Thực ra là: {hint}. Bây giờ bạn nhớ rồi chứ?",
#         "Mình giải thích cho bạn: {hint}. Sau này nhớ nha!",
#     ]

#     # Khi trả lời câu hỏi từ KB
#     ANSWER_INTRO = [
#         "Theo mình biết thì: {answer}",
#         "À, mình có thể trả lời câu đó! {answer}",
#         "Để mình xem... {answer}",
#         "Câu hỏi hay! {answer}",
#     ]

#     # Khi không biết (out-of-KB)
#     UNKNOWN_ANSWER = [
#         "Mình chưa học về điều đó! Bạn biết không? Dạy mình với!",
#         "Câu hỏi hay quá nhưng mình chưa biết. Bạn có thể giải thích cho mình không?",
#         "Hmm, chủ đề này mình chưa được học. Bạn biết thì dạy mình nhé!",
#     ]

#     GREETINGS = [
#         "Chào bạn! Hôm nay mình cùng học gì nhỉ?",
#         "Xin chào! Bạn muốn dạy mình về chủ đề gì hôm nay?",
#         "Chào! Mình sẵn sàng học rồi, bắt đầu thôi!",
#     ]

#     SESSION_END = [
#         "Buổi học hôm nay rất hay! Bạn đã dạy mình {n_taught} điều và trả lời đúng {correct}/{total} câu! Điểm: {score}! Hẹn gặp lại!",
#         "Cảm ơn bạn đã dạy mình! {correct}/{total} câu đúng, tổng {score} điểm. Bạn học rất giỏi!",
#     ]

#     @classmethod
#     def _p(cls, lst): return random.choice(lst)

#     @classmethod
#     def teach_ack(cls, topic: str = "", statement: str = "") -> str:
#         return cls._p(cls.TEACH_ACKS).format(
#             topic=topic or "điều đó",
#             statement=statement[:30] + "..." if len(statement) > 30 else statement
#         )

#     @classmethod
#     def quiz_inject(cls, topic: str, question: str) -> str:
#         return cls._p(cls.QUIZ_INJECT).format(topic=topic, question=question)

#     @classmethod
#     def correct(cls) -> str: return cls._p(cls.CORRECT)

#     @classmethod
#     def partial(cls) -> str: return cls._p(cls.PARTIAL)

#     @classmethod
#     def wrong(cls, hint: str = "") -> str:
#         return cls._p(cls.WRONG_GENTLE).format(hint=hint or "...")

#     @classmethod
#     def dont_know(cls, hint: str = "") -> str:
#         return cls._p(cls.DONT_KNOW).format(hint=hint or "...")

#     @classmethod
#     def answer_wrap(cls, answer: str) -> str:
#         return cls._p(cls.ANSWER_INTRO).format(answer=answer)

#     @classmethod
#     def unknown(cls) -> str: return cls._p(cls.UNKNOWN_ANSWER)

#     @classmethod
#     def greeting(cls) -> str: return cls._p(cls.GREETINGS)

#     @classmethod
#     def session_end(cls, n_taught: int, correct: int, total: int, score: int) -> str:
#         return cls._p(cls.SESSION_END).format(
#             n_taught=n_taught, correct=correct,
#             total=total, score=score
#         )


# # ══════════════════════════════════════════════════════════════
# # TUTOR ENGINE
# # ══════════════════════════════════════════════════════════════
# class TutorEngine:
#     """
#     Flexible conversational tutor. Không có phase cứng.

#     Flow:
#       - Trẻ nói gì cũng được, bất kỳ lúc nào
#       - Quiz inject tự nhiên sau QUIZ_TRIGGER turns
#       - Trả lời câu hỏi ngay lập tức, không block
#       - Document-aware: quiz chỉ dùng câu hỏi từ doc đang học
#     """

#     QUIZ_TRIGGER    = 4    # Inject quiz sau N teaching utterances
#     QUIZ_COOLDOWN   = 3    # Số turns tối thiểu giữa 2 câu quiz
#     MAX_HINT_TIMES  = 2    # Cho gợi ý tối đa n lần trước khi bỏ qua

#     # Phrases báo hiệu "không biết"
#     _DONT_KNOW_PATTERNS = [
#         "không biết", "chịu", "k biết", "hổng biết", "bí rồi",
#         "không nhớ", "quên rồi", "thôi bỏ qua", "pass", "skip",
#         "i don't know", "dunno", "no idea"
#     ]

#     def __init__(self, tools, llm, cloud_llm, evaluator,
#                  mastery_mgr, session_dao,
#                  hw_cb: Callable = None):
#         self.tools      = tools
#         self.llm        = llm
#         self.cloud      = cloud_llm
#         self.evaluator  = evaluator
#         self.mastery    = mastery_mgr
#         self.ses_dao    = session_dao
#         self.hw_cb      = hw_cb or (lambda e, d: None)

#         from core.intent import IntentClassifier
#         self.clf = IntentClassifier()

#         # Session state
#         self._mode          = TutorMode.IDLE
#         self._sid: str      = None
#         self._child: str    = None
#         self._doc_id: str   = None
#         self._memory        = None

#         # Conversation tracking
#         self._current_concept: Optional[str] = None  # concept_id đang focus
#         self._teach_turns   = 0
#         self._turns_total   = 0
#         self._turns_since_quiz = 0
#         self._score         = 0
#         self._correct       = 0
#         self._total_q       = 0
#         self._asked_qa_ids: List[str] = []

#         # Quiz state
#         self._pending_qa: Optional[Dict]  = None  # Câu hỏi đang chờ trả lời
#         self._hint_count    = 0     # Số lần đã gợi ý cho câu hiện tại
#         self._wrong_count   = 0     # Sai liên tiếp

#         # Topic history
#         self._mentioned_concepts: List[str] = []  # Theo dõi concepts đã đề cập

#     # ── Session ────────────────────────────────────────────────
#     def start_session(self, child_id: str, memory,
#                       doc_id: str = None,
#                       session_id: str = None) -> str:
#         self._sid       = session_id or self.ses_dao.create(child_id, doc_id)
#         self._child     = child_id
#         self._doc_id    = doc_id
#         self._memory    = memory
#         self._mode      = TutorMode.LISTENING
#         self._teach_turns = self._turns_total = self._turns_since_quiz = 0
#         self._score = self._correct = self._total_q = 0
#         self._pending_qa = None
#         self._hint_count = self._wrong_count = 0
#         self._current_concept = None
#         self._asked_qa_ids = []
#         self._mentioned_concepts = []

#         self.ses_dao.set_phase(self._sid, "listening")
#         self.hw_cb("phase", {"phase": "teach"})
#         logger.info(f"TutorSession {self._sid} | child: {child_id} | doc: {doc_id}")
#         return self._sid

#     def end_session(self) -> str:
#         self.ses_dao.end(self._sid)
#         self._mode = TutorMode.IDLE
#         n = self._memory.episodic.taught_count if self._memory else self._teach_turns
#         summary = Tutor.session_end(n, self._correct, self._total_q, self._score)
#         self.hw_cb("celebrate", {"score": self._score})
#         logger.info(f"Session ended: score={self._score}, correct={self._correct}/{self._total_q}")
#         return summary

#     # ── Main entry ─────────────────────────────────────────────
#     def process(self, text: str) -> str:
#         """
#         Nhận text từ ASR → trả về text cho TTS.
#         Entry point duy nhất.
#         """
#         # Normalize ASR output (ALL CAPS → lowercase)
#         text = normalize_asr(text)
#         if not text:
#             return ""

#         self._turns_total += 1
#         self._turns_since_quiz += 1
#         logger.info(f"[{self._mode.value}] ← {text[:80]}")

#         # Add to memory
#         if self._memory:
#             self._memory.add_turn("user", text)

#         # Classify intent
#         from core.intent import Intent
#         intent, conf = self.clf.classify(text, self._mode.value)
#         logger.info(f"[intent] {intent.name} ({conf:.2f})")

#         # ── Route by priority ──────────────────────────────────
#         resp = self._dispatch(text, intent, conf)

#         if self._memory and resp:
#             self._memory.add_turn("assistant", resp)

#         resp = self._clean(resp)
#         logger.info(f"[{self._mode.value}] → {resp[:100]}")
#         return resp

#     def _dispatch(self, text: str, intent, conf: float) -> str:
#         from core.intent import Intent

#         # 1. Greeting — always handle first
#         if intent == Intent.GREETING:
#             return self._on_greeting()

#         # 2. IDLE → auto-start listening
#         if self._mode == TutorMode.IDLE:
#             self._mode = TutorMode.LISTENING
#             return Tutor.greeting()

#         # 3. Quiz pending + child is answering → evaluate
#         if self._pending_qa and self._mode == TutorMode.QUIZ_WAIT:
#             if intent in (Intent.ANSWERING, Intent.TEACHING, Intent.UNKNOWN):
#                 return self._on_quiz_answer(text)

#         # 4. Child is asking a question → ALWAYS answer immediately
#         #    Even during quiz, answer questions first
#         if intent == Intent.ASKING:
#             return self._on_question(text)

#         # 5. Teaching / talking → record and respond
#         return self._on_teaching(text)

#     # ── Handlers ───────────────────────────────────────────────

#     def _on_greeting(self) -> str:
#         if self._mode == TutorMode.IDLE:
#             return Tutor.greeting()
#         # Greeting during session — acknowledge and continue
#         return random.choice([
#             "Chào bạn! Chúng ta đang học nha, tiếp tục đi!",
#             "Hi! Bạn muốn tiếp tục học hay nghỉ?",
#         ])

#     def _on_teaching(self, text: str) -> str:
#         """Trẻ đang dạy / nói chuyện. Lắng nghe và ghi nhớ."""
#         # Detect concept from text
#         concept_id, concept_name, sim = self._detect_concept(text)

#         if concept_id:
#             self._current_concept = concept_id
#             if concept_id not in self._mentioned_concepts:
#                 self._mentioned_concepts.append(concept_id)

#         # Record in memory
#         if self._memory:
#             self._memory.record_teaching(text, concept_id, sim or 0.0)

#         self._teach_turns += 1
#         # Log to DB
#         self.ses_dao.log_taught(
#             self._sid, self._child, text, concept_id, sim or 0.0
#         )

#         # Check: should we inject a quiz?
#         quiz_resp = self._maybe_inject_quiz(concept_id, concept_name)
#         if quiz_resp:
#             return quiz_resp

#         # Normal acknowledgement
#         return Tutor.teach_ack(
#             topic=concept_name or "điều bạn vừa nói",
#             statement=text
#         )

#     def _on_question(self, text: str) -> str:
#         """Trẻ hỏi → tìm trong KB → trả lời."""
#         self.hw_cb("thinking", {})

#         # Hybrid search
#         result = self.tools.hybrid_search(text, self._current_concept)

#         if result.ok and result.data:
#             confidence = result.data.get("confidence", "none")
#             ctx = result.ctx or ""
#             sim = result.data.get("vector", {}).get("top_score", 0.0) \
#                 if result.data.get("vector") else 0.0

#             if confidence in ("high", "low") and ctx and self.llm.ready:
#                 # LLM grounded answer
#                 llm_resp = self.llm.grounded(
#                     question=text,
#                     context=ctx,
#                     system=(
#                         "Bạn là robot gia sư dạy trẻ em. "
#                         "Trả lời ngắn gọn, dễ hiểu (1-2 câu). "
#                         "Chỉ dùng thông tin được cung cấp."
#                     ),
#                     history=self._memory.working.to_str() if self._memory else "",
#                     sim=sim,
#                     confidence=confidence,
#                     max_tokens=80
#                 )
#                 answer = self._clean(llm_resp["text"])
#                 if answer and not llm_resp.get("guarded"):
#                     return Tutor.answer_wrap(answer)

#         # Out of KB → invite to teach
#         return Tutor.unknown()

#     def _on_quiz_answer(self, text: str) -> str:
#         """Đánh giá câu trả lời quiz."""
#         qa = self._pending_qa
#         if not qa:
#             return self._on_teaching(text)

#         # Check "không biết"
#         if self._is_dont_know(text):
#             self._hint_count += 1
#             hint = qa.get("answer", "")[:60]
#             resp = Tutor.dont_know(hint=hint)
#             # Don't penalize BKT for "don't know"
#             # Move to next question
#             self._finish_quiz_item(bkt_correct=False, score_delta=0)
#             next_q = self._try_inject_quiz(self._current_concept,
#                                             self._get_concept_name(self._current_concept))
#             self._mode = TutorMode.LISTENING
#             return resp + (" " + next_q if next_q else "")

#         # Evaluate
#         eval_r = self.evaluator.evaluate(
#             child_answer   = text,
#             question       = qa.get("question", ""),
#             correct_answer = qa.get("answer", "")
#         )

#         label      = eval_r["label"]   # 1=correct, 2=partial, 0=wrong
#         delta      = eval_r["score_delta"]
#         bkt_ok     = eval_r["correct"]

#         # Build feedback
#         if label == 1:
#             feedback = Tutor.correct()
#             self._wrong_count = 0
#         elif label == 2:
#             feedback = Tutor.partial()
#             self._wrong_count = 0
#         else:
#             self._wrong_count += 1
#             hint = qa.get("answer", "")[:50]
#             if self._wrong_count <= self.MAX_HINT_TIMES:
#                 feedback = Tutor.wrong(hint=hint)
#                 # Don't finish quiz item yet — give another chance
#                 # Reset mode to keep waiting
#                 return feedback
#             else:
#                 feedback = Tutor.wrong(hint=hint)
#                 self._wrong_count = 0

#         # BKT update
#         if self._current_concept:
#             bkt_r = self.tools.get_mastery(
#                 self._child, self._current_concept, update=bkt_ok
#             )
#             if bkt_r.ok and bkt_r.data:
#                 if bkt_r.data.get("just_mastered"):
#                     cname = self._get_concept_name(self._current_concept)
#                     self.hw_cb("level_up", {"concept": cname})
#                     feedback += f" Bạn đã hiểu rõ về {cname} rồi!"

#         # Score
#         self._score  += delta
#         if bkt_ok: self._correct += 1
#         self.ses_dao.add_score(self._sid, delta, bkt_ok)
#         self.ses_dao.log_quiz(
#             self._sid, self._child,
#             self._current_concept or "",
#             qa.get("question",""), text,
#             1 if bkt_ok else 0, delta, qa.get("id")
#         )
#         self.hw_cb("quiz_result",
#                    {"correct": bkt_ok, "delta": delta, "score": self._score})

#         # Finish this quiz item
#         self._finish_quiz_item(bkt_correct=bkt_ok, score_delta=delta)
#         self._mode = TutorMode.LISTENING

#         # Naturally continue: maybe inject another quiz or return to chatting
#         follow_up = self._try_inject_quiz(
#             self._current_concept,
#             self._get_concept_name(self._current_concept)
#         )
#         if follow_up:
#             return feedback + " " + follow_up

#         return feedback + " Bạn tiếp tục dạy mình nhé!"

#     def _finish_quiz_item(self, bkt_correct: bool, score_delta: int):
#         self._total_q     += 1
#         self._pending_qa   = None
#         self._hint_count   = 0
#         self._turns_since_quiz = 0

#     # ── Quiz injection ─────────────────────────────────────────

#     def _maybe_inject_quiz(self, concept_id: Optional[str],
#                             concept_name: str = "") -> Optional[str]:
#         """
#         Quyết định có nên inject câu quiz tự nhiên không.
#         Trả về None nếu không inject.
#         """
#         # Điều kiện inject
#         if self._teach_turns < self.QUIZ_TRIGGER:
#             return None
#         if self._turns_since_quiz < self.QUIZ_COOLDOWN:
#             return None
#         if self._pending_qa:
#             return None
#         if not concept_id:
#             return None

#         return self._try_inject_quiz(concept_id, concept_name)

#     def _try_inject_quiz(self, concept_id: Optional[str],
#                           concept_name: str = "") -> Optional[str]:
#         """Thực sự inject quiz nếu có câu hỏi phù hợp."""
#         if not concept_id:
#             return None

#         # Lấy độ khó phù hợp từ BKT
#         m = self.tools.get_mastery(self._child, concept_id)
#         diff = None
#         if m.ok and m.data:
#             p = m.data.get("p_mastery", 0.3)
#             diff = self.mastery.bkt.difficulty(p)

#         # Query qa_bank — filter by concept_id (tự nhiên filter theo doc)
#         qa_r = self.tools.get_qa(
#             concept_id, difficulty=diff,
#             exclude=self._asked_qa_ids, limit=1
#         )

#         if not qa_r.ok or not qa_r.data.get("has"):
#             # No more questions in bank → try LLM adaptive
#             if self.llm.ready and self._teach_turns >= self.QUIZ_TRIGGER * 2:
#                 return self._inject_llm_quiz(concept_id, concept_name)
#             return None

#         qa = qa_r.data["items"][0]
#         self._pending_qa = qa
#         self._asked_qa_ids.append(qa.get("id", ""))
#         self._mode = TutorMode.QUIZ_WAIT
#         self.tools.qa.inc_used(qa.get("id", ""))
#         self.ses_dao.set_phase(self._sid, "quiz")
#         self.hw_cb("phase", {"phase": "quiz"})

#         cname = concept_name or self._get_concept_name(concept_id) or "chủ đề này"
#         return Tutor.quiz_inject(topic=cname, question=qa["question"])

#     def _inject_llm_quiz(self, concept_id: str,
#                           concept_name: str = "") -> Optional[str]:
#         """Tạo câu hỏi adaptive bằng LLM khi qa_bank cạn."""
#         g = self.tools.traverse_graph(concept_id=concept_id)
#         ctx = g.ctx if g.ok else ""
#         if not ctx:
#             return None

#         result = self.llm.grounded(
#             question=f"Đặt 1 câu hỏi ngắn và rõ ràng cho trẻ em về '{concept_name}'.",
#             context=ctx, system="Bạn là giáo viên tạo câu hỏi kiểm tra.",
#             sim=0.9, confidence="high", max_tokens=60
#         )
#         q_text = self._clean(result["text"])
#         if not q_text:
#             return None

#         self._pending_qa = {
#             "question": q_text,
#             "answer":   "",   # LLM-generated, no ground truth
#             "id":       None
#         }
#         self._mode = TutorMode.QUIZ_WAIT
#         cname = concept_name or "chủ đề này"
#         return Tutor.quiz_inject(topic=cname, question=q_text)

#     # ── Helpers ────────────────────────────────────────────────

#     def _detect_concept(self, text: str):
#         """Tìm concept liên quan đến utterance. Returns (id, name, score)."""
#         vec = self.tools.search_concepts(text, n=1)
#         if vec.ok and vec.data:
#             tops = vec.data.get("results", [])
#             if tops and tops[0].get("score", 0) >= 0.35:
#                 cid   = tops[0].get("concept_id", "")
#                 score = tops[0].get("score", 0.0)
#                 name  = self._get_concept_name(cid)
#                 return cid, name, score

#         # Fallback: graph text search
#         matches = self.tools.graph.text_search(text, limit=1)
#         if matches:
#             cid  = matches[0]["id"]
#             name = matches[0].get("name", "")
#             return cid, name, 0.4

#         return None, "", 0.0

#     def _get_concept_name(self, concept_id: Optional[str]) -> str:
#         if not concept_id:
#             return ""
#         g = self.tools.traverse_graph(concept_id=concept_id)
#         if g.ok and g.data:
#             node = g.data.get("node") or {}
#             return node.get("name", "")
#         return ""

#     def _is_dont_know(self, text: str) -> bool:
#         t = text.lower()
#         return any(p in t for p in self._DONT_KNOW_PATTERNS)

#     @staticmethod
#     def _clean(text: str) -> str:
#         """Remove LLM role prefixes and normalize whitespace."""
#         t = (text or "").strip()
#         for pfx in ("Robot:", "robot:", "Assistant:", "Trợ lý:", "Gia sư:"):
#             if t.startswith(pfx):
#                 t = t[len(pfx):].strip()
#         t = re.sub(r"\s+", " ", t)
#         return t

#     # ── Public properties ──────────────────────────────────────

#     @property
#     def mode(self) -> str:
#         return self._mode.value

#     # Backward-compat alias
#     @property
#     def phase(self) -> str:
#         return self._mode.value

#     def status(self) -> Dict:
#         return {
#             "session":       self._sid,
#             "child":         self._child,
#             "mode":          self._mode.value,
#             "teach_turns":   self._teach_turns,
#             "total_turns":   self._turns_total,
#             "quiz_pending":  bool(self._pending_qa),
#             "score":         self._score,
#             "correct":       f"{self._correct}/{self._total_q}",
#             "current_concept": self._current_concept,
#         }