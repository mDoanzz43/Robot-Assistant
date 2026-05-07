"""tests/test_workflow.py — Workflow Engine integration tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from workflow.engine import Engine, Phase


def _mock_tools(concept_id="c1", confidence="high",
                has_qa=True, p_mastery=0.4):
    t = MagicMock()
    t.search_concepts.return_value = MagicMock(
        ok=True,
        data={"results": [{"concept_id": concept_id, "score": 0.8}],
              "confidence": confidence, "top_score": 0.8},
        ctx="kiến thức về con mèo"
    )
    t.traverse_graph.return_value = MagicMock(
        ok=True,
        data={"concept_id": concept_id,
              "node": {"id": concept_id, "name": "con mèo"},
              "neighbors": [
                  {"id": "c2", "name": "động vật", "relation": "IS_A",
                   "weight": 1.0, "description": "sinh vật sống"}
              ]},
        ctx="con mèo là động vật có 4 chân"
    )
    t.get_mastery.return_value = MagicMock(
        ok=True,
        data={"p_mastery": p_mastery, "p_before": p_mastery - 0.05,
              "p_after": p_mastery, "correct": True,
              "mastered": p_mastery >= 0.7,
              "just_mastered": False, "next_diff": "medium"}
    )
    t.get_qa.return_value = MagicMock(
        ok=True,
        data={"items": [{"id": "qa1", "question": "Con mèo là gì?",
                         "answer": "con mèo là động vật", "type": "open"}],
              "has": has_qa}
    )
    t.vs = MagicMock()
    t.vs.embed_single.return_value = [0.1] * 384
    t.qa = MagicMock()
    return t


def _mock_llm(ready=True, text="Đây là câu trả lời của robot."):
    llm = MagicMock()
    type(llm).ready = PropertyMock(return_value=ready)
    llm.grounded.return_value = {
        "text": text, "guarded": False,
        "confidence": "high", "latency_ms": 100
    }
    llm.generate.return_value = "ĐÚNG! Bạn trả lời rất tốt rồi!"
    return llm


def _mock_evaluator(correct=True, label=1, delta=10):
    ev = MagicMock()
    ev.evaluate.return_value = {
        "correct": correct, "label": label,
        "score_delta": delta, "feedback": None
    }
    return ev


def _mock_memory():
    mem = MagicMock()
    mem.episodic.taught_count = 3
    mem.asked_qa = []
    mem.working.to_str.return_value = "Bé: xin chào\nRobot: chào!"
    return mem


def _make_engine(concept_id="c1", p_mastery=0.4,
                 eval_correct=True, has_qa=True,
                 llm_ready=True):
    tools     = _mock_tools(concept_id=concept_id,
                            p_mastery=p_mastery, has_qa=has_qa)
    llm       = _mock_llm(ready=llm_ready)
    cloud     = MagicMock(); cloud.ask.return_value = "Cloud answer"
    evaluator = _mock_evaluator(correct=eval_correct)
    mastery   = MagicMock()
    mastery.bkt = MagicMock()
    mastery.bkt.thr = 0.7
    mastery.bkt.difficulty.return_value = 3
    mastery.bkt.mastered.return_value = False
    ses_dao   = MagicMock()
    ses_dao.create.return_value = "sid1"
    hw_cb     = MagicMock()

    eng = Engine(tools=tools, llm=llm, cloud_llm=cloud,
                 evaluator=evaluator, mastery_mgr=mastery,
                 session_dao=ses_dao, hw_cb=hw_cb)
    return eng, ses_dao, hw_cb


class TestEngineSession:
    def test_start_session_sets_phase(self):
        eng, ses_dao, _ = _make_engine()
        mem = _mock_memory()
        sid = eng.start_session("child1", mem)
        assert eng.phase == "teach"
        assert sid == "sid1"

    def test_idle_returns_greeting(self):
        eng, ses_dao, _ = _make_engine()
        eng._phase = Phase.IDLE
        resp = eng.process("xin chào")
        assert len(resp) > 0

    def test_status_contains_expected_keys(self):
        eng, ses_dao, _ = _make_engine()
        mem = _mock_memory()
        eng.start_session("child1", mem)
        st = eng.status()
        assert "phase" in st
        assert "score" in st
        assert "quiz"  in st


class TestTeachPhase:
    def test_story_request_allowed_before_learning_progress(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._load_stories = MagicMock(return_value=[
            {
                "id": "01",
                "title": "Thánh Gióng",
                "text": "Ngày xưa có cậu bé Gióng lớn nhanh để đánh giặc cứu nước.",
            }
        ])

        resp = eng.process("bạn ơi, kể chuyện cho mình nghe đi")
        assert "Thánh Gióng" in resp

    def test_story_request_is_deferred_during_teach(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._teach_turns = 1

        resp = eng.process("bạn ơi, kể chuyện cho mình nghe đi")
        assert "học xong phần này" in resp

    def test_mimic_mode_activation_and_echo(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())

        activate = eng.process("bạn ơi, hãy bắt chước nhé")
        assert "bắt chước mode" in activate.lower()

        text = "con mèo đang ngủ"
        resp = eng.process(text)

        assert resp == text

    def test_fun_fact_request_without_data_returns_fallback(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._load_fun_facts = MagicMock(return_value=[])

        resp = eng.process("bạn cho mình một fun fact nhé")
        assert "chưa có fun fact" in resp.lower()

    def test_gesture_left_hand_command_triggers_hw(self):
        eng, _, hw = _make_engine()
        eng.start_session("child1", _mock_memory())

        resp = eng.process("bạn ơi giơ tay trái lên nhé")

        assert "tay tôi di chuyển" in resp.lower()
        hw.assert_any_call("gesture", {"action": "left_wave"})

    def test_gesture_handshake_command_triggers_hw(self):
        eng, _, hw = _make_engine()
        eng.start_session("child1", _mock_memory())

        resp = eng.process("bạn bắt tay nào")

        assert "bắt tay" in resp.lower()
        hw.assert_any_call("gesture", {"action": "handshake"})

    def test_teach_returns_acknowledgement(self):
        eng, _, _ = _make_engine()
        mem = _mock_memory()
        eng.start_session("child1", mem)
        resp = eng.process("con mèo là động vật có 4 chân")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_teach_increments_counter(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        before = eng._teach_turns
        eng.process("con mèo ăn cá")
        assert eng._teach_turns == before + 1

    def test_teach_offers_quiz_after_threshold(self):
        from config import TEACH_TURNS_BEFORE_CONFUSE
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._teach_turns = TEACH_TURNS_BEFORE_CONFUSE - 1
        eng._concept_id = "c1"
        resp = eng.process("con mèo là động vật")
        # Flexible flow: engine stays in teach and offers quiz opt-in.
        assert eng.phase == "teach"
        assert "kiểm tra" in resp or "quiz" in resp

    def test_teach_quiz_opt_in_goes_to_quiz(self):
        from config import TEACH_TURNS_BEFORE_CONFUSE
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._teach_turns = TEACH_TURNS_BEFORE_CONFUSE - 1
        eng._concept_id = "c1"
        eng.process("con mèo là động vật")
        resp = eng.process("quiz nhé")
        assert eng.phase in ("quiz", "reward")
        assert isinstance(resp, str)

    def test_topic_start_uses_explicit_topic_phrase(self):
        eng, _, _ = _make_engine(concept_id="solar_system")
        eng.start_session("child1", _mock_memory())
        eng.tools.search_concepts.side_effect = [
            MagicMock(
                ok=True,
                data={"results": [{"concept_id": "solar_system", "score": 0.9}]},
                ctx=""
            )
        ]
        eng.tools.traverse_graph.return_value = MagicMock(
            ok=True,
            data={
                "concept_id": "solar_system",
                "node": {"id": "solar_system", "name": "mặt trời"},
                "neighbors": []
            },
            ctx=""
        )

        resp = eng.process("Tớ muốn học về chủ đề hệ mặt trời")
        assert "hệ mặt trời" in resp.lower()


class TestConfusePhase:
    def test_confuse_asks_about_sub_concept(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._phase = Phase.CONFUSE
        eng._concept_id = "c1"
        resp = eng.process("con mèo ăn cá và tôm")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_confuse_transitions_to_quiz(self):
        from config import CONFUSE_TURNS
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        eng._phase = Phase.CONFUSE
        eng._concept_id = "c1"
        eng._confuse_turns = CONFUSE_TURNS - 1
        resp = eng.process("động vật có 4 chân là con mèo")
        # Should have transitioned to quiz
        assert eng.phase in ("quiz", "reward")


class TestQuizPhase:
    def test_quiz_asks_question_first(self):
        eng, _, _ = _make_engine()
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng._current_qa = None
        resp = eng.process("con mèo")
        assert isinstance(resp, str) and len(resp) > 0

    def test_quiz_correct_answer_adds_score(self):
        eng, _, hw = _make_engine(eval_correct=True)
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng._current_qa = {
            "id": "qa1", "question": "Con mèo là gì?",
            "answer": "con mèo là động vật"
        }
        before = eng._score
        eng.process("con mèo là động vật")
        assert eng._score >= before

    def test_quiz_wrong_answer_hw_callback(self):
        eng, _, hw = _make_engine(eval_correct=False)
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng._current_qa = {
            "id": "qa1", "question": "?", "answer": "đáp án"
        }
        eng.process("sai rồi")
        hw.assert_called()

    def test_quiz_ends_at_total(self):
        from config import QUIZ_TOTAL
        eng, _, _ = _make_engine()
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng._quiz_idx = QUIZ_TOTAL - 1
        eng._current_qa = {
            "id": "qa_last", "question": "Q", "answer": "A"
        }
        resp = eng.process("trả lời cuối")
        assert eng.phase in ("reward", "idle")

    def test_quiz_unknown_answer_triggers_explain_then_check(self):
        eng, _, _ = _make_engine(eval_correct=False)
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng._current_qa = {
            "id": "qa1", "question": "Con mèo là gì?", "answer": "Con mèo là động vật"
        }

        first = eng.process("mình không biết")
        assert "đáp án" in first.lower()
        assert "hiểu" in first.lower()

        second = eng.process("chưa hiểu")
        assert "giải thích kỹ hơn" in second.lower()

        third = eng.process("rồi")
        assert "câu tiếp theo" in third.lower() or "câu khác" in third.lower()

    def test_quiz_llm_path_has_answer_for_unknown(self):
        eng, _, _ = _make_engine(has_qa=False, llm_ready=True)
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.QUIZ
        eng._concept_id = "c1"
        eng.llm.grounded.return_value = {
            "text": "Q: Hành tinh nào gần Mặt trời nhất?\nA: Sao Thủy là hành tinh gần Mặt trời nhất.",
            "guarded": False,
            "confidence": "high",
            "latency_ms": 100,
        }

        first = eng.process("bắt đầu")
        assert "?" in first
        assert eng._current_qa is not None
        assert (eng._current_qa.get("answer") or "").strip() != ""

        explain = eng.process("mình không biết")
        assert "đáp án là" in explain.lower()


class TestRewardPhase:
    def test_reward_ends_session(self):
        eng, ses_dao, _ = _make_engine()
        mem = _mock_memory()
        eng.start_session("child1", mem)
        eng._phase = Phase.REWARD
        eng._correct = 3
        eng._quiz_idx = 5
        eng._score = 30
        resp = eng._reward()
        assert isinstance(resp, str)
        assert len(resp) > 0
        ses_dao.end.assert_called_once()
        assert eng.phase == "idle"


class TestAntiHallucination:
    def test_out_of_kb_uses_local_fallback(self):
        eng, _, _ = _make_engine()
        eng.start_session("child1", _mock_memory())
        route = {"path": "cloud", "reason": "question_oob",
                 "kwargs": {"text": "lỗ đen vũ trụ là gì"}}
        resp = eng._execute_route(route, "lỗ đen vũ trụ là gì")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_llm_not_called_when_no_context(self):
        tools = _mock_tools(confidence="none")
        tools.search_concepts.return_value = MagicMock(
            ok=True,
            data={"results": [], "confidence": "none", "top_score": 0.1},
            ctx=""
        )
        llm   = _mock_llm()
        cloud = MagicMock(); cloud.ask.return_value = "Cloud"
        ev    = _mock_evaluator()
        m_mgr = MagicMock()
        m_mgr.bkt = MagicMock(); m_mgr.bkt.thr = 0.7
        ses_dao = MagicMock(); ses_dao.create.return_value = "s1"
        eng = Engine(tools=tools, llm=llm, cloud_llm=cloud,
                     evaluator=ev, mastery_mgr=m_mgr,
                     session_dao=ses_dao)
        eng.start_session("c1", _mock_memory())
        # Question with no KB context → cloud path
        route = eng.router._route_question("lỗ đen", "c1")
        assert route["path"] == "cloud"
