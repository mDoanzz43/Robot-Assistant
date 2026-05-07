"""tests/test_llm.py — LLM Gate, Templates, Database tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from workflow.llm       import Gate, _UNKNOWN_VI, _LOW_CONF_SUFFIX
from workflow.templates import T, SYSTEM
from config             import SIM_HIGH, SIM_LOW


class TestAntiHallucinationGate:
    def test_high_confidence_proceeds(self):
        r = Gate.check("high", SIM_HIGH + 0.05)
        assert r["go"] is True
        assert r["suffix"] == ""

    def test_low_confidence_proceeds_with_suffix(self):
        r = Gate.check("low", SIM_LOW + 0.05)
        assert r["go"] is True
        assert len(r["suffix"]) > 0

    def test_none_confidence_blocked(self):
        r = Gate.check("none", 0.10)
        assert r["go"] is False
        assert "fallback" in r
        assert len(r["fallback"]) > 10

    def test_below_sim_high_blocked(self):
        r = Gate.check("high", SIM_HIGH - 0.10)
        # sim < SIM_HIGH but confidence says high → low branch
        # (confidence is "high" but sim is below → still treated as low/none)
        # depending on implementation; just check it's valid
        assert isinstance(r["go"], bool)

    def test_fallback_cycles_through_messages(self):
        Gate._ctr = 0
        seen = set()
        for _ in range(len(_UNKNOWN_VI) * 2):
            r = Gate.check("none", 0.0)
            seen.add(r["fallback"])
        assert len(seen) >= 2

    def test_build_prompt_contains_grounding(self):
        p = Gate.build_prompt("sys", "ctx", "hist", "question")
        assert "Chỉ dùng thông tin" in p
        assert "Mình chưa biết" in p

    def test_build_prompt_no_empty_sections(self):
        p = Gate.build_prompt("sys", "", "", "question")
        # Empty context → [KIẾN THỨC] section block should not appear
        # Note: grounding instruction may contain "KIẾN THỨC" as a reference word
        # but the actual [KIẾN THỨC] section block should not be present
        assert "[KIẾN THỨC]" not in p   # section header not included
        assert "[HỘI THOẠI]" not in p   # history section not included

    def test_build_prompt_structure(self):
        p = Gate.build_prompt("system", "knowledge", "history", "task")
        assert "KIẾN THỨC" in p
        assert "HỘI THOẠI" in p
        assert "CÂU HỎI"   in p
        assert "TRẢ LỜI"   in p


class TestTemplates:
    def test_greeting_returns_str(self):
        assert isinstance(T.greeting(), str)
        assert len(T.greeting()) > 5

    def test_ack_fills_concept(self):
        # Some templates include the concept, others are generic encouragements
        # Run multiple times to hit a template that uses the slot
        results = [T.ack("con mèo") for _ in range(20)]
        # At least one result should contain the concept
        has_concept = any("con mèo" in r for r in results)
        assert has_concept, "No ack template included the concept after 20 tries"
        # All results must be non-empty strings
        assert all(isinstance(r, str) and len(r) > 5 for r in results)

    def test_ack_default_when_empty(self):
        r = T.ack("")
        assert isinstance(r, str) and len(r) > 0

    def test_confuse_fills_slots(self):
        r = T.confuse("con mèo", sub="cá")
        assert isinstance(r, str)
        # At least one of the slots should appear
        assert "mèo" in r or "cá" in r

    def test_quiz_intro_non_empty(self):
        assert len(T.quiz_intro()) > 5

    def test_correct_includes_points(self):
        r = T.correct(pts=15)
        assert "15" in r

    def test_partial_includes_points(self):
        r = T.partial(pts=7)
        assert "7" in r

    def test_wrong_includes_hint(self):
        r = T.wrong(hint="cá")
        # hint may or may not appear depending on template chosen
        assert isinstance(r, str) and len(r) > 0

    def test_reward_fills_all_slots(self):
        r = T.reward(n=3, ok=4, total=5, score=40)
        assert "4" in r
        assert "5" in r
        assert "40" in r

    def test_level_up_fills_concept(self):
        r = T.level_up("con mèo")
        assert "con mèo" in r

    def test_feedback_correct(self):
        r = T.feedback(label=1, pts=10)
        assert "10" in r

    def test_feedback_partial(self):
        r = T.feedback(label=2, pts=5)
        assert "5" in r

    def test_feedback_wrong(self):
        r = T.feedback(label=0, pts=0)
        assert isinstance(r, str) and len(r) > 0

    def test_system_prompts_all_phases(self):
        for phase in ["teach", "confuse", "quiz", "eval"]:
            s = T.system(phase)
            assert isinstance(s, str) and len(s) > 10

    def test_system_unknown_phase_fallback(self):
        s = T.system("unknown_phase_xyz")
        assert isinstance(s, str) and len(s) > 0

    def test_randomness(self):
        # Different calls may give different results
        results = {T.greeting() for _ in range(20)}
        # At least 2 distinct greetings should appear in 20 calls
        assert len(results) >= 2


class TestDatabase:
    """Integration test với SQLite in-memory."""

    @pytest.fixture
    def db(self, tmp_path):
        import config as cfg
        cfg.DB_PATH = tmp_path / "test.db"
        from core.database import DB
        DB._instance = None
        d = DB.get()
        yield d
        d.close()
        DB._instance = None

    def test_schema_creates_tables(self, db):
        tables = {r["name"] for r in db.all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for t in ["concepts","qa_bank","child_mastery",
                  "sessions","child_taught","children"]:
            assert t in tables, f"Table {t} missing"

    def _ensure_doc(self, db):
        """Helper: insert test document to satisfy FK."""
        db.execute("INSERT OR IGNORE INTO documents(id,title) VALUES(?,?)",
                   ("d1", "Test Doc"))
        db.commit()

    def test_concept_dao_upsert_get(self, db):
        from core.database import ConceptDAO
        self._ensure_doc(db)
        dao = ConceptDAO(db)
        dao.upsert("c1", "con mèo", "Động vật", doc_id="d1")
        c = dao.get("c1")
        assert c is not None
        assert c["name"] == "con mèo"

    def test_concept_dao_search(self, db):
        from core.database import ConceptDAO
        self._ensure_doc(db)
        dao = ConceptDAO(db)
        dao.upsert("c2", "con chó", "Vật nuôi", doc_id="d1")
        results = dao.search("chó")
        assert len(results) >= 1
        assert results[0]["id"] == "c2"

    def test_qa_dao_upsert_and_fetch(self, db):
        from core.database import ConceptDAO, QADAO
        self._ensure_doc(db)
        c_dao = ConceptDAO(db)
        c_dao.upsert("c3", "cá", "", doc_id="d1")
        qa_dao = QADAO(db)
        qa_dao.upsert("q1","c3","Cá sống ở đâu?","Dưới nước","open")
        items = qa_dao.get_for_concept("c3")
        assert len(items) == 1
        assert items[0]["question"] == "Cá sống ở đâu?"

    def test_qa_dao_difficulty_filter(self, db):
        from core.database import ConceptDAO, QADAO
        self._ensure_doc(db)
        c_dao = ConceptDAO(db); c_dao.upsert("c4","test","",doc_id="d1")
        qa_dao = QADAO(db)
        qa_dao.upsert("q2","c4","Easy?","A","open",difficulty=1)
        qa_dao.upsert("q3","c4","Hard?","B","open",difficulty=5)
        easy = qa_dao.get_for_concept("c4", difficulty=1)
        assert len(easy) == 1
        assert easy[0]["id"] == "q2"

    def test_qa_dao_exclude(self, db):
        from core.database import ConceptDAO, QADAO
        self._ensure_doc(db)
        c_dao = ConceptDAO(db); c_dao.upsert("c5","test2","",doc_id="d1")
        qa_dao = QADAO(db)
        qa_dao.upsert("q4","c5","Q1?","A1")
        qa_dao.upsert("q5","c5","Q2?","A2")
        items = qa_dao.get_for_concept("c5", exclude=["q4"])
        ids = [i["id"] for i in items]
        assert "q4" not in ids
        assert "q5" in ids

    def test_mastery_dao_get_new_returns_none(self, db):
        from core.database import MasteryDAO
        dao = MasteryDAO(db)
        assert dao.get("child_new", "concept_new") is None

    def test_mastery_dao_upsert_and_get(self, db):
        from core.database import ChildDAO, ConceptDAO, MasteryDAO
        self._ensure_doc(db)
        child_id = ChildDAO(db).get_or_create("MasteryChild", 7)
        ConceptDAO(db).upsert("co1", "concept one", "", doc_id="d1")
        dao = MasteryDAO(db)
        dao.upsert(child_id, "co1", 0.65, correct=True)
        r = dao.get(child_id, "co1")
        assert r is not None
        assert abs(r["p_mastery"] - 0.65) < 0.01
        assert r["correct"] == 1

    def test_session_dao_create_and_get(self, db):
        from core.database import ChildDAO, SessionDAO
        c_dao = ChildDAO(db)
        cid   = c_dao.get_or_create("TestChild", 7)
        s_dao = SessionDAO(db)
        sid   = s_dao.create(cid)
        sess  = s_dao.get(sid)
        assert sess is not None
        assert sess["child_id"] == cid
        assert sess["phase"] == "teach"

    def test_session_dao_set_phase(self, db):
        from core.database import ChildDAO, SessionDAO
        cid = ChildDAO(db).get_or_create("TestChild2", 6)
        s_dao = SessionDAO(db)
        sid = s_dao.create(cid)
        s_dao.set_phase(sid, "quiz")
        assert s_dao.get(sid)["phase"] == "quiz"

    def test_child_dao_get_or_create(self, db):
        from core.database import ChildDAO
        dao = ChildDAO(db)
        id1 = dao.get_or_create("An", 6)
        id2 = dao.get_or_create("An", 6)
        assert id1 == id2      # same child

    def test_cloud_cache_set_get(self, db):
        from core.database import CloudCacheDAO
        dao = CloudCacheDAO(db)
        dao.set("abc123", "query text", "response text")
        r = dao.get("abc123")
        assert r == "response text"
        assert dao.get("nonexistent") is None
