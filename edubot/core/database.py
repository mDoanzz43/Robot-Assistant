"""
core/database.py — SQLite Schema + Data Access Objects
Toàn bộ persistent storage trong 1 file SQLite với WAL mode.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ══════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════
_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA cache_size   = -16000;

-- ── STATIC (build offline, immutable) ──────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    lang        TEXT DEFAULT 'vi',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concepts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    parent_id   TEXT REFERENCES concepts(id),
    doc_id      TEXT REFERENCES documents(id),
    difficulty  INTEGER DEFAULT 3 CHECK(difficulty BETWEEN 1 AND 5),
    age_min     INTEGER DEFAULT 4,
    age_max     INTEGER DEFAULT 12,
    verified    INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concept_relations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id  TEXT NOT NULL REFERENCES concepts(id),
    to_id    TEXT NOT NULL REFERENCES concepts(id),
    relation TEXT NOT NULL,
    weight   REAL  DEFAULT 1.0,
    UNIQUE(from_id, to_id, relation)
);

CREATE TABLE IF NOT EXISTS qa_bank (
    id          TEXT PRIMARY KEY,
    concept_id  TEXT NOT NULL REFERENCES concepts(id),
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    q_type      TEXT DEFAULT 'open',
    options     TEXT DEFAULT '[]',
    difficulty  INTEGER DEFAULT 3,
    used_count  INTEGER DEFAULT 0,
    verified    INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- ── DYNAMIC (cập nhật realtime) ─────────────────────────────
CREATE TABLE IF NOT EXISTS children (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    age        INTEGER DEFAULT 6,
    created_at TEXT DEFAULT (datetime('now')),
    last_seen  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    child_id   TEXT NOT NULL REFERENCES children(id),
    doc_id     TEXT REFERENCES documents(id),
    phase      TEXT DEFAULT 'teach',
    score      INTEGER DEFAULT 0,
    correct    INTEGER DEFAULT 0,
    total_q    INTEGER DEFAULT 0,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at   TEXT
);

CREATE TABLE IF NOT EXISTS child_mastery (
    child_id   TEXT NOT NULL REFERENCES children(id),
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    p_mastery  REAL    DEFAULT 0.3,
    attempts   INTEGER DEFAULT 0,
    correct    INTEGER DEFAULT 0,
    last_seen  TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (child_id, concept_id)
);

CREATE TABLE IF NOT EXISTS child_taught (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    child_id    TEXT NOT NULL,
    utterance   TEXT NOT NULL,
    concept_id  TEXT,
    sim_score   REAL DEFAULT 0.0,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quiz_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    child_id     TEXT NOT NULL,
    qa_id        TEXT,
    concept_id   TEXT,
    question     TEXT NOT NULL,
    child_answer TEXT NOT NULL,
    correct      INTEGER NOT NULL,
    score_delta  INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cloud_cache (
    hash     TEXT PRIMARY KEY,
    query    TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── INDICES ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_concepts_doc    ON concepts(doc_id);
CREATE INDEX IF NOT EXISTS idx_concepts_parent ON concepts(parent_id);
CREATE INDEX IF NOT EXISTS idx_qa_concept      ON qa_bank(concept_id, difficulty, used_count);
CREATE INDEX IF NOT EXISTS idx_mastery_child   ON child_mastery(child_id, p_mastery);
CREATE INDEX IF NOT EXISTS idx_session_child   ON sessions(child_id);
CREATE INDEX IF NOT EXISTS idx_taught_session  ON child_taught(session_id);
CREATE INDEX IF NOT EXISTS idx_quiz_session    ON quiz_log(session_id);
"""


# ══════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════
class DB:
    """Thread-safe SQLite wrapper, singleton."""
    _instance: Optional["DB"] = None

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path or cfg.DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def get(cls) -> "DB":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"DB ready: {self._path}")

    def execute(self, sql: str, p: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, p)

    def executemany(self, sql: str, ps) -> sqlite3.Cursor:
        return self._conn.executemany(sql, ps)

    def commit(self):
        self._conn.commit()

    def one(self, sql: str, p: tuple = ()) -> Optional[sqlite3.Row]:
        return self._conn.execute(sql, p).fetchone()

    def all(self, sql: str, p: tuple = ()) -> List[sqlite3.Row]:
        return self._conn.execute(sql, p).fetchall()

    def close(self):
        if self._conn:
            self._conn.close()
            DB._instance = None


# ══════════════════════════════════════════════════════════════
# DAOs
# ══════════════════════════════════════════════════════════════
class ConceptDAO:
    def __init__(self, db: DB): self.db = db

    def upsert(self, id: str, name: str, desc: str = "",
               parent_id: str = None, doc_id: str = None,
               difficulty: int = 3, verified: int = 1):
        self.db.execute(
            "INSERT OR REPLACE INTO concepts"
            "(id,name,description,parent_id,doc_id,difficulty,verified)"
            "VALUES(?,?,?,?,?,?,?)",
            (id, name, desc, parent_id, doc_id, difficulty, verified)
        )
        self.db.commit()

    def get(self, id: str) -> Optional[Dict]:
        r = self.db.one("SELECT * FROM concepts WHERE id=?", (id,))
        return dict(r) if r else None

    def search(self, name: str, limit=5) -> List[Dict]:
        rows = self.db.all(
            "SELECT * FROM concepts WHERE name LIKE ? AND verified=1 LIMIT ?",
            (f"%{name}%", limit)
        )
        return [dict(r) for r in rows]

    def children(self, parent_id: str) -> List[Dict]:
        rows = self.db.all(
            "SELECT * FROM concepts WHERE parent_id=? AND verified=1", (parent_id,))
        return [dict(r) for r in rows]

    def add_relation(self, from_id: str, to_id: str, relation: str, weight=1.0):
        self.db.execute(
            "INSERT OR REPLACE INTO concept_relations(from_id,to_id,relation,weight)"
            "VALUES(?,?,?,?)", (from_id, to_id, relation, weight)
        )
        self.db.commit()

    def relations(self, concept_id: str) -> List[Dict]:
        rows = self.db.all("""
            SELECT cr.*, c.name AS to_name
            FROM concept_relations cr JOIN concepts c ON cr.to_id=c.id
            WHERE cr.from_id=? ORDER BY cr.weight DESC
        """, (concept_id,))
        return [dict(r) for r in rows]


class QADAO:
    def __init__(self, db: DB): self.db = db

    def upsert(self, id: str, concept_id: str, question: str, answer: str,
               q_type="open", options: list = None, difficulty: int = 3):
        self.db.execute(
            "INSERT OR REPLACE INTO qa_bank"
            "(id,concept_id,question,answer,q_type,options,difficulty)"
            "VALUES(?,?,?,?,?,?,?)",
            (id, concept_id, question, answer, q_type,
             json.dumps(options or []), difficulty)
        )
        self.db.commit()

    def get_for_concept(self, concept_id: str, difficulty: int = None,
                        exclude: List[str] = None, limit=3,
                        nearest_difficulty: bool = False) -> List[Dict]:
        params: list = [concept_id, 1]
        sql = "SELECT * FROM qa_bank WHERE concept_id=? AND verified=?"
        order = "used_count ASC, id ASC"
        if difficulty is not None and not nearest_difficulty:
            sql += " AND difficulty=?"
            params.append(difficulty)
        if exclude:
            ph = ",".join("?" * len(exclude))
            sql += f" AND id NOT IN ({ph})"; params.extend(exclude)
        if difficulty is not None and nearest_difficulty:
            sql += " ORDER BY ABS(difficulty - ?) ASC, " + order + " LIMIT ?"
            params.extend([difficulty, limit])
        else:
            sql += " ORDER BY " + order + " LIMIT ?"
            params.append(limit)
        rows = self.db.all(sql, tuple(params))
        out = []
        for r in rows:
            d = dict(r)
            d["options"] = json.loads(d.get("options", "[]"))
            out.append(d)
        return out

    def inc_used(self, id: str):
        self.db.execute("UPDATE qa_bank SET used_count=used_count+1 WHERE id=?", (id,))
        self.db.commit()

    def bulk_load(self, items: List[Dict]):
        for q in items:
            self.upsert(q["id"], q["concept_id"], q["question"],
                        q["answer"], q.get("type","open"),
                        q.get("options",[]), q.get("difficulty",3))


class MasteryDAO:
    def __init__(self, db: DB): self.db = db

    def get(self, child_id: str, concept_id: str) -> Optional[Dict]:
        r = self.db.one(
            "SELECT * FROM child_mastery WHERE child_id=? AND concept_id=?",
            (child_id, concept_id)
        )
        return dict(r) if r else None

    def upsert(self, child_id: str, concept_id: str,
               p: float, correct: bool):
        ex = self.get(child_id, concept_id)
        c = (ex["correct"] if ex else 0) + (1 if correct else 0)
        if ex:
            self.db.execute("""
                UPDATE child_mastery
                SET p_mastery=?,attempts=attempts+1,correct=?,last_seen=datetime('now')
                WHERE child_id=? AND concept_id=?
            """, (p, c, child_id, concept_id))
        else:
            self.db.execute(
                "INSERT INTO child_mastery(child_id,concept_id,p_mastery,attempts,correct)"
                "VALUES(?,?,?,1,?)",
                (child_id, concept_id, p, 1 if correct else 0)
            )
        self.db.commit()

    def all_for_child(self, child_id: str) -> List[Dict]:
        return [dict(r) for r in self.db.all(
            "SELECT cm.*,c.name FROM child_mastery cm "
            "JOIN concepts c ON cm.concept_id=c.id "
            "WHERE cm.child_id=? ORDER BY cm.p_mastery DESC",
            (child_id,)
        )]

    def weak(self, child_id: str, thr=0.4, limit=5) -> List[Dict]:
        return [dict(r) for r in self.db.all("""
            SELECT cm.*,c.name,c.difficulty FROM child_mastery cm
            JOIN concepts c ON cm.concept_id=c.id
            WHERE cm.child_id=? AND cm.p_mastery<? ORDER BY cm.p_mastery LIMIT ?
        """, (child_id, thr, limit))]


class SessionDAO:
    def __init__(self, db: DB): self.db = db

    def create(self, child_id: str, doc_id: str = None) -> str:
        sid = uuid.uuid4().hex[:8]
        self.db.execute(
            "INSERT INTO sessions(id,child_id,doc_id) VALUES(?,?,?)",
            (sid, child_id, doc_id)
        )
        self.db.commit()
        return sid

    def get(self, sid: str) -> Optional[Dict]:
        r = self.db.one("SELECT * FROM sessions WHERE id=?", (sid,))
        return dict(r) if r else None

    def set_phase(self, sid: str, phase: str):
        self.db.execute("UPDATE sessions SET phase=? WHERE id=?", (phase, sid))
        self.db.commit()

    def add_score(self, sid: str, delta: int, correct: bool):
        self.db.execute("""
            UPDATE sessions SET score=score+?,
            correct=correct+?,total_q=total_q+1 WHERE id=?
        """, (delta, 1 if correct else 0, sid))
        self.db.commit()

    def end(self, sid: str):
        self.db.execute(
            "UPDATE sessions SET ended_at=datetime('now') WHERE id=?", (sid,))
        self.db.commit()

    def log_taught(self, sid: str, child_id: str, utterance: str,
                   concept_id: str = None, sim_score: float = 0.0):
        self.db.execute(
            "INSERT INTO child_taught(session_id,child_id,utterance,concept_id,sim_score)"
            "VALUES(?,?,?,?,?)",
            (sid, child_id, utterance, concept_id, sim_score)
        )
        self.db.commit()

    def log_quiz(self, sid: str, child_id: str, concept_id: str,
                 question: str, child_answer: str,
                 correct: int, score_delta: int, qa_id: str = None):
        self.db.execute("""
            INSERT INTO quiz_log
            (session_id,child_id,qa_id,concept_id,question,child_answer,correct,score_delta)
            VALUES(?,?,?,?,?,?,?,?)
        """, (sid, child_id, qa_id, concept_id, question,
              child_answer, correct, score_delta))
        self.db.commit()

    def recent_taught(self, child_id: str, n=5) -> List[Dict]:
        return [dict(r) for r in self.db.all("""
            SELECT utterance,concept_id,created_at FROM child_taught
            WHERE child_id=? ORDER BY created_at DESC LIMIT ?
        """, (child_id, n))]


class ChildDAO:
    def __init__(self, db: DB): self.db = db

    def get_or_create(self, name: str, age: int = 6) -> str:
        r = self.db.one("SELECT id FROM children WHERE name=?", (name,))
        if r:
            cid = r["id"]
            self.db.execute(
                "UPDATE children SET last_seen=datetime('now') WHERE id=?", (cid,))
            self.db.commit()
        else:
            cid = uuid.uuid4().hex[:8]
            self.db.execute(
                "INSERT INTO children(id,name,age) VALUES(?,?,?)", (cid, name, age))
            self.db.commit()
        return cid


class CloudCacheDAO:
    def __init__(self, db: DB): self.db = db

    def get(self, hash: str) -> Optional[str]:
        r = self.db.one("SELECT response FROM cloud_cache WHERE hash=?", (hash,))
        return r["response"] if r else None

    def set(self, hash: str, query: str, response: str):
        self.db.execute(
            "INSERT OR REPLACE INTO cloud_cache(hash,query,response) VALUES(?,?,?)",
            (hash, query, response)
        )
        self.db.commit()
