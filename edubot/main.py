"""
main.py — EduRobot Bootstrap & Run Loop
Khởi động toàn bộ hệ thống theo thứ tự đúng.

Usage:
    python main.py                    # interactive text mode (debug)
    python main.py --mode production  # ASR + TTS (hardware)
    python main.py --mode api         # FastAPI REST server
"""
import argparse
import json
import os
import signal
import sys
import threading
import uuid
import re
from pathlib import Path
from typing import Optional
from loguru import logger

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as cfg


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
def _setup_log():
    logger.remove()
    logger.add(sys.stderr, level=cfg.LOG_LEVEL,
               format="<green>{time:HH:mm:ss}</green> | "
                      "<level>{level:<8}</level> | {message}")
    logger.add(cfg.LOG_FILE, rotation="10 MB", level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# ══════════════════════════════════════════════════════════════
# SYSTEM CONTAINER
# ══════════════════════════════════════════════════════════════
class EduBot:
    """Khởi tạo và giữ tất cả components."""

    def __init__(self):
        # DB
        self.db       = None
        self.c_dao    = None
        self.qa_dao   = None
        self.m_dao    = None
        self.s_dao    = None
        self.child_dao = None
        self.cc_dao   = None
        # Knowledge
        self.vector   = None
        self.graph    = None
        self.tools    = None
        # Assessment
        self.mastery  = None
        self.evaluator = None
        # Workflow
        self.llm      = None
        self.cloud    = None
        self.engine   = None
        # Hardware
        self.hw       = None
        self.asr      = None
        self.tts      = None
        self._ready   = False

    # ── Step-by-step init ─────────────────────────────────────
    def _resolve_preferred_bundle(self) -> Optional[Path]:
        """Prefer explicit or fresh offline-built bundles over legacy data/ paths."""
        env_dir = os.getenv("EDUBOT_KB_DIR", "").strip()
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))

        # Auto-prefer freshly rebuilt bundles when present.
        candidates.extend([
            ROOT / "output_data_new",
            ROOT / "output_data_clean24",
        ])

        for d in candidates:
            graph = d / "graph" / "knowledge_graph.json"
            chroma = d / "chroma" / "chroma.sqlite3"
            if graph.exists() and chroma.exists():
                return d
        return None

    def _prepare_runtime_paths(self):
        """Set DB/Graph/Chroma paths before DB init, if a preferred bundle exists."""
        bundle = self._resolve_preferred_bundle()
        if not bundle:
            return

        db = bundle / "edubot.db"
        graph = bundle / "graph" / "knowledge_graph.json"
        chroma = bundle / "chroma"

        if db.exists():
            cfg.DB_PATH = db
        cfg.GRAPH_PATH = graph
        cfg.CHROMA_DIR = chroma
        logger.warning(f"Using knowledge bundle: {bundle}")

    def _detect_knowledge_bundle(self) -> Optional[Path]:
        """Find the richest offline-built bundle under data/documents/*."""
        docs_root = cfg.DATA_DIR / "documents"
        if not docs_root.exists():
            return None

        candidates = []
        for d in docs_root.iterdir():
            if not d.is_dir():
                continue
            graph = d / "graph" / "knowledge_graph.json"
            chroma = d / "chroma" / "chroma.sqlite3"
            if graph.exists() and chroma.exists():
                try:
                    with open(graph, encoding="utf-8") as f:
                        g = json.load(f)
                    n_nodes = len(g.get("nodes", []))
                    n_edges = len(g.get("edges", []))
                except Exception:
                    n_nodes = 0
                    n_edges = 0
                score = n_nodes * 10 + n_edges
                candidates.append((score, graph.stat().st_mtime, d, n_nodes, n_edges))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = candidates[0]
        logger.warning(
            f"Detected bundle candidates={len(candidates)}; "
            f"picked='{best[2].name}' nodes={best[3]} edges={best[4]}"
        )
        return best[2]

    def _prepare_knowledge_paths(self):
        """Use default data paths, fallback to latest built bundle if needed."""
        # Paths were already selected explicitly (e.g., output_data_new).
        if cfg.GRAPH_PATH.exists() and cfg.CHROMA_DIR.exists():
            return

        default_graph = cfg.DATA_DIR / "graph" / "knowledge_graph.json"
        bundle = self._detect_knowledge_bundle()

        if default_graph.exists():
            return
        if not bundle:
            return

        cfg.GRAPH_PATH = bundle / "graph" / "knowledge_graph.json"
        cfg.CHROMA_DIR = bundle / "chroma"
        logger.warning(
            "Default knowledge is empty. "
            f"Using bundle: {bundle}"
        )

    def _init_db(self):
        from core.database import DB, ConceptDAO, QADAO, MasteryDAO
        from core.database import SessionDAO, ChildDAO, CloudCacheDAO
        self.db        = DB.get()
        self.c_dao     = ConceptDAO(self.db)
        self.qa_dao    = QADAO(self.db)
        self.m_dao     = MasteryDAO(self.db)
        self.s_dao     = SessionDAO(self.db)
        self.child_dao = ChildDAO(self.db)
        self.cc_dao    = CloudCacheDAO(self.db)
        self._bootstrap_qa_bank_if_empty()
        logger.info("✓ DB")

    def _bootstrap_qa_bank_if_empty(self):
        try:
            row = self.db.one("SELECT COUNT(1) AS n FROM qa_bank")
            total = int(row["n"]) if row else 0
            if total > 0:
                return

            qa_json = cfg.DATA_DIR / "qa_bank.json"
            if not qa_json.exists():
                return

            with open(qa_json, encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, list) and items:
                self.qa_dao.bulk_load(items)
                logger.warning(f"QA bank was empty; loaded {len(items)} items from {qa_json}")
        except Exception as e:
            logger.warning(f"Could not bootstrap qa_bank.json: {e}")

    def _init_knowledge(self):
        self._prepare_knowledge_paths()
        from knowledge.vector import VectorStore
        from knowledge.graph  import KnowledgeGraph
        from knowledge.tools  import Tools
        self.vector = VectorStore().init()
        self.graph  = KnowledgeGraph().load()
        self.tools  = Tools(self.vector, self.graph,
                            self.qa_dao, None,  # mastery set below
                            self.s_dao)
        logger.info(f"✓ Knowledge: {self.vector.doc_count} vecs, "
                    f"{self.graph.n_nodes} nodes")

    def _init_assessment(self):
        from assessment.bkt       import BKT, MasteryManager
        from assessment.evaluator import AnswerEvaluator
        bkt           = BKT()
        self.mastery  = MasteryManager(self.m_dao, bkt)
        self.tools.mastery = self.mastery        # inject back
        self.evaluator = AnswerEvaluator(self.vector, bkt)
        logger.info("✓ BKT + Evaluator")

    def _init_llm(self):
        from workflow.llm import LLMEngine, CloudLLM
        self.llm   = LLMEngine()
        ok = self.llm.load()
        if ok:
            self.evaluator.llm = self.llm        # inject into evaluator
            logger.info(f"✓ LLM ({self.llm.stats().get('backend', 'unknown')})")
        else:
            logger.warning("⚠ LLM not loaded — PATH_B limited")
        self.cloud = CloudLLM(cache_dao=self.cc_dao)
        logger.info("✓ Cloud fallback disabled (local-only)")

    def _init_hardware(self):
        from hardware.manager import HardwareManager
        self.hw = HardwareManager()
        self.hw.init()
        logger.info("✓ Hardware (ESP32 + LCD)")

    def _init_engine(self):
        from workflow.engine import Engine
        self.engine = Engine(
            tools      = self.tools,
            llm        = self.llm,
            cloud_llm  = self.cloud,
            evaluator  = self.evaluator,
            mastery_mgr = self.mastery,
            session_dao = self.s_dao,
            hw_cb      = self.hw,
        )
        logger.info("✓ Workflow Engine")

    def _init_asr_tts(self):
        from hardware.asr_bridge import ASRBridge
        from hardware.tts_bridge import TTSBridge
        self.asr = ASRBridge()
        self.tts = TTSBridge()
        try:
            self.asr.init()
            logger.info("✓ ASR preloaded")
        except Exception as e:
            logger.warning(f"ASR preload skipped: {e}")
        logger.info("✓ ASR + TTS bridges")

    def boot(self) -> bool:
        # Select runtime knowledge paths before DB is initialized.
        self._prepare_runtime_paths()

        steps = [
            ("Database",    self._init_db),
            ("Knowledge",   self._init_knowledge),
            ("Assessment",  self._init_assessment),
            ("LLM",         self._init_llm),
            ("Hardware",    self._init_hardware),
            ("Engine",      self._init_engine),
            ("ASR/TTS",     self._init_asr_tts),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:
                logger.error(f"Boot failed [{name}]: {e}")
                import traceback; traceback.print_exc()
                return False

        self._ready = True
        logger.info("🤖 EduRobot ready!")
        return True

    def shutdown(self):
        logger.info("Shutting down...")
        if self.hw:
            self.hw.shutdown()
        if self.asr:
            self.asr.stop()
        if self.db:
            self.db.close()
        logger.info("Bye!")

    # ── New session helper ────────────────────────────────────
    def new_session(self, name: str, age: int = 6,
                    doc_id: str = None):
        """Tạo child + memory + start session."""
        from core.memory import Memory
        child_id   = self.child_dao.get_or_create(name, age)
        session_id = self.s_dao.create(child_id, doc_id)
        memory     = Memory(self.s_dao, session_id, child_id)
        self.engine.start_session(
            child_id,
            memory,
            doc_id,
            session_id=session_id,
        )
        return child_id, session_id


# ══════════════════════════════════════════════════════════════
# RUN MODES
# ══════════════════════════════════════════════════════════════

def run_interactive(bot: EduBot):
    """
    Text-based loop — dùng để debug / test không cần hardware ASR/TTS.
    """
    from workflow.templates import T
    print("\n" + "="*50)
    print(" EduRobot — Interactive (text) Mode")
    print(" Gõ 'quit' thoát | 'status' xem trạng thái")
    print("="*50)

    bot.hw("laughing", {})

    name = input("Tên của bé: ").strip() or "Bé"
    age  = int(input("Tuổi (Enter=6): ").strip() or 6)
    doc  = os.getenv("EDUBOT_DOC_ID", "").strip() or None

    child_id, sid = bot.new_session(name, age, doc)
    greeting = (
        f"Xin chào {name}, chúc cậu một ngày tốt lành. "
        "Mình là rô bốt thông minh đồng hành cùng với cậu đây, hôm nay tớ với cậu sẽ học tập về chủ đề gì thế nhỉ?"
    )
    print(f"\nRobot: {greeting}\n")

    while True:
        try:
            user = input("Bé: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user: continue
        if user.lower() == "quit": break
        if user.lower() == "status":
            print(bot.engine.status()); continue

        resp = bot.engine.process(user)
        if resp:
            print(f"\nRobot: {resp}\n")
        if bot.engine.terminate_requested:
            print("--- Session ended ---"); break


def run_interactive_voice(bot: EduBot):
    """Interactive voice loop: mic input + speaker output."""
    from workflow.templates import T

    print("\n" + "=" * 50)
    print(" EduRobot — Interactive (voice) Mode")
    print(" Nói 'quit' để thoát | 'status' để xem trạng thái | Ctrl+C để dừng ngay")
    print("=" * 50)

    bot.hw("laughing", {})

    def _capture_one_utterance(asr, timeout: float = 8.0) -> str:
        """Capture a single ASR utterance by running `listen_loop` in a short-lived thread.
        Returns the recognized text or empty string on timeout.
        """
        stop_evt = threading.Event()
        result = {"text": ""}

        def _cb(t: str):
            # store first utterance and stop
            result["text"] = t.strip()
            stop_evt.set()

        th = threading.Thread(target=asr.listen_loop, args=(_cb, stop_evt), daemon=True)
        th.start()
        stop_evt.wait(timeout=timeout)
        # ensure ASR loop stops
        try:
            asr.stop()
        except Exception:
            pass
        th.join(timeout=1.0)
        return result["text"] or ""

    def _extract_name_from_utterance(text: str) -> str:
        """Simple heuristic NLU to extract a name from common Vietnamese phrases.
        Handles examples like: 'Tôi tên là Hải', 'Hải', 'tên Long'.
        Returns the extracted name or empty string if not found.
        """
        if not text:
            return ""
        t = text.strip()
        # remove surrounding quotes
        t = re.sub(r'^["\'`\s]+|["\'`\s]+$', '', t)
        # common prefixes to strip
        prefixes = [r"^tôi tên là\s+", r"^tên tôi là\s+", r"^mình tên là\s+", r"^tên\s+", r"^tôi là\s+", r"^mình là\s+"]
        low = t.lower()
        for p in prefixes:
            m = re.match(p, low)
            if m:
                # remove the matched prefix from original-case string
                stripped = re.sub(p, "", t, flags=re.IGNORECASE).strip()
                # take first 3 words as name at most
                parts = stripped.split()
                return " ".join(parts[:3]).strip()
        # fallback: if utterance is short (1-3 words), treat it as name
        parts = t.split()
        if 1 <= len(parts) <= 3:
            return " ".join(parts).strip()
        # otherwise no confident name
        return ""

    # Voice-first onboarding: prompt for child's name via TTS and capture via ASR
    doc = os.getenv("EDUBOT_DOC_ID", "").strip() or None
    try:
        bot.asr.init()
    except Exception as e:
        logger.error(f"ASR init failed: {e}")
        logger.info("Fallback to interactive text mode")
        run_interactive(bot)
        return

    prompt = (
        "Hây chào bạn, trợ lý thông minh của bạn đã xuất hiện rồi đây. "
        "Cho mình hỏi bạn tên là gì ý nhỉ?"
    )
    print(f"\nRobot (prompting for name): {prompt}\n")
    bot.hw("speaking", {})
    bot.tts.speak(prompt, length_scale=cfg.TTS_LENGTH_SCALE)
    bot.hw("listening", {})

    # capture one utterance for the name
    name_utt = _capture_one_utterance(bot.asr, timeout=8.0)
    name = _extract_name_from_utterance(name_utt) or "Bé"
    age = 6
    print(f"Detected name utterance: '{name_utt}' -> parsed name: '{name}'")

    bot.new_session(name, age, doc)
    greeting = (
        f"Xin chào {name}, chúc cậu một ngày tốt lành. "
        "Mình là rô bốt thông minh của cậu đây, hôm nay chúng mình sẽ học tập về chủ đề gì nhỉ?"
    )
    print(f"\nRobot: {greeting}\n")
    bot.hw("speaking", {})
    bot.tts.speak(greeting)
    bot.hw("listening", {})

    stop_evt = threading.Event()

    def on_utterance(text: str):
        user = text.strip()
        if not user:
            return

        print(f"Bé: {user}")

        low = user.lower()
        if low == "status":
            st = bot.engine.status()
            print(st)
            bot.asr.pause()
            bot.hw("speaking", {})
            bot.tts.speak("Mình đã in trạng thái ra màn hình rồi nhé.")
            bot.hw("listening", {})
            bot.asr.resume()
            return

        bot.hw("thinking", {})
        resp = bot.engine.process(user)
        if resp:
            print(f"\nRobot: {resp}\n")
            bot.asr.pause()
            bot.hw("speaking", {})
            kind = bot.engine.last_response_kind
            if kind == "story":
                bot.tts.speak(resp, length_scale=cfg.TTS_STORY_LENGTH_SCALE)
                wav_path = bot.engine.last_story_audio_path
                if wav_path:
                    bot.tts.play_wav(wav_path, blocking=True)
                follow_up = "Bạn có muốn nghe câu chuyện khác, học tập tiếp hay bắt chước không?"
                bot.tts.speak(follow_up, length_scale=cfg.TTS_LENGTH_SCALE)
                print(f"\nRobot: {follow_up}\n")
            elif kind == "mimic":
                bot.tts.speak(resp, length_scale=0.86)
            else:
                bot.tts.speak(resp, length_scale=cfg.TTS_LENGTH_SCALE)
            bot.hw("listening", {})
            bot.asr.resume()

        if bot.engine.terminate_requested:
            stop_evt.set()

    asr_thread = threading.Thread(
        target=bot.asr.listen_loop,
        args=(on_utterance, stop_evt),
        daemon=True,
    )
    asr_thread.start()
    logger.info("Interactive voice mode running...")

    try:
        while not stop_evt.is_set():
            stop_evt.wait(timeout=0.3)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received in interactive voice mode")
        stop_evt.set()
        pass
    finally:
        bot.asr.stop()
        asr_thread.join(timeout=2.0)
        print("\n--- Voice session stopped ---")


def run_production(bot: EduBot):
    """
    Production loop: ASR → Engine → TTS.
    """
    from workflow.templates import T

    stop_evt = threading.Event()
    name  = "Bé"
    age   = 6
    child_id, sid = bot.new_session(name, age)

    # Greeting qua TTS
    greeting = (
        f"Xin chào {name}, chúc cậu một ngày tốt lành. "
        "Mình là rô bốt thông minh của cậu đây, hôm nay chúng mình sẽ học tập về chủ đề gì nhỉ?"
    )
    bot.hw("speaking", {})
    bot.tts.speak(greeting)
    bot.hw("listening", {})

    def on_utterance(text: str):
        """Callback từ ASR thread."""
        bot.hw("thinking", {})
        resp = bot.engine.process(text)
        if resp:
            bot.hw("speaking", {})
            kind = bot.engine.last_response_kind
            if kind == "story":
                bot.tts.speak(resp, length_scale=cfg.TTS_STORY_LENGTH_SCALE)
                wav_path = bot.engine.last_story_audio_path
                if wav_path:
                    bot.tts.play_wav(wav_path, blocking=True)
                bot.tts.speak(
                    "Bạn có muốn nghe câu chuyện khác, học tập tiếp hay bắt chước không?",
                    length_scale=cfg.TTS_LENGTH_SCALE,
                )
            elif kind == "mimic":
                bot.tts.speak(resp, length_scale=0.86)
            else:
                bot.tts.speak(resp, length_scale=cfg.TTS_LENGTH_SCALE)
            bot.hw("listening", {})
        if bot.engine.terminate_requested:
            stop_evt.set()

    # Khởi động ASR trong thread riêng
    try:
        bot.asr.init()
    except Exception as e:
        logger.error(f"ASR init failed: {e}")
        logger.info("Falling back to interactive mode")
        run_interactive(bot)
        return

    asr_thread = threading.Thread(
        target=bot.asr.listen_loop,
        args=(on_utterance, stop_evt),
        daemon=True
    )
    asr_thread.start()
    logger.info("Production loop running. Waiting for speech...")

    try:
        stop_evt.wait()    # block main thread
    except KeyboardInterrupt:
        pass
    finally:
        bot.asr.stop()
        asr_thread.join(timeout=2.0)


def run_api(bot: EduBot, host: str = cfg.API_HOST if hasattr(cfg,"API_HOST") else "0.0.0.0",
            port: int = 8765):
    """FastAPI REST server."""
    try:
        from api.server import build_app
        import uvicorn
        app = build_app(bot)
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError as e:
        logger.error(f"API mode requires fastapi + uvicorn: {e}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    _setup_log()
    parser = argparse.ArgumentParser(description="EduRobot")
    parser.add_argument("--mode", choices=["interactive", "interactive_voice", "production", "api"],
                        default="interactive")
    args = parser.parse_args()

    bot = EduBot()

    def _sig(s, f):
        logger.info(f"Signal {s}")
        bot.shutdown(); sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    logger.info("="*50)
    logger.info(f"EduRobot — mode: {args.mode}")
    logger.info("="*50)

    if not bot.boot():
        logger.error("Boot failed"); sys.exit(1)

    try:
        if args.mode == "interactive":
            run_interactive(bot)
        elif args.mode == "interactive_voice":
            run_interactive_voice(bot)
        elif args.mode == "production":
            run_production(bot)
        elif args.mode == "api":
            run_api(bot)
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
    
    
    
