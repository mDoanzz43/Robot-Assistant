"""
api/server.py — FastAPI REST Server (optional)
Dùng khi muốn điều khiển robot từ web/app.
"""
from typing import Optional
from loguru import logger


def build_app(bot):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn pydantic")

    app = FastAPI(title="EduRobot API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class ProcessReq(BaseModel):
        text:     str
        child_id: Optional[str] = None

    class SessionReq(BaseModel):
        name:   str
        age:    int = 6
        doc_id: Optional[str] = None

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "llm":    bot.llm.ready if bot.llm else False,
            "phase":  bot.engine.phase if bot.engine else "idle",
            "tools":  bot.tools.stats() if bot.tools else {},
        }

    @app.post("/session/start")
    def start_session(req: SessionReq):
        child_id, sid = bot.new_session(req.name, req.age, req.doc_id)
        return {"session_id": sid, "child_id": child_id,
                "greeting": bot.engine.process("")}

    @app.post("/process")
    def process(req: ProcessReq):
        if not req.text.strip():
            raise HTTPException(400, "Empty text")
        resp = bot.engine.process(req.text)
        return {
            "response": resp,
            "phase":    bot.engine.phase,
            "status":   bot.engine.status(),
        }

    @app.get("/status")
    def status():
        return bot.engine.status() if bot.engine else {}

    @app.get("/mastery/{child_id}")
    def mastery(child_id: str):
        if not bot.mastery:
            raise HTTPException(503, "Mastery not ready")
        return bot.mastery.summary(child_id)

    @app.post("/session/end")
    def end_session():
        if not bot.engine:
            raise HTTPException(503)
        return {"summary": bot.engine.end_session()}

    return app
