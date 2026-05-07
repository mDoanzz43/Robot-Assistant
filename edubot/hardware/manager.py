"""
hardware/manager.py — Hardware Event Manager
Nhận events từ WorkflowEngine, dispatch tới ESP32 + LCD.
"""
from typing import Any, Dict
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.esp32 import ESP32
from hardware.lcd   import LCD


class HardwareManager:
    """
    Được pass vào Engine như hw_cb(event, data).
    Events: phase | quiz_result | level_up | celebrate | thinking
    """

    def __init__(self):
        self.esp = ESP32()
        self.lcd = LCD()

    def init(self):
        self.esp.connect()
        return self

    def __call__(self, event: str, data: Dict[str, Any] = None):
        """Callable interface — pass trực tiếp làm hw_cb."""
        data = data or {}
        logger.debug(f"HW [{event}] {data}")

        if event == "phase":
            phase = data.get("phase", "")
            if phase == "teach":
                self.esp.teach_mode()
                self.lcd.show("listening", "Mình đang lắng nghe")
            elif phase == "confuse":
                self.esp.thinking()
                self.lcd.show("confused", "Mình chưa hiểu...")
            elif phase == "quiz":
                self.esp.quiz_mode()
                self.lcd.show("thinking", "Mình hỏi nhé!")
            elif phase == "reward":
                self.esp.heart()
                self.lcd.show("excited", "Tuyệt vời!")
            elif phase == "idle":
                self.esp.listening()
                self.lcd.show("listening", "Bạn muốn học gì tiếp?")

        elif event == "quiz_result":
            if data.get("correct"):
                self.esp.correct()
                self.lcd.show("happy", f"+{data.get('delta',0)} điểm!")
            else:
                self.esp.wrong()
                self.lcd.show("confused", "Thử lại nhé!")

        elif event == "level_up":
            self.esp.celebrate()
            concept = data.get("concept", "")
            self.lcd.show("excited", f"Master: {concept}!")

        elif event == "celebrate":
            self.esp.heart()
            self.lcd.score_screen(
                data.get("score", 0), 0, 0
            )

        elif event == "thinking":
            self.esp.thinking()
            self.lcd.loading()

        elif event == "speaking":
            self.esp.speaking()
            self.lcd.show("happy", "Mình đang nói...")

        elif event == "listening":
            self.esp.listening()
            self.lcd.show("listening", "Mình đang nghe đây")

        elif event == "laughing":
            self.esp.laughing()
            self.lcd.show("happy", "Xin chào nhé!")

        elif event == "gesture":
            action = (data.get("action") or "").lower()
            if action == "left_wave":
                self.esp.left_wave()
                self.lcd.show("happy", "Tay trái đang di chuyển")
            elif action == "right_wave":
                self.esp.right_wave()
                self.lcd.show("happy", "Tay phải đang di chuyển")
            elif action == "both_swing":
                self.esp.both_swing()
                self.lcd.show("happy", "Hai tay đang di chuyển")
            elif action == "handshake":
                self.esp.handshake()
                self.lcd.show("excited", "Mời bạn bắt tay!")

    def shutdown(self):
        self.esp.reset()
        self.lcd.clear()
        self.esp.close()
