"""
hardware/esp32.py — ESP32 UART Bridge
Giao tiếp qua Serial JSON commands.

ESP32 Arduino firmware cần implement:
  {"cmd":"led",   "r":0,"g":200,"b":50}   → set LED RGB
  {"cmd":"strip", "effect":"rainbow"}      → LED strip effect
  {"cmd":"servo", "id":0, "angle":45}      → servo angle
  {"cmd":"buzz",  "freq":880,"dur":300}    → buzzer beep
  {"cmd":"reset"}                          → all off
"""
import json
import threading
import time
from typing import Dict, Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ESP32_PORT, ESP32_BAUD


class ESP32:
    def __init__(self, port=ESP32_PORT, baud=ESP32_BAUD):
        self._port = port
        self._baud = baud
        self._ser: Optional[object] = None
        self._ok  = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            import serial
            from serial.tools import list_ports

            candidates = [self._port]
            try:
                for p in list_ports.comports():
                    dev = p.device
                    if dev and dev not in candidates:
                        candidates.append(dev)
            except Exception:
                pass

            last_err = None
            for port in candidates:
                try:
                    self._ser = serial.Serial(port, self._baud, timeout=1.0)
                    time.sleep(2.0)   # ESP32 boot
                    self._port = port
                    self._ok = True
                    logger.info(f"ESP32 connected: {self._port}@{self._baud}")
                    return True
                except Exception as e:
                    last_err = e
                    if self._ser:
                        try:
                            self._ser.close()
                        except Exception:
                            pass
                        self._ser = None

            logger.warning(f"ESP32 not available ({last_err}) — mock mode")
            self._ok = False
        except Exception as e:
            logger.warning(f"ESP32 not available ({e}) — mock mode")
            self._ok = False
        return self._ok

    def send(self, cmd: Dict) -> bool:
        if not self._ok:
            logger.debug(f"ESP32 mock: {cmd}")
            return True
        try:
            with self._lock:
                self._ser.write((json.dumps(cmd) + "\n").encode())
                self._ser.flush()
            return True
        except Exception as e:
            logger.error(f"ESP32 send: {e}")
            return False

    def send_char(self, state: str) -> bool:
        """Send one-letter state command expected by Arduino LCD script."""
        token = (state or "").strip()[:1]
        if not token:
            return False
        if not self._ok:
            logger.info(f"esp32: mock char={token}")
            return True
        try:
            with self._lock:
                self._ser.write((token + "\n").encode())
                self._ser.flush()
            logger.info(f"esp32: state/action {token}")
            return True
        except Exception as e:
            logger.error(f"ESP32 send_char: {e}")
            return False

    def close(self):
        if self._ser:
            try: self._ser.close()
            except: pass

    # ── High-level commands ───────────────────────────────────
    def correct(self):
        self.send({"cmd":"led","r":0,"g":200,"b":60})
        self.send({"cmd":"servo","id":0,"angle":45})

    def wrong(self):
        self.send({"cmd":"led","r":200,"g":150,"b":0})
        self.send({"cmd":"servo","id":0,"angle":90})

    def thinking(self):
        # Arduino LCD script: T = THINKING
        logger.info("esp32: Thinking")
        self.send_char("T")

    def celebrate(self):
        # Arduino LCD script: H = HEART_EYES
        logger.info("esp32: Heart")
        self.send_char("H")

    def teach_mode(self):
        # Learning mode should be listening while waiting user input.
        logger.info("esp32: Listening")
        self.send_char("L")

    def quiz_mode(self):
        # Keep quiz as thinking display for now.
        self.send_char("T")

    def listening(self):
        # Arduino LCD script: L = LISTENING
        logger.info("esp32: Listening")
        self.send_char("L")

    def speaking(self):
        # Arduino LCD script: S = SPEAKING
        logger.info("esp32: Speaking")
        self.send_char("S")

    def heart(self):
        # Arduino LCD script: H = HEART_EYES
        logger.info("esp32: Heart")
        self.send_char("H")

    def laughing(self):
        # Arduino LCD script: A = LAUGHING
        logger.info("esp32: Laughing")
        self.send_char("A")

    # ── Servo gestures (r/l/b/g) ──────────────────────────────
    def right_wave(self):
        logger.info("esp32: Right wave")
        self.send_char("r")

    def left_wave(self):
        logger.info("esp32: Left wave")
        self.send_char("l")

    def both_swing(self):
        logger.info("esp32: Both swing")
        self.send_char("b")

    def handshake(self):
        logger.info("esp32: Handshake")
        self.send_char("g")

    def reset(self):
        self.send({"cmd":"reset"})
