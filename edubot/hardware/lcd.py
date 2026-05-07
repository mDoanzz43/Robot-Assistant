"""
hardware/lcd.py — ST7789 TFT 2.4" Display (240×320)
Render facial expressions + text bằng Pillow.
"""
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LCD_W, LCD_H

EXPR = {
    "happy":     {"eyes": "circle",  "mouth": "smile",    "color": (0, 200, 100)},
    "thinking":  {"eyes": "squint",  "mouth": "neutral",  "color": (0, 100, 200)},
    "confused":  {"eyes": "wide",    "mouth": "open",     "color": (200, 130, 0)},
    "excited":   {"eyes": "star",    "mouth": "bigsmile", "color": (200, 0, 160)},
    "listening": {"eyes": "wide",    "mouth": "neutral",  "color": (0, 140, 220)},
    "neutral":   {"eyes": "circle",  "mouth": "neutral",  "color": (100, 100, 120)},
}
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class LCD:
    def __init__(self):
        self._ok  = False
        self._disp = None
        self._init()

    def _init(self):
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._Image = Image
            self._Draw  = ImageDraw
            self._Font  = ImageFont
            self._ok = True
            # Try hardware init (Adafruit / st7789 python driver)
            # Adjust GPIO pins to your actual wiring
            try:
                import board, digitalio, busio, adafruit_st7789
                spi = busio.SPI(board.SCK, MOSI=board.MOSI)
                cs  = digitalio.DigitalInOut(board.D8)
                dc  = digitalio.DigitalInOut(board.D25)
                rst = digitalio.DigitalInOut(board.D24)
                self._disp = adafruit_st7789.ST7789(
                    spi, height=LCD_H, width=LCD_W,
                    rotation=90, cs=cs, dc=dc, rst=rst,
                    baudrate=24_000_000
                )
                logger.info("LCD hardware ready")
            except Exception as hw_e:
                logger.warning(f"LCD hardware not available ({hw_e}) — framebuffer mock")
        except ImportError:
            logger.warning("Pillow not installed — LCD disabled")
            self._ok = False

    def show(self, expr: str = "neutral", text: str = ""):
        if not self._ok:
            logger.debug(f"LCD mock: {expr} | {text[:40]}")
            return
        try:
            img  = self._Image.new("RGB", (LCD_W, LCD_H), (15, 15, 25))
            draw = self._Draw.Draw(img)
            cfg  = EXPR.get(expr, EXPR["neutral"])
            c    = cfg["color"]

            # Eyes
            if cfg["eyes"] == "circle":
                draw.ellipse([55,75,95,115], fill=c)
                draw.ellipse([145,75,185,115], fill=c)
            elif cfg["eyes"] == "wide":
                draw.ellipse([50,70,100,120], fill=c)
                draw.ellipse([140,70,190,120], fill=c)
            elif cfg["eyes"] == "squint":
                draw.rectangle([55,95,95,110], fill=c)
                draw.rectangle([145,95,185,110], fill=c)
            elif cfg["eyes"] == "star":
                for cx, cy in [(75,95),(165,95)]:
                    for dx,dy in [(0,-14),(10,10),(-14,4),(14,4),(-10,10)]:
                        draw.line([cx,cy,cx+dx,cy+dy], fill=c, width=3)

            # Mouth
            if cfg["mouth"] == "smile":
                draw.arc([65,138,175,188], 0, 180, fill=c, width=4)
            elif cfg["mouth"] == "bigsmile":
                draw.arc([55,133,185,193], 0, 180, fill=c, width=6)
            elif cfg["mouth"] == "neutral":
                draw.line([75,163,165,163], fill=c, width=3)
            elif cfg["mouth"] == "open":
                draw.ellipse([85,150,155,185], outline=c, width=3)

            # Text overlay (word wrap)
            if text:
                try:
                    font = self._Font.truetype(_FONT_PATH, 15)
                except Exception:
                    font = self._Font.load_default()
                words, lines, line = text.split(), [], ""
                for w in words:
                    if len(line+w) < 26: line += w + " "
                    else: lines.append(line.strip()); line = w + " "
                if line: lines.append(line.strip())
                y = 205
                for ln in lines[:3]:
                    draw.text((8, y), ln, fill=(230,230,230), font=font)
                    y += 20

            self._push(img)
        except Exception as e:
            logger.error(f"LCD render: {e}")

    def _push(self, img):
        if self._disp:
            import numpy as np
            self._disp.image(img)

    def score_screen(self, score: int, ok: int, total: int):
        self.show("excited", f"Score: {score}\n{ok}/{total} đúng!")

    def loading(self, msg="Đang nghĩ..."):
        self.show("thinking", msg)

    def clear(self):
        self.show("neutral", "")
