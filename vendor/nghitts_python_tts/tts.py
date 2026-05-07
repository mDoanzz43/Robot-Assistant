"""
Vendor: Vietnamese TTS Engine (core wrapper)
See original project for full implementation and credits.
"""

# Minimal wrapper placeholder copied from original project for discoverability.
try:
    from python_tts.tts import VietnameseTTS as OriginalVietnameseTTS
except Exception:
    OriginalVietnameseTTS = None

class VietnameseTTS(OriginalVietnameseTTS if OriginalVietnameseTTS else object):
    pass
