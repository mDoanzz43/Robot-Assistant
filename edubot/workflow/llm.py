"""
workflow/llm.py — LLM Engine + Anti-Hallucination Gate + Cloud Fallback
llama.cpp với CUDA offload cho Jetson Nano Maxwell (arch 53).
"""
import hashlib
import time
from pathlib import Path
from typing import Dict, Iterator, Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

LLM_BACKEND = getattr(cfg, "LLM_BACKEND", "ollama").lower()
OLLAMA_HOST = getattr(cfg, "OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = getattr(cfg, "OLLAMA_MODEL", "qwen2.5:1.5b")

LLM_MODEL_PATH = getattr(cfg, "LLM_MODEL_PATH", "")
LLM_GPU_LAYERS = getattr(cfg, "LLM_GPU_LAYERS", 0)
LLM_CTX = getattr(cfg, "LLM_CTX", 2048)
LLM_TEMPERATURE = getattr(cfg, "LLM_TEMPERATURE", 0.3)
LLM_MAX_TOKENS = getattr(cfg, "LLM_MAX_TOKENS", 120)
LLM_REPEAT_PEN = getattr(cfg, "LLM_REPEAT_PEN", 1.1)
LLM_N_BATCH = getattr(cfg, "LLM_N_BATCH", 512)

SIM_HIGH = getattr(cfg, "SIM_HIGH", 0.62)
SIM_LOW = getattr(cfg, "SIM_LOW", 0.45)
ENABLE_CLOUD_FALLBACK = bool(getattr(cfg, "ENABLE_CLOUD_FALLBACK", False))


# ══════════════════════════════════════════════════════════════
# ANTI-HALLUCINATION GATE
# ══════════════════════════════════════════════════════════════
_UNKNOWN_VI = [
    "Ồ, mình chưa học về điều này! Bạn có thể dạy mình không?",
    "Mình chưa biết cái này. Bạn biết không? Dạy mình với!",
    "Câu này khó quá, mình chưa tìm được câu trả lời. Hỏi thầy cô nhé!",
    "Mình chưa có thông tin đó. Bạn muốn dạy mình không?",
]
_LOW_CONF_SUFFIX = [
    " (Mình không chắc lắm — hỏi thêm thầy cô nhé!)",
    " Bạn kiểm tra lại với sách hoặc thầy cô để chắc hơn nhé!",
]

_GROUNDING = (
    "Chỉ dùng thông tin được cung cấp bên dưới để trả lời. "
    "Nếu không đủ thông tin, hãy nói thẳng: 'Mình chưa biết điều này.' "
    "Trả lời ngắn gọn, dễ hiểu cho trẻ em, tối đa 2 câu."
)

_STOP_SEQS = ["</s>", "[/INST]", "Bé:", "Trẻ:", "User:", "\n\n\n"]


class Gate:
    """Hard constraint: không gọi LLM khi không có grounding."""
    _ctr = 0

    @classmethod
    def check(cls, confidence: str, sim: float) -> Dict:
        if confidence == "high" and sim >= SIM_HIGH:
            return {"go": True, "suffix": ""}
        if confidence == "low" and sim >= SIM_LOW:
            s = _LOW_CONF_SUFFIX[cls._ctr % len(_LOW_CONF_SUFFIX)]
            cls._ctr += 1
            return {"go": True, "suffix": s}
        # Below threshold
        r = _UNKNOWN_VI[cls._ctr % len(_UNKNOWN_VI)]
        cls._ctr += 1
        return {"go": False, "fallback": r}

    @staticmethod
    def build_prompt(system: str, context: str,
                     history: str, question: str) -> str:
        parts = [f"{system}\n{_GROUNDING}"]
        if context:  parts.append(f"[KIẾN THỨC]\n{context}")
        if history:  parts.append(f"[HỘI THOẠI]\n{history}")
        parts.append(f"[CÂU HỎI]\n{question}")
        parts.append("[TRẢ LỜI]")
        return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
# LLM ENGINE
# ══════════════════════════════════════════════════════════════
class LLMEngine:
    """
    Singleton LLM. Gọi load() 1 lần lúc boot.
    Model: Qwen2.5-1.5B-Instruct Q4_K_M
    """

    def __init__(self):
        self._llm   = None
        self._ollama = None
        self._ready = False
        self._calls = 0
        self._tokens = 0
        self.gate   = Gate()
        self._backend = LLM_BACKEND

    def load(self) -> bool:
        if self._ready:
            return True

        if self._backend == "ollama":
            try:
                import ollama
                self._ollama = ollama.Client(host=OLLAMA_HOST)
                self._ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt="hi",
                    options={"num_predict": 1, "temperature": 0.0},
                )
                self._ready = True
                logger.info(f"LLM loaded via Ollama (CPU/GPU by Ollama): {OLLAMA_MODEL}")
                return True
            except Exception as e:
                logger.error(f"Ollama load failed: {e}")
                return False

        if not Path(LLM_MODEL_PATH).exists():
            logger.error(
                f"Model not found: {LLM_MODEL_PATH}\n"
                "Download: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF\n"
                "File: qwen2.5-1.5b-instruct-q4_k_m.gguf"
            )
            return False
        try:
            from llama_cpp import Llama
            t0 = time.perf_counter()
            self._llm = Llama(
                model_path   = str(LLM_MODEL_PATH),
                n_gpu_layers = LLM_GPU_LAYERS,
                n_ctx        = LLM_CTX,
                n_batch      = LLM_N_BATCH,
                verbose      = False,
                use_mlock    = True,
                use_mmap     = True,
            )
            self._ready = True
            logger.info(
                f"LLM loaded in {time.perf_counter()-t0:.1f}s — "
                f"GPU layers: {LLM_GPU_LAYERS}"
            )
            return True
        except ImportError:
            logger.error(
                "llama-cpp-python not built with CUDA.\n"
                "Build:\n"
                "  CMAKE_ARGS='-DLLAMA_CUBLAS=ON "
                "-DCMAKE_CUDA_ARCHITECTURES=53' "
                "FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir"
            )
            return False
        except Exception as e:
            logger.error(f"LLM load failed: {e}")
            return False

    def generate(self, prompt: str,
                 max_tokens: int = LLM_MAX_TOKENS,
                 temperature: float = LLM_TEMPERATURE,
                 stop: list = None) -> str:
        if not self._ready:
            return ""
        try:
            if self._backend == "ollama" and self._ollama is not None:
                out = self._ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=False,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "repeat_penalty": LLM_REPEAT_PEN,
                    },
                )
                text = (out.get("response") or "").strip()
                self._calls += 1
                return text

            out = self._llm(
                prompt,
                max_tokens    = max_tokens,
                temperature   = temperature,
                repeat_penalty = LLM_REPEAT_PEN,
                stop          = stop or _STOP_SEQS,
                echo          = False,
            )
            text = out["choices"][0]["text"].strip()
            toks = out.get("usage", {}).get("completion_tokens", 0)
            self._calls  += 1
            self._tokens += toks
            return text
        except Exception as e:
            logger.error(f"LLM generate: {e}")
            return ""

    def stream(self, prompt: str,
               max_tokens: int = LLM_MAX_TOKENS) -> Iterator[str]:
        """Streaming: yield token ngay khi có → TTS bắt đầu sớm hơn."""
        if not self._ready:
            yield ""; return
        try:
            if self._backend == "ollama" and self._ollama is not None:
                for chunk in self._ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=True,
                    options={
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": max_tokens,
                        "repeat_penalty": LLM_REPEAT_PEN,
                    },
                ):
                    yield chunk.get("response", "")
                return

            for chunk in self._llm(
                prompt, max_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
                stop=_STOP_SEQS, echo=False, stream=True
            ):
                yield chunk["choices"][0]["text"]
        except Exception as e:
            logger.error(f"LLM stream: {e}")
            yield ""

    def grounded(self, question: str, context: str, system: str,
                 history: str = "", sim: float = 1.0,
                 confidence: str = "high",
                 max_tokens: int = LLM_MAX_TOKENS) -> Dict:
        """
        Generate với anti-hallucination gate.
        Returns {text, guarded, confidence, latency_ms}
        """
        check = Gate.check(confidence, sim)
        if not check["go"]:
            return {
                "text":    check["fallback"],
                "guarded": True,
                "confidence": "none",
                "latency_ms": 0
            }
        prompt = Gate.build_prompt(system, context, history, question)
        t0 = time.perf_counter()
        text = self.generate(prompt, max_tokens=max_tokens)
        ms   = round((time.perf_counter()-t0)*1000, 1)

        sfx = check.get("suffix", "")
        if sfx and text:
            text = text.rstrip() + sfx

        return {"text": text, "guarded": False,
                "confidence": confidence, "latency_ms": ms}

    @property
    def ready(self) -> bool: return self._ready

    def stats(self) -> Dict:
        return {"calls": self._calls, "tokens": self._tokens,
                "ready": self._ready, "backend": self._backend}


# ══════════════════════════════════════════════════════════════
# CLOUD FALLBACK (Gemini Flash)
# ══════════════════════════════════════════════════════════════
class CloudLLM:
    """Cloud fallback intentionally disabled for local-only setup."""

    def __init__(self, cache_dao=None):
        self._cache_dao = cache_dao   # CloudCacheDAO

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def ask(self, prompt: str, child_age: int = 6) -> str:
        if not ENABLE_CLOUD_FALLBACK:
            return "Mình chưa có dữ liệu này trong bài học hiện tại. Bạn dạy mình thêm nhé!"

        h = self._hash(prompt)
        # Check DB cache
        if self._cache_dao:
            cached = self._cache_dao.get(h)
            if cached:
                logger.debug(f"Cloud cache hit: {h}")
                return cached
        return "Mình chưa có dữ liệu này trong bài học hiện tại. Bạn dạy mình thêm nhé!"
