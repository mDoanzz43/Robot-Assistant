#!/bin/bash
# scripts/setup.sh — EduRobot Full Setup trên Jetson Nano 4GB
# Ubuntu 18.04 / JetPack 4.6 / Maxwell GPU (CUDA arch 53)
#
# Chạy: bash scripts/setup.sh
# Thời gian ước tính: 30–60 phút

set -e
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GRN}[✓] $*${NC}"; }
warn() { echo -e "${YLW}[!] $*${NC}"; }
err()  { echo -e "${RED}[✗] $*${NC}"; exit 1; }

echo -e "${GRN}"
echo "╔══════════════════════════════════════════╗"
echo "║     EduRobot — Jetson Nano Setup         ║"
echo "╚══════════════════════════════════════════╝${NC}"

# ── Kiểm tra môi trường ────────────────────────────────────────
command -v python3 &>/dev/null || err "python3 not found"
PYTHON=$(which python3)
ok "Python: $($PYTHON --version)"

if command -v nvcc &>/dev/null; then
    ok "CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ,)"
else
    warn "nvcc not found — ensure JetPack 4.6 is installed"
fi

# ── [1/7] System deps ─────────────────────────────────────────
echo -e "\n${GRN}[1/7] System dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    build-essential cmake git \
    libopenblas-dev liblapack-dev \
    libjpeg-dev libpng-dev libfreetype6-dev \
    libsndfile1-dev portaudio19-dev \
    alsa-utils sox \
    screen htop nano
ok "System deps installed"

# # ── [2/7] zram swap (bắt buộc với 4GB RAM) ───────────────────
# echo -e "\n${GRN}[2/7] Setting up 4GB zram swap...${NC}"
# if ! swapon --show | grep -q zram 2>/dev/null; then
#     sudo modprobe zram 2>/dev/null || true
#     # Try zram-config first
#     if dpkg -l | grep -q zram-config; then
#         ok "zram-config already installed"
#     else
#         sudo bash -c 'echo 4294967296 > /sys/block/zram0/disksize' 2>/dev/null && \
#         sudo mkswap /sys/block/zram0 2>/dev/null && \
#         sudo swapon /sys/block/zram0 2>/dev/null && \
#         ok "zram 4GB swap activated" || warn "zram setup failed — continue anyway"
#     fi
# else
#     ok "Swap already configured: $(free -h | grep Swap | awk '{print $2}')"
# fi

# # ── [3/7] Python venv ─────────────────────────────────────────
# echo -e "\n${GRN}[3/7] Python virtual environment...${NC}"
# if [ ! -d "venv" ]; then
#     $PYTHON -m venv venv
#     ok "venv created"
# else
#     ok "venv already exists"
# fi
# source venv/bin/activate
# pip install --upgrade pip setuptools wheel --quiet
# ok "venv active: $(which python)"

# ── [4/7] Python packages ─────────────────────────────────────
echo -e "\n${GRN}[4/7] Installing Python packages...${NC}"
pip install \
    chromadb>=0.5.0 \
    "sentence-transformers>=2.7.0" \
    "networkx>=3.1" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.3.0" \
    "fastapi>=0.111.0" "uvicorn>=0.29.0" "pydantic>=2.7.0" \
    "pyserial>=3.5" \
    "sounddevice>=0.4.6" \
    "Pillow>=10.0.0" \
    "python-dotenv>=1.0.1" \
    "loguru>=0.7.2" \
    "orjson>=3.10.0" \
    "httpx>=0.27.0" \
    "pytest>=7.4.0" "pytest-mock>=3.12.0" \
    --quiet
ok "Python packages installed"

# ── [5/7] llama-cpp-python với CUDA ──────────────────────────
echo -e "\n${GRN}[5/7] Building llama-cpp-python with CUDA...${NC}"
echo "    (Maxwell arch 53 = Jetson Nano GPU)"
echo "    This can take 15–30 minutes..."

if python -c "from llama_cpp import Llama; print('ok')" 2>/dev/null; then
    ok "llama-cpp-python already installed"
else
    export CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=53"
    export FORCE_CMAKE=1
    pip install llama-cpp-python \
        --no-cache-dir \
        2>&1 | grep -E "(Building|Successfully|error|Error)" || true
    if python -c "from llama_cpp import Llama" 2>/dev/null; then
        ok "llama-cpp-python built with CUDA"
    else
        warn "llama-cpp-python build failed — LLM will not be available"
        warn "Try manually: CMAKE_ARGS='-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=53' FORCE_CMAKE=1 pip install llama-cpp-python"
    fi
fi

# ── [6/7] Download LLM model ──────────────────────────────────
echo -e "\n${GRN}[6/7] LLM model...${NC}"
MODEL_DIR="data/models"
MODEL_FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$MODEL_FILE"

mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    ok "Model exists: $MODEL_PATH ($SIZE)"
else
    echo "    Downloading $MODEL_FILE (~1.1GB)..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$MODEL_URL" -O "$MODEL_PATH"
    else
        curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
    fi
    ok "Model downloaded: $MODEL_PATH"
fi

# ── [7/7] Pre-download embedding model ───────────────────────
echo -e "\n${GRN}[7/7] Embedding model (all-MiniLM-L6-v2)...${NC}"
python - <<'EOF'
import sys
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                            cache_folder='data/embed_cache')
    v = m.encode(["test"])
    print(f"    Embedding dims: {len(v[0])}")
    print("    OK")
except Exception as e:
    print(f"    Warning: {e}")
EOF
ok "Embedding model ready"

# ── Test CUDA inference ────────────────────────────────────────
echo -e "\n${GRN}Testing CUDA inference...${NC}"
python - <<'EOF'
import sys
try:
    from llama_cpp import Llama
    llm = Llama("data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                n_gpu_layers=20, n_ctx=256, verbose=False)
    out = llm("Xin chào! ", max_tokens=8)
    text = out["choices"][0]["text"]
    print(f"    Output: {text.strip()}")
    print("    LLM OK")
except Exception as e:
    print(f"    Warning: {e}")
EOF

# ── Run tests ──────────────────────────────────────────────────
echo -e "\n${GRN}Running tests (no hardware required)...${NC}"
python -m pytest tests/ \
    --ignore=tests/test_workflow.py \
    -q --tb=short 2>&1 | tail -20 || warn "Some tests failed — check output"

# ── Setup .env ────────────────────────────────────────────────
if [ ! -f ".env" ]; then
cat > .env <<'ENVEOF'
# EduRobot Environment Variables
GEMINI_API_KEY=your_key_here

# ASR model directory (sherpa-onnx zipformer)
ASR_MODEL_DIR=/path/to/your/sherpa-onnx-model

# Piper TTS
PIPER_BIN=/usr/local/bin/piper
PIPER_MODEL=/path/to/your/vi_VN-vais1000-medium.onnx
ENVEOF
    ok ".env created — edit with your actual paths"
fi

# ── Systemd service ────────────────────────────────────────────
cat > /tmp/edurobot.service <<SVCEOF
[Unit]
Description=EduRobot AI Educational Assistant
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/python main.py --mode production
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

echo ""
echo -e "${GRN}══════════════════════════════════════════${NC}"
echo -e "${GRN}  ✅  EduRobot setup complete!${NC}"
echo -e "${GRN}══════════════════════════════════════════${NC}"
echo ""
echo "  Quick start:"
echo "    source venv/bin/activate"
echo "    python main.py                    # interactive (debug)"
echo "    python main.py --mode production  # with ASR+TTS"
echo ""
echo "  Build knowledge base (run on PC first):"
echo "    export GEMINI_API_KEY=your_key"
echo "    python scripts/offline_build.py \\"
echo "        --input docs/bai1.txt \\"
echo "        --topic 'động vật' --age 6"
echo "    rsync -avz data/ user@jetson:~/edubot/data/"
echo ""
echo "  Edit ASR+TTS paths:"
echo "    nano .env"
echo "    nano hardware/asr_bridge.py   (model file names)"
echo ""
echo "  Install as service:"
echo "    sudo cp /tmp/edurobot.service /etc/systemd/system/"
echo "    sudo systemctl enable edurobot"
echo "    sudo systemctl start edurobot"
echo -e "${GRN}══════════════════════════════════════════${NC}"
