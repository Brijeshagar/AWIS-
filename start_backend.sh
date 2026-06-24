#!/usr/bin/env bash
# ============================================================
#  AWIS — Start Backend (macOS / Linux)
#  Requires Python 3.10+ and packages from requirements.txt
# ============================================================
set -euo pipefail

# Move to the repo root (wherever this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo " AWIS Backend — FastAPI + Uvicorn"
echo " =================================="
echo ""

# ── Python check ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo " [ERROR] python3 not found. Install Python 3.10+ and try again."
    exit 1
fi

# ── Virtual environment ───────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    echo " Creating virtual environment (.venv) ..."
    python3 -m venv .venv
    echo " Installing dependencies ..."
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── Load .env if present ──────────────────────────────────────
if [ -f ".env" ]; then
    echo " Loading .env ..."
    export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)
fi

# ── Defaults ─────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

echo " Starting FastAPI on http://${HOST}:${PORT}"
echo " Swagger UI: http://localhost:${PORT}/docs"
echo " Press Ctrl+C to stop."
echo ""

exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --reload
