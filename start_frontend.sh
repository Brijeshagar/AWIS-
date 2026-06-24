#!/usr/bin/env bash
# ============================================================
#  AWIS — Start Frontend (macOS / Linux)
#  Requires Node.js 18+ and npm
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

echo ""
echo " AWIS Frontend — React + Vite Dev Server"
echo " ========================================="
echo ""

# ── Node check ───────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo " [ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# ── Install deps if missing ───────────────────────────────────
if [ ! -d "node_modules" ]; then
    echo " Installing npm dependencies ..."
    npm install
fi

echo " Starting Vite dev server ..."
echo " Open: http://localhost:5173"
echo " Press Ctrl+C to stop."
echo ""

exec npm run dev
