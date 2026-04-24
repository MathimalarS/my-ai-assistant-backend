#!/usr/bin/env bash
# ── DocMind startup script ──────────────────────────────────────────────────
set -e

echo "📚 DocMind — Document Q&A Assistant"
echo "────────────────────────────────────"

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✓ Loaded .env"
fi

# Check Python
python3 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'" \
  && echo "✓ Python OK" || exit 1

# Install deps
echo "📦 Installing dependencies…"
pip install -r requirements.txt -q

echo ""
echo "🚀 Starting server on http://localhost:${PORT:-8000}"
echo "   Docs: http://localhost:${PORT:-8000}/docs"
echo ""

# Run
python3 -m uvicorn app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --reload
