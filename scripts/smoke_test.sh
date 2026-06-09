#!/usr/bin/env sh
# Smoke-test production or local API endpoints.
# Usage: ./scripts/smoke_test.sh [BASE_URL]
# Example: ./scripts/smoke_test.sh https://vectormind-ngut.onrender.com

set -e

BASE_URL="${1:-http://localhost:8000}"
SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())' 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo '00000000-0000-4000-8000-000000000001')"

echo "Smoke testing: $BASE_URL"
echo ""

echo "== GET /health =="
curl -sf "$BASE_URL/health" | python -m json.tool || curl -sf "$BASE_URL/health"
echo ""

echo "== GET /status (with session) =="
curl -sf -H "X-Session-Id: $SESSION_ID" "$BASE_URL/status" | python -m json.tool || \
  curl -sf -H "X-Session-Id: $SESSION_ID" "$BASE_URL/status"
echo ""

echo "Smoke test passed."
