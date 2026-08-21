#!/usr/bin/env bash
# LlamaDeck service starter — boots the FastAPI backend on :8770 if not already up.
# Idempotent: re-running when already up just verifies and exits green.
#
# LlamaDeck is a single-process app — FastAPI serves the built SvelteKit static
# from /backend/lld/static, so there's no separate frontend to start.

set -euo pipefail

# Resolve the repo root from this script's location — works from any clone.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

PORT=8770
HEALTH_URL="http://127.0.0.1:$PORT/health"

# Is anything listening on this port? Only used to tell "already running" from
# "something else has the port", so a false negative just means we try to start
# and uvicorn reports the conflict itself.
#
# `ss` is iproute2: Linux only. macOS has neither it nor a `netstat` that takes
# these flags, and without a fallback the check silently answered "free" there
# and the message about a port conflict never appeared.
port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -tln "sport = :$1" 2>/dev/null | grep -q LISTEN
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -qE "[.:]$1[[:space:]]+.*LISTEN"
    else
        return 1
    fi
}
wait_for() {
    local url="$1"; local label="$2"; local tries=60
    for i in $(seq 1 $tries); do
        if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
            echo "  -> $label ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  !! $label did not become ready in ${tries}s — see $LOG_DIR/llamadeck.log"
    return 1
}

echo "======================================"
echo " LlamaDeck service start"
echo "======================================"

if curl -sf --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[llamadeck] already healthy on :$PORT ✓"
elif port_busy $PORT; then
    echo "[llamadeck] port :$PORT busy but /health not responding — manual check needed"
    exit 1
else
    echo "[llamadeck] starting uv run llamadeck serve on :$PORT …"
    cd "$PROJECT_DIR"
    # 9>&- : never inherit the launcher's single-instance flock fd — the backend
    # outlives the launcher and would hold the lock forever (see llamadeck-launcher.sh).
    nohup uv run llamadeck serve > "$LOG_DIR/llamadeck.log" 2>&1 9>&- &
    disown
    wait_for "$HEALTH_URL" "llamadeck"
fi

echo ""
echo "======================================"
echo " READY"
echo "  ui+api : http://127.0.0.1:$PORT/"
echo "  mcp    : http://127.0.0.1:$PORT/mcp/"
echo "  log    : $LOG_DIR/llamadeck.log"
echo "======================================"
