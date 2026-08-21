#!/usr/bin/env bash
# End-to-end tests for llamadeck-launcher.sh, with stubs standing in for the browser,
# the terminal and the backend.
#
# These exist because every launcher bug so far has been a SILENT one: the icon
# is clicked, nothing opens, and no log anywhere says why. Each case below is a
# failure that actually shipped. The assertion is always the same — did a
# browser window get opened, yes or no.
#
# Run: bash scripts/test-launcher.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_LAUNCHER="${LAUNCHER:-$HERE/llamadeck-launcher.sh}"
WORK="$(mktemp -d)"

# Mirror the project layout so the launcher resolves PROJECT_DIR the way it
# does in production and calls its own sibling scripts — including the nested
# `llamadeck-launcher.sh --open-only` the slow path runs inside the terminal, which
# is precisely the call that was broken.
mkdir -p "$WORK/proj/scripts"
cp "$REAL_LAUNCHER" "$WORK/proj/scripts/llamadeck-launcher.sh"
chmod +x "$WORK/proj/scripts/llamadeck-launcher.sh"
LAUNCHER="$WORK/proj/scripts/llamadeck-launcher.sh"
STUB="$WORK/bin"
MARK_BROWSER="$WORK/browser-opened"
MARK_TERM="$WORK/terminal-ran"
PORT=18770                       # nothing listens here: backend is "down"
PASS=0
FAIL=0

mkdir -p "$STUB"

# Every case runs with this as $HOME. See run_launcher.
FAKE_HOME="$WORK/home"
mkdir -p "$FAKE_HOME/.config"

# Stub browser: records that a window would have opened, into whichever marker
# the current case is watching. Per-case markers matter because the slow path
# leaves a terminal (and the launcher inside it) alive on purpose — a straggler
# from case 1 would otherwise write into a later case's marker and be read as
# that case's result.
cat > "$STUB/brave-browser" <<'STUBEOF'
#!/bin/sh
echo "$@" >> "${LLAMADECK_TEST_MARKER:?marker not set}"
exit 0
STUBEOF

# Stub terminal that behaves like ptyxis: runs the command and STAYS IN THE
# FOREGROUND until it finishes. gnome-terminal forks and returns immediately,
# which is why the foreground case went untested and shipped broken.
#
# stdin must stay OPEN. The command ends in `read`, waiting for the human to
# press Enter, and that is what keeps the terminal — and the launcher that
# spawned it — alive while the nested launcher runs. Feeding it /dev/null lets
# `read` return instantly, the terminal exits, the lock is released, and the
# test passes no matter what the launcher does. That mistake made an earlier
# version of this file green against a launcher that was visibly broken.
cat > "$STUB/ptyxis" <<STUBEOF
#!/bin/bash
echo ran >> "$MARK_TERM"
shift            # --new-window
shift            # --
exec "\$@" < <(sleep 25)
STUBEOF

# Stub notifier. report_failure fires for real on the macOS-guard case and on
# every terminal fallback, and notify-send -u critical never expires — so an
# unstubbed run pinned a "LlamaDeck" banner on the developer's actual desktop,
# one per run, until they were dismissed by hand. A test must not touch the
# session it runs in.
#
# Wired in through LLAMADECK_NOTIFY_CMD rather than PATH on purpose: the slow path
# runs the nested launcher under `bash -l`, and a login shell rebuilds PATH
# from the profile scripts, so a PATH stub is silently bypassed exactly where
# a stray notification is most likely. The stub still takes the real
# notify-send arguments, so the --print-id/--replace-id path is genuinely
# exercised; echoing an id back is what lets the launcher store one.
NOTIFY_CMD="$STUB/notify-send"
NOTIFY_LOG="$WORK/notified"
cat > "$NOTIFY_CMD" <<STUBEOF
#!/bin/sh
echo "\$@" >> "$NOTIFY_LOG"
echo 4242
exit 0
STUBEOF
printf '#!/bin/sh\nexit 0\n' > "$STUB/zenity"
printf '#!/bin/sh\nexit 0\n' > "$STUB/osascript"

chmod +x "$STUB/brave-browser" "$STUB/ptyxis" \
         "$NOTIFY_CMD" "$STUB/zenity" "$STUB/osascript"

# A fake llamadeck-start.sh whose "backend" is a plain HTTP responder, so the slow
# path has something real to become healthy against. It sits where the real
# llamadeck-start.sh would, so the launcher finds it without any patching.
cat > "$WORK/proj/scripts/llamadeck-start.sh" <<STUBEOF
#!/bin/bash
python3 -c "
import http.server, threading, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{\"status\":\"ok\"}')
    def log_message(self, *a): pass
s = http.server.HTTPServer(('127.0.0.1', $PORT), H)
open('$WORK/backend.pid','w').write(str(__import__('os').getpid()))
s.serve_forever()
" &
disown
sleep 1
echo "[fake] backend up on $PORT"
STUBEOF
chmod +x "$WORK/proj/scripts/llamadeck-start.sh"

cleanup() {
    [ -f "$WORK/backend.pid" ] && kill "$(cat "$WORK/backend.pid")" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

check() {  # $1=label  $2=expected(open|noop)
    local got="noop"
    [ -s "$MARK_BROWSER" ] && got="open"
    if [ "$got" = "$2" ]; then
        echo "  PASS  $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $1 (expected $2, got $got)"
        FAIL=$((FAIL + 1))
    fi
}

reset() {
    rm -f "$MARK_TERM" "$WORK/lock"
    [ -f "$WORK/backend.pid" ] && kill "$(cat "$WORK/backend.pid")" 2>/dev/null
    rm -f "$WORK/backend.pid"
    sleep 0.3
}

use_marker() {  # switch the marker every case watches
    MARK_BROWSER="$WORK/opened-$1"
    rm -f "$MARK_BROWSER"
}

run_launcher() {
    # HOME is redirected for every case, not just the installer ones. The
    # launcher reads and WRITES under it — the browser profile lives at
    # $HOME/.config/brave-llamadeck, and the migration from the pre-rename name
    # moves a directory there. Run against the real home and a test run
    # rearranges the developer's own browser profile while their app window is
    # open. A suite must not touch the session it runs in.
    env HOME="$FAKE_HOME" \
        PATH="$STUB:$PATH" \
        LLAMADECK_PORT="${CASE_PORT:-$PORT}" \
        LLAMADECK_LOCKFILE="$WORK/lock" \
        LLAMADECK_START_LOG="$WORK/start.log" \
        LLAMADECK_TEST_MARKER="$MARK_BROWSER" \
        LLAMADECK_NOTIFY_CMD="$NOTIFY_CMD" \
        LLAMADECK_NOTIFY_ID_FILE="$WORK/notify-id" \
        timeout 40 bash "$LAUNCHER" "$@" >/dev/null 2>&1
}

echo "launcher: $LAUNCHER"
echo

# ── 1. slow path: backend down, terminal does not fork ───────────────
# The shipped bug: the outer launcher held the single-instance lock for the
# whole life of the terminal, so the launcher started INSIDE the terminal hit
# the lock and exited silently. Terminal opened, "opening Brave…" printed, no
# window ever appeared.
#
# The window must appear WHILE THE TERMINAL IS STILL OPEN. Waiting for the
# launcher to return instead lets a much later fallback open it and calls that
# a pass — but the user, sitting in front of an open terminal, has already
# concluded the app is broken. So: start it, wait a beat, assert, then let it
# finish.
reset
use_marker case1
run_launcher &
RUN_PID=$!
sleep 10
[ -s "$MARK_TERM" ] && echo "  (terminal was launched and is still open)" \
                    || echo "  (terminal NOT launched)"
check "slow path opens a window while the terminal is still open" open
kill $RUN_PID 2>/dev/null
wait $RUN_PID 2>/dev/null

# ── 2. a second click while the terminal is still open ───────────────
# Same root cause: with the lock held, every later click was a silent no-op.
# The backend from case 1 is still up, so this is the fast path.
use_marker case2
run_launcher
check "second click still opens/focuses a window" open

# ── 3. --open-only never starts a terminal ───────────────────────────
use_marker case3
rm -f "$MARK_TERM"
run_launcher --open-only
if [ -s "$MARK_TERM" ]; then
    echo "  FAIL  --open-only must not spawn a terminal"
    FAIL=$((FAIL + 1))
else
    echo "  PASS  --open-only does not spawn a terminal"
    PASS=$((PASS + 1))
fi
check "--open-only opens a window when the backend is up" open

# ── 4. lingering browser process, no wmctrl/xdotool ──────────────────
# The focus fast path used to return unconditionally on a pgrep match, and
# neither focus tool exists on a stock GNOME/Wayland box.
use_marker case4
setsid bash -c 'exec -a "brave --user-data-dir=/home/x/.config/br""ave-llamadeck" sleep 20' >/dev/null 2>&1 &
LINGER=$!
sleep 0.5
run_launcher
kill $LINGER 2>/dev/null
check "lingering browser process does not suppress the window" open

# ── 5. backend never comes up → must not open a dead URL ─────────────
reset
use_marker case5
CASE_PORT=18771 run_launcher --open-only
check "--open-only refuses to open a window with no backend" noop

# ── 6/7. the lock mechanism itself is unavailable ────────────────────
# flock(1) is util-linux: macOS, the BSDs and slim containers do not ship it,
# and some filesystems cannot lock. `if ! flock -n 9` read "command not found"
# (127) as "someone else holds the lock" and exited 0, so on those systems the
# icon did nothing, every time. Only status 1 means another launcher has it.
reset
"$WORK/proj/scripts/llamadeck-start.sh" >/dev/null 2>&1   # backend up: fast path only

use_marker case6
printf '#!/bin/sh\nexit 64\n' > "$STUB/flock"        # present but unusable
chmod +x "$STUB/flock"
run_launcher
check "a flock that fails for any reason other than 'held' still opens" open
rm -f "$STUB/flock"

# A PATH with everything the launcher needs EXCEPT flock — the macOS case.
use_marker case7
FARM="$WORK/nofl0ck"
mkdir -p "$FARM"
for d in /usr/bin /bin; do
    for f in "$d"/*; do
        n=$(basename "$f")
        [ "$n" = "flock" ] && continue
        [ -e "$FARM/$n" ] || ln -s "$f" "$FARM/$n" 2>/dev/null
    done
done
env PATH="$STUB:$FARM" LLAMADECK_PORT="$PORT" LLAMADECK_LOCKFILE="$WORK/lock" \
    LLAMADECK_TEST_MARKER="$MARK_BROWSER" \
    LLAMADECK_NOTIFY_CMD="$NOTIFY_CMD" LLAMADECK_NOTIFY_ID_FILE="$WORK/notify-id" \
    timeout 40 bash "$LAUNCHER" >/dev/null 2>&1
check "no flock installed at all still opens (macOS, BSD, slim containers)" open

# ── 8. an unsupported platform must say so, not fail quietly ─────────
# Every piece this script needs is missing on macOS, and each one failed
# silently, so the app looked broken rather than unsupported.
use_marker case8
printf '#!/bin/sh\necho Darwin\n' > "$STUB/uname"
chmod +x "$STUB/uname"
GUARD_OUT="$WORK/guard.txt"
rm -f "$NOTIFY_LOG" "$WORK/notify-id"
env PATH="$STUB:$PATH" LLAMADECK_PORT="$PORT" LLAMADECK_LOCKFILE="$WORK/lock" \
    LLAMADECK_TEST_MARKER="$MARK_BROWSER" \
    LLAMADECK_NOTIFY_CMD="$NOTIFY_CMD" LLAMADECK_NOTIFY_ID_FILE="$WORK/notify-id" \
    timeout 20 bash "$LAUNCHER" >"$GUARD_OUT" 2>&1
GUARD_RC=$?

if [ "$GUARD_RC" -ne 0 ] && grep -qi "macos" "$GUARD_OUT"; then
    echo "  PASS  macOS exits non-zero and explains why"
    PASS=$((PASS + 1))
else
    echo "  FAIL  macOS should exit non-zero with an explanation (rc=$GUARD_RC)"
    FAIL=$((FAIL + 1))
fi
if grep -q "llamadeck serve" "$GUARD_OUT"; then
    echo "  PASS  the message names the command that does work"
    PASS=$((PASS + 1))
else
    echo "  FAIL  the message must name a command that works"
    FAIL=$((FAIL + 1))
fi
check "macOS opens no window" noop

# stderr is nowhere when the launcher was started by a .desktop entry, so the
# banner is the entire user-visible half of this guard.
if [ -s "$NOTIFY_LOG" ]; then
    echo "  PASS  the guard reaches the desktop, not only stderr"
    PASS=$((PASS + 1))
else
    echo "  FAIL  no desktop notification — no window and no banner is the silent failure this suite exists to catch"
    FAIL=$((FAIL + 1))
fi

# ── 9. a repeated failure replaces its banner, it does not stack ─────
# -u critical never expires, so one banner per invocation meant a click-happy
# user (and every run of this file) buried the top of the screen in identical
# "LlamaDeck" notifications that had to be dismissed by hand.
env PATH="$STUB:$PATH" LLAMADECK_PORT="$PORT" LLAMADECK_LOCKFILE="$WORK/lock" \
    LLAMADECK_TEST_MARKER="$MARK_BROWSER" \
    LLAMADECK_NOTIFY_CMD="$NOTIFY_CMD" LLAMADECK_NOTIFY_ID_FILE="$WORK/notify-id" \
    timeout 20 bash "$LAUNCHER" >/dev/null 2>&1
rm -f "$STUB/uname"

if [ "$(grep -c . "$NOTIFY_LOG" 2>/dev/null)" = 2 ] \
   && grep -q -- "-r 4242" "$NOTIFY_LOG"; then
    echo "  PASS  a second failure replaces the banner instead of stacking"
    PASS=$((PASS + 1))
else
    echo "  FAIL  second failure did not reuse the first banner's id:"
    sed 's/^/          /' "$NOTIFY_LOG" 2>/dev/null
    FAIL=$((FAIL + 1))
fi

# ── 10. the window identity the desktop entry has to match ───────────
# The shipped bug: StartupWMClass said LlamaDeck, because --class=LlamaDeck is passed. An
# --app= window ignores --class and is named after its URL, so on Wayland GNOME
# matched the window to nothing — a generic cog in the dock, and a brand new
# window on every click because the shell could not see the app was running.
#
# Nothing here can talk to a compositor, so the assertion is the one that
# actually broke: whatever the launcher opens, the installed entry must claim
# the same class, and it must not be the value that matched nothing.
use_marker case10
rm -f "$MARK_TERM"
CLASS="$(env PATH="$STUB:$PATH" bash "$LAUNCHER" --print-wm-class 2>/dev/null)"
if [ -n "$MARK_BROWSER" ] && [ -s "$MARK_BROWSER" ] || [ -s "$MARK_TERM" ]; then
    echo "  FAIL  --print-wm-class must answer and exit, not launch anything"
    FAIL=$((FAIL + 1))
else
    echo "  PASS  --print-wm-class launches nothing"
    PASS=$((PASS + 1))
fi

case "$CLASS" in
    brave-127.0.0.1__-Default)
        echo "  PASS  class follows the browser it would open ($CLASS)"
        PASS=$((PASS + 1)) ;;
    *)
        echo "  FAIL  expected brave-127.0.0.1__-Default from the stubbed brave, got '$CLASS'"
        FAIL=$((FAIL + 1)) ;;
esac

# A different browser has a different product prefix, and the entry has to
# follow it — a hardcoded "brave-" would match nothing on a Chromium box.
# An isolated PATH, not $STUB plus /usr/bin: the developer's real Brave lives
# there and is checked first, so a "chromium-only box" that can still see it
# proves nothing. Only what the launcher needs before it answers is linked in.
mkdir -p "$WORK/chromium-only"
cp "$STUB/brave-browser" "$WORK/chromium-only/chromium"
for tool in dirname; do
    ln -sf "$(command -v "$tool")" "$WORK/chromium-only/$tool"
done
# bash by absolute path — `env` resolves the command through the PATH it is
# setting, and bash does not live in the isolated directory.
CLASS_CHROMIUM="$(env PATH="$WORK/chromium-only" \
    "$BASH" "$LAUNCHER" --print-wm-class 2>/dev/null)"
if [ "$CLASS_CHROMIUM" = "chromium-127.0.0.1__-Default" ]; then
    echo "  PASS  a Chromium-only box gets its own class ($CLASS_CHROMIUM)"
    PASS=$((PASS + 1))
else
    echo "  FAIL  chromium-only box got '$CLASS_CHROMIUM'"
    FAIL=$((FAIL + 1))
fi

# ── 11. the installer stamps that class into the entry ───────────────
# The template ships @WMCLASS@; an unsubstituted placeholder matches nothing,
# which is the same broken dock the whole case exists to prevent.
if env HOME="$FAKE_HOME" PATH="$STUB:$PATH" \
       bash "$HERE/install-desktop.sh" >"$WORK/install.log" 2>&1; then
    ENTRY="$FAKE_HOME/.local/share/applications/llamadeck.desktop"
    GOT="$(sed -n 's/^StartupWMClass=//p' "$ENTRY" 2>/dev/null)"
    if [ "$GOT" = "$CLASS" ] && [ -n "$GOT" ]; then
        echo "  PASS  installed entry claims the window the launcher opens ($GOT)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  StartupWMClass is '$GOT', launcher opens '$CLASS'"
        FAIL=$((FAIL + 1))
    fi
    if grep -q '@' "$ENTRY" 2>/dev/null; then
        echo "  FAIL  unsubstituted placeholder left in the entry:"
        grep '@' "$ENTRY" | sed 's/^/          /'
        FAIL=$((FAIL + 1))
    else
        echo "  PASS  no placeholder survives into the installed entry"
        PASS=$((PASS + 1))
    fi
else
    echo "  FAIL  install-desktop.sh exited non-zero:"
    sed 's/^/          /' "$WORK/install.log"
    FAIL=$((FAIL + 1))
fi

# ── 12. carrying the pre-rename browser profile across ───────────────
# The app window lives in its own browser profile, which used to be named
# after the project's old name. Two things have to hold, and the second one
# was learned the hard way — an unguarded version of this moved a profile out
# from under a running Chromium.
assert_profile() {  # $1=label  $2=expected dir under FAKE_HOME/.config
    if [ -d "$FAKE_HOME/.config/$2" ]; then
        echo "  PASS  $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $1 (no $FAKE_HOME/.config/$2; have: $(ls "$FAKE_HOME/.config" 2>/dev/null | tr '\n' ' '))"
        FAIL=$((FAIL + 1))
    fi
}

reset
use_marker case12
rm -rf "$FAKE_HOME/.config/brave-lsc" "$FAKE_HOME/.config/brave-llamadeck"
mkdir -p "$FAKE_HOME/.config/brave-lsc/Default"
echo marker > "$FAKE_HOME/.config/brave-lsc/Default/Preferences.marker"
CASE_PORT=$PORT run_launcher --print-wm-class
assert_profile "an unused pre-rename browser profile is carried over" brave-llamadeck
if [ -f "$FAKE_HOME/.config/brave-llamadeck/Default/Preferences.marker" ]; then
    echo "  PASS  the carried-over profile keeps its contents"
    PASS=$((PASS + 1))
else
    echo "  FAIL  the carried-over profile lost its contents"
    FAIL=$((FAIL + 1))
fi

# A live Chromium holds SingletonLock -> "<host>-<pid>". Renaming the directory
# under it corrupts its session, and pointing this run at a fresh directory
# opens a SECOND window instead of raising the one already on screen.
reset
use_marker case13
rm -rf "$FAKE_HOME/.config/brave-lsc" "$FAKE_HOME/.config/brave-llamadeck"
mkdir -p "$FAKE_HOME/.config/brave-lsc"
sleep 60 &
HOLDER=$!
ln -s "$(hostname)-$HOLDER" "$FAKE_HOME/.config/brave-lsc/SingletonLock"
CASE_PORT=$PORT run_launcher --print-wm-class
if [ -d "$FAKE_HOME/.config/brave-lsc" ] && [ ! -e "$FAKE_HOME/.config/brave-llamadeck" ]; then
    echo "  PASS  a profile a browser is still using is left where it is"
    PASS=$((PASS + 1))
else
    echo "  FAIL  the launcher moved a profile out from under a running browser"
    FAIL=$((FAIL + 1))
fi
kill $HOLDER 2>/dev/null
wait $HOLDER 2>/dev/null

# ...and once that browser is gone, the next launch does carry it over.
CASE_PORT=$PORT run_launcher --print-wm-class
assert_profile "the move happens on the next launch, once the browser is gone" brave-llamadeck

echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
