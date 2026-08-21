#!/usr/bin/env bash
# LlamaDeck desktop launcher — invoked from llamadeck.desktop.
#
# Logic:
#   1. Backend up → focus or open Brave in app mode
#   2. Down → open gnome-terminal that runs llamadeck-start.sh, then once
#             /health responds, open Brave as app
#
# Window identity (this is what makes the dock icon right, and what makes a
# second click focus the window instead of opening another one):
#   An --app= window does NOT take its class from --class. Chromium derives it
#   from the URL instead, so on Wayland GNOME sees `brave-127.0.0.1__-Default`
#   and matched it to no .desktop file at all — hence a generic cog in the
#   dock, and every click on the launcher starting a fresh window because the
#   shell had no idea the app was already running. `app_wm_class` below
#   derives that string, install-desktop.sh writes it into StartupWMClass, and
#   the X11 path still matches through --class=LlamaDeck → llamadeck.desktop.

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRAVE_PROFILE="$HOME/.config/brave-llamadeck"
LEGACY_PROFILE="$HOME/.config/brave-lsc"

# Is a Chromium still using this profile directory? It keeps SingletonLock as a
# symlink to "<host>-<pid>", so the answer is a readlink and a kill -0 — no
# pgrep, which on this box matches the very shell doing the asking.
profile_in_use() {
    local target pid
    target="$(readlink "$1/SingletonLock" 2>/dev/null)" || return 1
    pid="${target##*-}"
    case "$pid" in ''|*[!0-9]*) return 1 ;; esac
    kill -0 "$pid" 2>/dev/null
}

# The app window runs in its own browser profile, which used to be named after
# the project's old name. Carry it over rather than starting people on a blank
# profile — it holds the window geometry, the zoom level and anything the
# dashboard logged into.
#
# Not unconditionally, though: the window from before the rename may still be
# open on that directory right now. Renaming it out from under a running
# Chromium corrupts its session, and pointing this run at the new empty
# directory opens a SECOND window instead of raising the one already there. So
# while it is in use, keep using it, and migrate on a later launch.
if [ ! -e "$BRAVE_PROFILE" ] && [ -d "$LEGACY_PROFILE" ]; then
    if profile_in_use "$LEGACY_PROFILE"; then
        BRAVE_PROFILE="$LEGACY_PROFILE"
    elif ! mv "$LEGACY_PROFILE" "$BRAVE_PROFILE" 2>/dev/null; then
        BRAVE_PROFILE="$LEGACY_PROFILE"
    fi
fi

# The port is a setting (controller_bind_port), not a constant. Hardcoding 8770
# meant the icon opened a dead URL for anyone who changed it. Read the setting,
# fall back to the default, and let the environment override for tests.
STATE_DIR="${LLAMADECK_STATE_DIR:-$HOME/.config/llamadeck}"
if [ -z "${LLAMADECK_PORT:-}" ] && [ -r "$STATE_DIR/settings.json" ] && command -v python3 >/dev/null; then
    LLAMADECK_PORT="$(python3 -c "
import json,sys
try:
    print(json.load(open(sys.argv[1])).get('controller_bind_port') or '')
except Exception:
    pass
" "$STATE_DIR/settings.json" 2>/dev/null)"
fi
case "${LLAMADECK_PORT:-}" in
    ''|*[!0-9]*) LLAMADECK_PORT=8770 ;;
esac

HEALTH_URL="http://127.0.0.1:$LLAMADECK_PORT/health"
APP_URL="http://127.0.0.1:$LLAMADECK_PORT/"
LOCKFILE="${LLAMADECK_LOCKFILE:-/tmp/llamadeck-launcher.lock}"
START_LOG="${LLAMADECK_START_LOG:-/tmp/llamadeck-launcher-start.log}"

# A stale :1 here sent the browser to a display that does not exist on a
# Wayland session (this box runs :0). Only fall back when nothing is set.
export DISPLAY="${DISPLAY:-:0}"

backend_ok() { curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; }

# A launcher run from a .desktop entry has nowhere to print. Anything that
# stops the app from opening has to reach the user through the desktop itself,
# otherwise the icon just appears to do nothing.
#
# The notifier is a variable, not a hardcoded name, for one reason: the slow
# path runs the nested launcher under `bash -l`, and a login shell rewrites
# PATH from /etc/profile and ~/.profile. A test that stubs notify-send by
# putting a fake one first on PATH therefore does NOT hold across that call,
# and the suite posted critical banners to the developer's own desktop. An
# environment variable survives the rewrite; PATH does not.
NOTIFY_BIN="${LLAMADECK_NOTIFY_CMD:-notify-send}"

# Where the id of the last banner is remembered, so a repeat replaces it. Per
# session, because notification ids are only meaningful to the running server.
NOTIFY_ID_FILE="${LLAMADECK_NOTIFY_ID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/llamadeck-launcher-notify-id}"

report_failure() {
    local msg="$1"
    if command -v "$NOTIFY_BIN" >/dev/null; then
        # -u critical is deliberate: per the desktop notification spec a
        # critical banner never expires, and "you clicked the icon and no
        # window is coming" is exactly the message that must not disappear
        # before it is read.
        #
        # But never-expiring plus one-per-invocation stacks. Every icon click
        # on a broken box, and every run of the test suite, pinned another
        # copy at the top of the screen until they were dismissed by hand.
        # --print-id/--replace-id makes a repeat overwrite the previous banner
        # instead of adding to the pile: still impossible to miss, still one.
        local prev=0 id
        [ -r "$NOTIFY_ID_FILE" ] && read -r prev <"$NOTIFY_ID_FILE" 2>/dev/null
        case "$prev" in ''|*[!0-9]*) prev=0 ;; esac

        if id="$("$NOTIFY_BIN" -u critical -p -r "$prev" \
                     -h string:desktop-entry:llamadeck \
                     "LlamaDeck" "$msg" 2>/dev/null)"; then
            case "$id" in
                ''|*[!0-9]*) : ;;
                *) printf '%s\n' "$id" >"$NOTIFY_ID_FILE" 2>/dev/null || true ;;
            esac
        else
            # -p/-r predate nothing in libnotify, but a minimal reimplementation
            # may not have them. A stacking banner beats a silent failure.
            "$NOTIFY_BIN" -u critical "LlamaDeck" "$msg" 2>/dev/null || true
        fi
    elif command -v zenity >/dev/null; then
        zenity --error --title="LlamaDeck" --text="$msg" 2>/dev/null || true
    elif command -v osascript >/dev/null; then
        osascript -e "display notification \"$msg\" with title \"LlamaDeck\"" \
            2>/dev/null || true
    fi
    echo "[launcher] $msg" >&2
}

# ── which browser, and the window identity that follows from it ──────
# Any Chromium-family browser will do — they all support --app= (a window with
# no tabs or address bar, which is what makes this feel like a desktop app
# rather than a browser tab). Brave first because that is what this was
# written against; the rest keep the launcher working on a machine that never
# installed it.
pick_browser() {
    local cand
    for cand in brave-browser brave brave-browser-stable \
                chromium chromium-browser google-chrome google-chrome-stable \
                microsoft-edge microsoft-edge-stable; do
        if command -v "$cand" >/dev/null; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

# The WM class (X11) / app_id (Wayland) the browser gives an --app= window.
#
# GNOME matches this against StartupWMClass to decide the window belongs to
# llamadeck.desktop. Everything the user notices hangs off that one match: the dock
# shows the app's icon rather than a generic cog, and clicking the icon while
# the app is open focuses that window instead of launching a second copy.
#
# --class does not decide it. Chromium honours --class for ordinary browser
# windows, but an --app= window is named after its URL. Measured here with
# WAYLAND_DEBUG=1, for `--app=http://127.0.0.1:8770/ --class=LlamaDeck`:
#
#     set_app_id("brave-127.0.0.1__-Default")
#
# Note what is missing from it: the port. Chromium builds the name from host
# and path only, so this string does not move when controller_bind_port does.
#
# The prefix is the browser's product name, which is why this lives next to
# pick_browser rather than in the installer — the entry and the window it has
# to match must never be derived from two different browsers.
app_wm_class() {
    local bin product
    bin="$(pick_browser || true)"
    case "$bin" in
        brave*)          product=brave ;;
        chromium*)       product=chromium ;;
        google-chrome*)  product=chrome ;;
        microsoft-edge*) product=msedge ;;
        vivaldi*)        product=vivaldi ;;
        # No Chromium anywhere: open_brave falls back to an ordinary xdg-open
        # tab, which no StartupWMClass can match. Name the default rather than
        # printing nothing, so the installer still writes a usable entry.
        "")              product=brave ;;
        *)               product="${bin%%-*}" ;;
    esac
    printf '%s-127.0.0.1__-Default\n' "$product"
}

# Asked by install-desktop.sh, which cannot guess this on its own. Answered
# before the platform guard because it is a pure string: the installer has its
# own, better-worded guard, and a notification about macOS is not an answer to
# a question about window classes.
if [ "${1:-}" = "--print-wm-class" ]; then
    app_wm_class
    exit 0
fi

# ── platform guard ───────────────────────────────────────────────────
# This script is XDG desktop integration and nothing else: it wants a Chromium
# binary on PATH, xdg-open, one of the terminals listed further down, and a
# .desktop entry that install-desktop.sh put in place. None of that exists on
# macOS or Windows.
#
# LlamaDeck itself runs fine there — accel.py builds a Metal backend, vram.py
# budgets Apple unified memory, and the platform tests cover both. Only the
# icon does not. Saying so is the whole point: an unsupported platform used to
# look exactly like a broken app, because every one of those missing pieces
# failed quietly.
case "$(uname -s 2>/dev/null)" in
    Linux|*BSD|GNU*|"") ;;
    Darwin)
        report_failure "This launcher is Linux/XDG only, so the icon cannot work on macOS. \
LlamaDeck itself does run here — start it with:  cd $PROJECT_DIR && uv run llamadeck serve \
 — then open $APP_URL in any browser."
        exit 1 ;;
    *)
        report_failure "This launcher supports Linux desktops only ($(uname -s) detected). \
Run 'uv run llamadeck serve' in $PROJECT_DIR and open $APP_URL in a browser."
        exit 1 ;;
esac

open_brave() {
    # Already running → focus the existing window rather than opening a second.
    # This is a best-effort fast path ONLY: it must never be the sole attempt.
    # wmctrl and xdotool are both absent on a default GNOME/Wayland install, and
    # a lingering Brave process with no window still matches the pgrep — so the
    # old "matched → return" made the icon do nothing at all. Falling through to
    # the launch below is safe: re-invoking Chromium with the same
    # --user-data-dir hands the request to the running instance, which raises
    # its window instead of starting a second copy.
    if pgrep -f "$(basename "$BRAVE_PROFILE")" >/dev/null 2>&1; then
        if command -v wmctrl >/dev/null; then
            if wmctrl -xa "LlamaDeck.LlamaDeck" 2>/dev/null \
                || wmctrl -a "LlamaDeck" 2>/dev/null \
                || wmctrl -a "127.0.0.1:$LLAMADECK_PORT" 2>/dev/null; then
                return 0
            fi
        elif command -v xdotool >/dev/null; then
            if xdotool search --class LlamaDeck windowactivate 2>/dev/null; then
                return 0
            fi
        fi
    fi

    local brave_bin=""
    brave_bin="$(pick_browser || true)"
    if [ -z "$brave_bin" ]; then
        # No Chromium anywhere (a Firefox-only box). Hand off to the default
        # browser: an ordinary tab, but the app still opens instead of failing.
        xdg-open "$APP_URL" >/dev/null 2>&1 9>&- &
        disown
        return 0
    fi

    # Wipe stored session/tab state so launcher always opens at APP_URL only
    local default_dir="$BRAVE_PROFILE/Default"
    if [ -d "$default_dir" ]; then
        rm -f "$default_dir/Last Tabs" \
              "$default_dir/Last Session" \
              "$default_dir/Current Tabs" \
              "$default_dir/Current Session" \
              2>/dev/null || true
        if [ -f "$default_dir/Preferences" ]; then
            python3 - "$default_dir/Preferences" <<'PY' 2>/dev/null || true
import json, sys
p = sys.argv[1]
try:
    with open(p) as fh: d = json.load(fh)
except Exception:
    sys.exit(0)
(d.setdefault("profile", {}))["exit_type"] = "Normal"
(d.setdefault("profile", {}))["exited_cleanly"] = True
with open(p, "w") as fh: json.dump(d, fh)
PY
        fi
    fi

    # Ozone backend: native wayland on a Wayland session, x11 everywhere else.
    #
    # x11 used to be the unconditional default, for the dock icon: Ozone's
    # Wayland backend may ignore --class, and without it GNOME cannot match the
    # window to the .desktop entry. But on a Wayland session with FRACTIONAL
    # scaling that costs sharpness, and it is not subtle. Measured here on a
    # 2560x1440 panel at GNOME scale 1.25:
    #
    #   XWayland screen for that output ... 3072x1728  (logical 2048x1152 x1.5)
    #   Xft.dpi 96 -> browser device scale  1.0
    #   so Brave renders 3072x1728 real pixels, and the compositor squeezes
    #   them into the panel's 2560x1440 — a 0.83x resample of the whole
    #   window. Every glyph goes soft, and the UI lands at 83% of its
    #   intended size, which is why the in-app zoom gets reached for.
    #
    # On the Wayland backend the browser is told the 1.25 scale directly, draws
    # 2560x1440 device pixels and is presented 1:1. No resample, no blur.
    #
    # Force the old behaviour (e.g. if the dock icon matters more than
    # sharpness on an unscaled display) with:
    #   LLAMADECK_OZONE_PLATFORM=x11 scripts/llamadeck-launcher.sh
    #
    # Wayland was once suspected of causing desktop-wide cursor stutter here.
    # It does not: switching back changed nothing, and the cause turned out to
    # be the compositor, which is a separate problem from this window.
    local ozone="${LLAMADECK_OZONE_PLATFORM:-}"
    if [ -z "$ozone" ]; then
        if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
            ozone=wayland
        else
            ozone=x11
        fi
    fi

    "$brave_bin" \
        --app="$APP_URL" \
        --class=LlamaDeck \
        --name=llamadeck \
        --ozone-platform="$ozone" \
        --user-data-dir="$BRAVE_PROFILE" \
        --no-first-run --no-default-browser-check \
        --disable-session-crashed-bubble \
        --hide-crash-restore-bubble \
        --restore-last-session=false \
        --disable-features=InfiniteSessionRestore,Translate \
        --disable-infobars \
        >/dev/null 2>&1 9>&- &
    disown
}

# ── --open-only: browser only, no lock, cannot start a terminal ──────
# Used by the terminal the slow path opens, once llamadeck-start.sh has returned. It
# must NOT take the single-instance lock — the launcher that opened that
# terminal is still holding it — and must not be able to re-enter the slow
# path, or a failed health check would spawn a second terminal, then a third.
if [ "${1:-}" = "--open-only" ]; then
    for _ in $(seq 1 60); do
        backend_ok && break
        sleep 1
    done
    if ! backend_ok; then
        report_failure "backend never became healthy on :$LLAMADECK_PORT — not opening a window"
        exit 1
    fi
    open_brave
    exit 0
fi

# ── single-instance guard ────────────────────────────────────────────
# Guards the DECISION below, nothing more. Two rules keep it from becoming the
# thing that breaks the app:
#   1. Every process spawned below closes fd 9 (`9>&-`). The lock lives on the
#      fd, so anything that inherits it holds the lock for its whole lifetime —
#      a backend started here kept it forever.
#   2. It is released before the slow path blocks on a terminal. ptyxis stays
#      in the foreground (gnome-terminal, which this was written against, forks
#      and returns), so holding it across that call made every icon click a
#      silent no-op for as long as the terminal window stayed open.
#   3. It is optional. flock(1) is util-linux: macOS, the BSDs and slim
#      containers do not have it, and `if ! flock -n 9` treats "command not
#      found" (127) exactly like "someone else holds the lock" — so the icon
#      did nothing at all there, forever. Guarding against two near-
#      simultaneous clicks is worth having; it is not worth never opening.
HAVE_LOCK=0
if command -v flock >/dev/null 2>&1 && (exec 9>"$LOCKFILE") 2>/dev/null; then
    exec 9>"$LOCKFILE"
    flock -n 9
    case $? in
        0) HAVE_LOCK=1 ;;
        1) exit 0 ;;   # another launcher genuinely holds it: nothing to do
        *) : ;;        # flock unusable here (no util-linux, a filesystem that
                       # cannot lock) — carry on unguarded rather than never
                       # opening. Only status 1 means "someone else has it".
    esac
fi

# ── fast path: backend up → just open Brave ──────────────────────────
if backend_ok; then
    open_brave
    exit 0
fi

# ── slow path: start service in a terminal, then open Brave ──────────
unset GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR GDK_BACKEND
unset GTK_EXE_PREFIX GTK_PATH GTK_IM_MODULE_FILE
unset GIO_MODULE_DIR GIO_LAUNCHED_DESKTOP_FILE GIO_LAUNCHED_DESKTOP_FILE_PID
unset GSETTINGS_SCHEMA_DIR BAMF_DESKTOP_FILE_HINT
unset LD_LIBRARY_PATH LD_PRELOAD
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
unset SNAP SNAP_REVISION SNAP_REAL_HOME SNAP_USER_DATA SNAP_USER_COMMON
unset SNAP_INSTANCE_NAME SNAP_ARCH SNAP_CONTEXT SNAP_EUID SNAP_UID
unset SNAP_LAUNCHER_ARCH_TRIPLET
export XDG_CONFIG_DIRS="/etc/xdg/xdg-ubuntu:/etc/xdg"
export XDG_DATA_DIRS="/usr/share/ubuntu:/usr/local/share:/usr/share:/var/lib/snapd/desktop"
export XDG_DATA_HOME="$HOME/.local/share"

# The three absolute paths this used to check are all absent on a stock Ubuntu
# 26.04 GNOME install (ptyxis replaced gnome-terminal), which silently pushed
# every launch onto the no-terminal branch below. Search by name, newest
# defaults first, and remember that not every terminal takes --title.
TERMINAL_CMD=""
for cand in ptyxis gnome-terminal kgx konsole xfce4-terminal mate-terminal \
            tilix terminator alacritty kitty foot wezterm xterm; do
    if command -v "$cand" >/dev/null; then
        TERMINAL_CMD="$(command -v "$cand")"
        break
    fi
done

INNER_CMD="'$PROJECT_DIR/scripts/llamadeck-start.sh'; echo ''; echo '[launcher] opening Brave…'; ('$PROJECT_DIR/scripts/llamadeck-launcher.sh' --open-only &); echo '[launcher] Terminal stays open for service logs. Press Enter to close.'; read"

# Release the single-instance lock BEFORE the terminal call. Terminals that do
# not fork (ptyxis) block here for the life of the window, and the launcher we
# are about to run inside it — plus every later icon click — would find the
# lock still held and exit without a word. llamadeck-start.sh is idempotent, so the
# worst case after releasing is a second harmless health check.
if [ "$HAVE_LOCK" = "1" ]; then
    flock -u 9
    exec 9>&-
fi

if [ -n "$TERMINAL_CMD" ]; then
    case "$(basename "$TERMINAL_CMD")" in
        ptyxis)
            # ptyxis has no --title and needs --new-window before --.
            "$TERMINAL_CMD" --new-window -- bash -l -c "$INNER_CMD" 9>&- ;;
        xterm|alacritty|kitty|foot|wezterm)
            "$TERMINAL_CMD" -e bash -l -c "$INNER_CMD" 9>&- ;;
        *)
            "$TERMINAL_CMD" --title="LlamaDeck services" -- bash -l -c "$INNER_CMD" 9>&- ;;
    esac
    if [ $? -ne 0 ]; then
        # Wrong flags for this terminal, or it refused to start. Fall back to
        # starting in-process so a terminal quirk can't block the whole app.
        report_failure "terminal '$TERMINAL_CMD' failed — starting backend without it"
        if ! bash "$PROJECT_DIR/scripts/llamadeck-start.sh" 9>&- >"$START_LOG" 2>&1; then
            report_failure "LlamaDeck could not start its backend. See $START_LOG"
            exit 1
        fi
        open_brave
    fi
else
    # No terminal anywhere. Start in-process — but never fail silently, or the
    # icon click looks like a no-op with nothing to read anywhere.
    if ! bash "$PROJECT_DIR/scripts/llamadeck-start.sh" 9>&- >"$START_LOG" 2>&1; then
        report_failure "LlamaDeck could not start its backend. See $START_LOG"
        exit 1
    fi
    open_brave
fi
