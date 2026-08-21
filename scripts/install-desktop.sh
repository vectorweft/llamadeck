#!/usr/bin/env bash
# Install LlamaDeck as a desktop app (Ubuntu / GNOME + any XDG-compliant DE).
# Idempotent: re-running just re-links + refreshes caches.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .desktop entries are XDG. On macOS this used to write the file anyway and
# then print "find it via Activities", which is simply untrue there.
case "$(uname -s 2>/dev/null)" in
    Linux|*BSD|GNU*|"") ;;
    *)
        echo "This installer creates an XDG .desktop entry, which $(uname -s) does not use." >&2
        echo "LlamaDeck itself runs here — start it with:" >&2
        echo "    cd $PROJECT_DIR && uv run llamadeck serve" >&2
        echo "then open http://127.0.0.1:8770/ in a browser." >&2
        exit 1 ;;
esac
APPS_DIR="$HOME/.local/share/applications"
ICONS_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"

mkdir -p "$APPS_DIR" "$ICONS_DIR"

# ── 1. .desktop file (template → real values for this clone) ─────────
# StartupWMClass is what makes GNOME tie the browser window to this entry, and
# getting it wrong is not cosmetic: an unmatched window means a generic cog in
# the dock AND a fresh window on every click, because the shell cannot tell the
# app is already running. The launcher derives it, since only it knows which
# browser it will use. Run through `bash` — the executable bit is not set until
# step 3.
WMCLASS="$(bash "$PROJECT_DIR/scripts/llamadeck-launcher.sh" --print-wm-class 2>/dev/null)"
case "$WMCLASS" in
    ''|*[!A-Za-z0-9._-]*) WMCLASS="brave-127.0.0.1__-Default" ;;
esac
sed -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
    -e "s|@WMCLASS@|$WMCLASS|g" "$PROJECT_DIR/scripts/llamadeck.desktop" \
    > "$APPS_DIR/llamadeck.desktop"
chmod 644 "$APPS_DIR/llamadeck.desktop"
echo "[1/4] installed $APPS_DIR/llamadeck.desktop (StartupWMClass=$WMCLASS)"

# ── 2. icon (SVG, scalable) ───────────────────────────────────────────
cp "$PROJECT_DIR/assets/llamadeck-icon-wordmark.svg" "$ICONS_DIR/llamadeck.svg"
echo "[2/4] installed $ICONS_DIR/llamadeck.svg"

# ── 3. launcher + start scripts executable ────────────────────────────
chmod +x "$PROJECT_DIR/scripts/llamadeck-launcher.sh" \
         "$PROJECT_DIR/scripts/llamadeck-start.sh"
echo "[3/4] marked launcher scripts executable"

# ── 4. refresh caches so GNOME picks it up immediately ────────────────
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache >/dev/null; then
    gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi
echo "[4/4] refreshed desktop + icon caches"

# ── the pre-rename entry, if this machine has one ────────────────────
# Leaving it installed means two icons in the launcher, one of them pointing
# at scripts/lsc-launcher.sh — a path that no longer exists in this clone, so
# clicking it does nothing at all. That is the exact failure mode this
# launcher has been hardened against; do not reintroduce it by omission.
LEGACY_DESKTOP="$APPS_DIR/lsc.desktop"
LEGACY_ICON="$ICONS_DIR/lsc.svg"
if [ -e "$LEGACY_DESKTOP" ] || [ -e "$LEGACY_ICON" ]; then
    rm -f "$LEGACY_DESKTOP" "$LEGACY_ICON"
    command -v update-desktop-database >/dev/null && \
        update-desktop-database "$APPS_DIR" 2>/dev/null || true
    echo "      removed the old LSC entry ($LEGACY_DESKTOP)"
fi

echo ""
echo "✓ LlamaDeck installed. Find it via:"
echo "    Activities (Super key) → search 'LlamaDeck'"
echo ""
echo "  First launch auto-starts the backend in a terminal."
echo "  Subsequent launches just focus the existing app window (fast)."
echo ""
echo "  To uninstall:  rm $APPS_DIR/llamadeck.desktop $ICONS_DIR/llamadeck.svg"
