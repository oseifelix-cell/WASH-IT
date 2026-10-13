#!/usr/bin/env bash
# pi_kiosk_launch.sh — Robust kiosk launcher.
#
# Replaces the old fixed "sleep 8 then launch Chromium" approach, which
# raced the desktop session and the backend on a cold boot -- sometimes
# 8s was enough, sometimes not, causing the intermittent white-screen
# hang that needed several reboots to clear.
#
# This actively WAITS for each real precondition instead of guessing:
#   1. The X11 desktop session is actually accepting commands
#   2. The backend's WebSocket port is actually listening (not just that
#      the systemd service "started" -- Type=simple marks a service
#      active the instant the process launches, well before the model
#      and camera have finished loading, which takes 10-15+ seconds)
#   3. A completely fresh Chromium profile every boot, so a stale/corrupt
#      profile from an earlier improper shutdown can never cause a
#      persistent blank-screen hang.

set -u
USER_NAME="$(whoami)"
PROJECT_DIR="/home/$USER_NAME/washit"
PROFILE_DIR="/home/$USER_NAME/.config/chromium-kiosk"
HTML_FILE="$PROJECT_DIR/washit_v7_live.html"
BACKEND_PORT=8000
LOG_PREFIX="[kiosk-launch]"

echo "$LOG_PREFIX Waiting for the X11 desktop session..."
X_READY=0
for i in $(seq 1 60); do
    if xset q > /dev/null 2>&1; then
        X_READY=1
        echo "$LOG_PREFIX X11 ready after ${i}s"
        break
    fi
    sleep 1
done
if [ "$X_READY" -eq 0 ]; then
    echo "$LOG_PREFIX WARNING: X11 never confirmed ready after 60s -- proceeding anyway"
fi

# Disable screen blanking now that X is confirmed up
xset s off        2>/dev/null
xset -dpms        2>/dev/null
xset s noblank     2>/dev/null

echo "$LOG_PREFIX Waiting for the backend (port $BACKEND_PORT)..."
BACKEND_READY=0
for i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/$BACKEND_PORT) 2>/dev/null; then
        exec 3<&- 3>&-
        BACKEND_READY=1
        echo "$LOG_PREFIX Backend ready after ${i}s"
        break
    fi
    sleep 1
done
if [ "$BACKEND_READY" -eq 0 ]; then
    echo "$LOG_PREFIX WARNING: backend never confirmed listening after 120s -- "
    echo "$LOG_PREFIX launching the kiosk anyway (it will retry the connection itself)"
fi

echo "$LOG_PREFIX Wiping Chromium profile for a clean start..."
rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

echo "$LOG_PREFIX Launching Chromium..."
exec /usr/bin/chromium-browser \
    --user-data-dir="$PROFILE_DIR" \
    --kiosk \
    --start-fullscreen \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-translate \
    --check-for-update-interval=31536000 \
    "file://$HTML_FILE"
