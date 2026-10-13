#!/usr/bin/env bash
# pi_kiosk_launch_firefox.sh — Firefox ESR kiosk launcher.
#
# Same wait-for-readiness logic as pi_kiosk_launch.sh (X11, then the
# backend), just launching Firefox ESR instead of Chromium -- an A/B test
# against Chromium's suspected GPU/renderer instability on the Pi's
# vc4-kms-v3d driver stack (the likely cause of the white-screen hangs and
# possibly the unexplained mid-session WebSocket resets).
#
# Kept as a SEPARATE script from pi_kiosk_launch.sh so switching back to
# Chromium is a one-line change, not a rewrite, if this doesn't help.

set -u
USER_NAME="$(whoami)"
PROJECT_DIR="/home/$USER_NAME/washit"
PROFILE_DIR="/home/$USER_NAME/.mozilla-kiosk-profile"
HTML_FILE="$PROJECT_DIR/washit_v7_live.html"
BACKEND_PORT=8000
LOG_PREFIX="[kiosk-launch-firefox]"

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
    echo "$LOG_PREFIX launching the kiosk anyway"
fi

echo "$LOG_PREFIX Wiping Firefox profile for a clean start..."
rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

# Kiosk-appropriate prefs: no crash-restore prompt, no default-browser
# nag, no update checks -- written before first launch so Firefox never
# shows a single interactive dialog on this device.
cat > "$PROFILE_DIR/user.js" <<'PREFS'
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.sessionstore.resume_session_once", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("app.update.auto", false);
user_pref("app.update.enabled", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.disableResetPrompt", true);
user_pref("browser.sessionstore.max_resumed_crashes", -1);
PREFS

echo "$LOG_PREFIX Launching Firefox ESR..."
exec firefox-esr \
    -profile "$PROFILE_DIR" \
    -kiosk \
    "file://$HTML_FILE"
