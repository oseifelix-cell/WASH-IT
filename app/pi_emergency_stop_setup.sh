#!/usr/bin/env bash
# pi_emergency_stop_setup.sh — Grants the minimum sudo permission needed for
# the UI's Emergency Stop button to work.
#
# Without this, pi_server.py's "sudo systemctl stop ..." calls would hang
# forever waiting for a password that never comes (since there's no
# interactive terminal to type it into), and the button would silently do
# nothing.
#
# This is deliberately narrow -- it grants passwordless sudo for exactly
# two specific commands (stopping these two named services), not blanket
# sudo access.
#
# USAGE (on the Pi):
#   chmod +x pi_emergency_stop_setup.sh
#   ./pi_emergency_stop_setup.sh

set -e
USER_NAME="$(whoami)"
RULE_FILE="/etc/sudoers.d/washit-emergency-stop"

echo "Granting $USER_NAME passwordless sudo for exactly:"
echo "  systemctl stop washit-kiosk.service"
echo "  systemctl stop washit-server.service"

# visudo -c validates syntax before installing -- a broken sudoers file can
# lock out ALL sudo access, so never skip this check
TMP=$(mktemp)
cat > "$TMP" <<EOF
$USER_NAME ALL=(root) NOPASSWD: /usr/bin/systemctl stop washit-kiosk.service
$USER_NAME ALL=(root) NOPASSWD: /usr/bin/systemctl stop washit-server.service
EOF

sudo visudo -c -f "$TMP" && sudo cp "$TMP" "$RULE_FILE" && sudo chmod 440 "$RULE_FILE"
rm "$TMP"

echo ""
echo "Done. Test it directly (should NOT prompt for a password):"
echo "  sudo systemctl stop washit-kiosk.service"
echo "  sudo systemctl start washit-kiosk.service   # bring it back"
