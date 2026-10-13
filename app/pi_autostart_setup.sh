#!/usr/bin/env bash
# pi_autostart_setup.sh — Install systemd services so WASH-IT starts itself
# on boot: no SSH, no manual commands, no keyboard needed.
#
# Two services:
#   washit-server.service  -- runs pi_server.py (starts in IDLE, does no
#                              camera/model work until the UI connects)
#   washit-kiosk.service   -- launches Chromium in kiosk mode, pointed at
#                              the local HTML file, after the server + the
#                              desktop session are both up
#
# USAGE (on the Pi):
#   chmod +x pi_autostart_setup.sh
#   ./pi_autostart_setup.sh

set -e

PROJECT_DIR="$HOME/washit"
VENV_DIR="$PROJECT_DIR"   # the venv IS ~/washit itself, per your setup
USER_NAME="$(whoami)"

echo "=== Installing washit-server.service ==="
sudo tee /etc/systemd/system/washit-server.service > /dev/null <<EOF
[Unit]
Description=WASH-IT backend (camera + model + WebSocket server)
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python3 pi_server.py --flip
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "=== Installing washit-kiosk.service ==="
sudo tee /etc/systemd/system/washit-kiosk.service > /dev/null <<EOF
[Unit]
Description=WASH-IT kiosk display
After=washit-server.service graphical.target
Requires=washit-server.service
Wants=graphical.target

[Service]
Type=simple
User=$USER_NAME
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$USER_NAME/.Xauthority
ExecStart=/bin/bash $PROJECT_DIR/pi_kiosk_launch.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable washit-server.service
sudo systemctl enable washit-kiosk.service

echo ""
echo "=== Done ==="
echo "  Both services are enabled but NOT started yet. Test manually first:"
echo "    sudo systemctl start washit-server.service"
echo "    sudo systemctl status washit-server.service"
echo "    journalctl -u washit-server.service -f     # watch live logs"
echo ""
echo "  Once that looks healthy:"
echo "    sudo systemctl start washit-kiosk.service"
echo ""
echo "  If both look right, reboot to confirm true auto-start end to end:"
echo "    sudo reboot"
echo ""
echo "  To go back to manual control at any time:"
echo "    sudo systemctl stop washit-server washit-kiosk"
echo "    sudo systemctl disable washit-server washit-kiosk"
