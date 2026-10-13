#!/usr/bin/env bash
# pi_kiosk_watchdog_setup.sh — Installs a systemd timer that restarts the
# kiosk automatically every 20 minutes.
#
# This doesn't fix a root cause we haven't actually confirmed yet (an
# intermittent Chromium hang is genuinely hard to detect reliably from
# outside the browser) -- it's a pragmatic, standard kiosk-deployment
# safety net: if Chromium ever goes unresponsive, the device recovers on
# its own well within a normal wait, instead of needing someone to notice
# and manually power-cycle the Pi.
#
# Combined with pi_kiosk_launch.sh (which fixes the COLD-BOOT race that
# caused the white screen at startup), this covers both "goes wrong at
# boot" and "goes wrong sometime during use."
#
# USAGE (on the Pi):
#   chmod +x pi_kiosk_watchdog_setup.sh
#   ./pi_kiosk_watchdog_setup.sh

set -e

sudo tee /etc/systemd/system/washit-kiosk-restart.service > /dev/null <<'EOF'
[Unit]
Description=Periodic WASH-IT kiosk restart (self-healing safety net)

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl restart washit-kiosk.service
EOF

sudo tee /etc/systemd/system/washit-kiosk-restart.timer > /dev/null <<'EOF'
[Unit]
Description=Restart the WASH-IT kiosk every 20 minutes

[Timer]
OnUnitActiveSec=20min
OnBootSec=20min
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now washit-kiosk-restart.timer

echo ""
echo "=== Done ==="
echo "  Check it's scheduled:"
echo "    systemctl list-timers washit-kiosk-restart.timer"
echo ""
echo "  A restart every 20 minutes is a brief (few-second) flicker on the"
echo "  screen, not a full reboot -- the backend keeps running the whole"
echo "  time, only the browser window restarts."
echo ""
echo "  To change the interval, edit OnUnitActiveSec in:"
echo "    sudo systemctl edit --full washit-kiosk-restart.timer"
