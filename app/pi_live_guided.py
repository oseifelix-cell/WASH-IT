"""
pi_live_guided.py — Full guided WHO hand-hygiene protocol, live on the Pi
==========================================================================
Reuses live_guided.py's phase machine, HUD drawing, and scoring logic
COMPLETELY UNCHANGED. Two things are adapted for the Pi:

  1. Camera: live_guided.py's open_camera() opens a USB/DirectShow webcam
     via cv2.VideoCapture, which doesn't exist on this board (confirmed
     via v4l2-ctl -- only the CSI Unicam interface is present). This
     wraps picamera2 in a small adapter exposing the same .read() /
     .release() interface, so live_guided.py's own camera loop doesn't
     need to change at all.

  2. Timing: TARGET_FPS / SEQ_LEN / STRIDE are patched to the values
     already validated against recorded video (5 FPS / 22-frame window /
     stride 4 -- correctly caught a genuinely missing step with zero
     false hits) BEFORE live_webcam.py and live_guided.py are imported,
     so their derived constants (BUF_LEN, VIS_WINDOW) compute correctly
     against the new values instead of the original 15 FPS / 64-frame ones.

REQUIRES A DISPLAY. live_guided.py draws its HUD with cv2.imshow, which
needs an actual desktop session on the Pi (HDMI/touchscreen with the
desktop environment running, or a VNC session) -- NOT a plain headless
SSH terminal. If you're SSH'd in with nothing attached to the Pi's video
output, this will fail with a "cannot open display" error.

Copy alongside the existing bundle (all already on the Pi except this file):
    pi_live_guided.py
    live_guided.py
    live_webcam.py
    live_sequence_stream.py
    test_continuous_sequence_local.py
    model_tcn_attention.py
    student_best.pt
    norm_stats.npz
    hand_landmarker.task

USAGE (run this ON the Pi's own desktop/touchscreen terminal, not headless SSH):
    python3 pi_live_guided.py
    python3 pi_live_guided.py --target-fps 5 --seq-len 22 --stride 4
    python3 pi_live_guided.py --flip     # if left/right look mirrored

Controls (same as the original): [r] restart   [q] quit
"""

import argparse, time

import cv2
from picamera2 import Picamera2

import test_continuous_sequence_local as tcs

# Must match the exact title string live_guided.py's run_guided() passes to
# cv2.imshow() -- pre-creating the window under this name lets us control
# its size/position before live_guided's own bare imshow() calls take over.
WINDOW_TITLE = "WHO hand hygiene — guided"


class Picamera2Capture:
    """Wraps picamera2 to look like cv2.VideoCapture (.read() / .release()),
    so live_guided.py's camera loop works completely unchanged."""
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)  # let auto-exposure/white-balance settle

    def read(self):
        frame = self.picam2.capture_array()   # BGR-ordered despite "RGB888"
        return True, frame

    def release(self):
        self.picam2.stop()


def main():
    ap = argparse.ArgumentParser(description="Guided WHO protocol, live on the Pi")
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--seq-len", type=int, default=22)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--step-seconds", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=8.0)
    ap.add_argument("--min-detect", type=float, default=1.2)
    ap.add_argument("--min-visibility", type=float, default=0.2)
    ap.add_argument("--motion-thresh", type=float, default=0.006)
    ap.add_argument("--soap-seconds", type=float, default=5.0)
    ap.add_argument("--rinse-seconds", type=float, default=5.0)
    ap.add_argument("--retry-seconds", type=float, default=8.0)
    ap.add_argument("--reveal-grace", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    # ── Patch timing BEFORE importing live_webcam / live_guided, so their
    #    derived constants (BUF_LEN, VIS_WINDOW...) compute against the
    #    new values instead of the original 15 FPS / 64-frame numbers.
    #    These modules do `from test_continuous_sequence_local import X`,
    #    which binds the value AT IMPORT TIME -- so the patch has to
    #    happen first, in this same process, before that import runs.
    tcs.TARGET_FPS = args.target_fps
    tcs.SEQ_LEN = args.seq_len
    tcs.STRIDE = args.stride
    print(f">>> TARGET_FPS={tcs.TARGET_FPS}  SEQ_LEN={tcs.SEQ_LEN}  "
          f"STRIDE={tcs.STRIDE}  (window = {tcs.SEQ_LEN/tcs.TARGET_FPS:.1f}s)")

    import live_webcam
    import live_guided
    print(f">>> live_guided BUF_LEN={live_guided.BUF_LEN}  "
          f"VIS_WINDOW={live_guided.VIS_WINDOW}\n")

    # ── Swap the camera opener for the Pi Camera Module ──────────────────
    def open_camera_pi(cam_id):
        try:
            return Picamera2Capture(args.width, args.height)
        except Exception as e:
            print(f"  x Could not open the CSI camera: {e}")
            print(f"    Check `libcamera-hello --list-cameras` and the "
                  f"ribbon cable seating.")
            return None
    live_guided.open_camera = open_camera_pi

    # ── Pre-create the display window with an explicit size, matching what
    #    worked in pi_display_test.py -- without this, the Qt HighGUI window
    #    live_guided's bare cv2.imshow() auto-creates ends up oversized with
    #    the actual frame squeezed into a small corner instead of filling it.
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.width, args.height)
    cv2.moveWindow(WINDOW_TITLE, 0, 0)

    live_guided.run_guided(
        cam_id=0,  # unused by open_camera_pi, kept for signature compatibility
        flip=args.flip, stride=args.stride,
        step_seconds=args.step_seconds, grace=args.grace,
        min_detect=args.min_detect, min_visibility=args.min_visibility,
        motion_thresh=args.motion_thresh, soap_seconds=args.soap_seconds,
        rinse_seconds=args.rinse_seconds, retry_seconds=args.retry_seconds,
        reveal_grace=args.reveal_grace,
    )


if __name__ == "__main__":
    main()
