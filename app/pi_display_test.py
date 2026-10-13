"""
pi_display_test.py — Isolate whether the display issue is Wayland/X11-related
==========================================================================
Does NOTHING except open the camera and show raw frames with cv2.imshow.
No MediaPipe, no model, no live_guided logic -- if the video feed doesn't
appear on screen here either, the problem is confirmed to be the display
backend (most likely Wayland vs X11), not anything in the WASH-IT pipeline.

USAGE (on the Pi's own desktop terminal, or via SSH with DISPLAY=:0):
    DISPLAY=:0 python3 pi_display_test.py

Press 'q' in the window to quit.
"""

import time
import cv2
from picamera2 import Picamera2

print("Opening camera...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1.0)
print("Camera open. Attempting to show window...")

cv2.namedWindow("display test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("display test", 640, 480)
cv2.moveWindow("display test", 50, 50)

n = 0
t0 = time.time()
while True:
    frame = picam2.capture_array()
    cv2.putText(frame, f"frame {n}  t={time.time()-t0:.1f}s", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("display test", frame)
    n += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if n == 1:
        print("First imshow call completed without error.")
    if n % 30 == 0:
        print(f"  frame {n}, still running (Ctrl+C or 'q' in window to stop)")

picam2.stop()
cv2.destroyAllWindows()
print("Done.")
