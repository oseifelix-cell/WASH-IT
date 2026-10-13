"""
pir_test.py — Isolated PIR sensor test, completely separate from pi_server.py
==========================================================================
Tests ONLY the GPIO/gpiozero PIR initialization and motion detection --
nothing else (no camera, no model, no WebSocket). This lets us find out
fast whether the sensor itself, the wiring, or the gpiozero pin factory is
the problem, without waiting through pi_server.py's full ~15s startup
each time.

USAGE:
    python3 pir_test.py --pin 4
    python3 pir_test.py --pin 4 --seconds 90
"""

import argparse, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--pin", type=int, default=4, help="BCM GPIO pin number")
ap.add_argument("--seconds", type=float, default=90.0, help="How long to watch")
args = ap.parse_args()

print(f"[1] Python version: {sys.version}", flush=True)

print(f"[2] Importing gpiozero...", flush=True)
t0 = time.time()
try:
    from gpiozero import MotionSensor
    from gpiozero.pins.pigpio import PiGPIOFactory
    from gpiozero import Device
    print(f"    OK ({time.time()-t0:.2f}s). Default pin factory: {Device.pin_factory}", flush=True)
except Exception as e:
    print(f"    x Import failed: {e!r}", flush=True)
    sys.exit(1)

print(f"[3] Initializing MotionSensor on GPIO{args.pin}...", flush=True)
print(f"    (if this hangs with no further output, that's the actual bug --", flush=True)
print(f"     Ctrl+C after ~15s if nothing prints and paste me exactly this far)", flush=True)
t0 = time.time()
try:
    pir = MotionSensor(args.pin)
    print(f"    OK ({time.time()-t0:.2f}s)", flush=True)
except Exception as e:
    print(f"    x Failed after {time.time()-t0:.2f}s: {e!r}", flush=True)
    sys.exit(1)

print(f"[4] Current sensor state: motion_detected={pir.motion_detected}", flush=True)

print(f"[5] Watching for motion for {args.seconds:.0f}s -- wave your hand in "
      f"front of the sensor now...", flush=True)

count = [0]
def on_motion():
    count[0] += 1
    print(f"    >>> MOTION DETECTED (#{count[0]}) at t={time.time()-t0:.1f}s", flush=True)

def on_no_motion():
    print(f"    ... motion cleared at t={time.time()-t0:.1f}s", flush=True)

pir.when_motion = on_motion
pir.when_no_motion = on_no_motion

t0 = time.time()
while time.time() - t0 < args.seconds:
    time.sleep(1.0)

print(f"\n[done] Total motion events detected: {count[0]}", flush=True)
if count[0] == 0:
    print("  No motion detected at all -- check wiring (VCC/GND/OUT), the", flush=True)
    print("  sensor's sensitivity/delay potentiometers, and that you were", flush=True)
    print("  actually in front of it during the test window.", flush=True)
