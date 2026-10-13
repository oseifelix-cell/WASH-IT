"""
pi_server.py — WASH-IT WebSocket backend, adapted for the Raspberry Pi
==========================================================================
Same Engine (phase machine, scoring, WebSocket protocol) as server.py,
UNCHANGED -- the browser UI (washit_v7_live.html / washit_live_patch.js)
needs no changes at all, since the JSON message schema is identical.

Three things are adapted for the Pi:
  1. TARGET_FPS/SEQ_LEN/STRIDE patched to the validated config (5fps /
     22-frame window / stride 4) BEFORE any project modules are imported,
     so their derived constants compute correctly.
  2. Camera: picamera2 instead of cv2.VideoCapture (no USB webcam on this
     board -- confirmed via v4l2-ctl, only the CSI Unicam interface exists).
  3. camera_loop is split into two threads -- capture (fast, ~30fps: reads
     frames + JPEG-encodes them) and inference (MediaPipe + model, ~5fps)
     -- so the browser's video preview stays smooth instead of freezing
     while MediaPipe processes each sample, exactly the same fix applied
     to pi_live_guided_threaded.py.

Run (on the Pi, with a network reachable by the kiosk browser -- localhost
is fine, this never needs the internet):
    python3 pi_server.py                 # port 8000
    python3 pi_server.py --flip --port 8000

Then point washit_v7_live.html's "Live Camera Mode" box at:
    ws://localhost:8000/ws
"""

import functools
print = functools.partial(print, flush=True)  # every print in this file flushes immediately --
                                                 # without this, output can sit buffered for a
                                                 # while under systemd, making log timestamps look
                                                 # bunched together even when events were spread
                                                 # out in real time (misleading for diagnosis)

import test_continuous_sequence_local as _tcs
_tcs.TARGET_FPS = 5.0
_tcs.SEQ_LEN = 22
_tcs.STRIDE = 4
print(f">>> Patched TARGET_FPS={_tcs.TARGET_FPS}  SEQ_LEN={_tcs.SEQ_LEN}  "
      f"STRIDE={_tcs.STRIDE}  (window={_tcs.SEQ_LEN/_tcs.TARGET_FPS:.1f}s)")


import argparse, asyncio, base64, json, os, queue, subprocess, threading, time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
from picamera2 import Picamera2

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip install websockets")

from test_continuous_sequence_local import (
    MP_TASK, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF, MAX_HANDS,
    TARGET_FPS, SEQ_LEN, NUM_CLASSES, STEP_NAMES,
    mediapipe_to_raw, build_features, load_model,
)
from live_webcam import open_camera
from live_guided import (
    fresh, activity_state, step_verdict, score_ordered, PRIME_COUNTDOWN,
)

REDO_GETSET = 3.0   # seconds of "get ready" before each redo step
REDO_INTRO  = 3.0   # seconds of "missed steps" alert before the redo begins


# ── guidance text helpers (mirror the OpenCV HUD hints) ──────────────────────
def gate_hint(state, doing):
    if state == "WASHING": return f"Good — keep {doing}"
    if state == "IDLE":    return "Keep your hands moving"
    return "Show your hands to the camera"


def run_hint(state, pred, expected):
    if state == "NO_HANDS": return "Show your hands"
    if state == "IDLE":     return "Start rubbing"
    return ""


# ── the inference engine: one frame in, one status dict out ──────────────────
class Engine:
    def __init__(self, model, mean, std, landmarker, cfg):
        self.model, self.mean, self.std = model, mean, std
        self.lm = landmarker
        self.cfg = cfg
        self.prog_start = time.time()
        self.reset()

    def reset(self):
        self.s = fresh(NUM_CLASSES)
        self.last_sample = 0.0
        self.fps_t, self.fps_n, self.fps_val = time.time(), 0, TARGET_FPS
        self.sample_dt = 1.0 / TARGET_FPS
        self.session_start = time.time()
        self.run_start = None
        self.rinse_start = None
        self.done_time = None
        self.final = None
        self._last_phase = None
        self._last_slot = -1

    # encode a downscaled JPEG -> raw bytes (sent as a binary WS frame --
    # no base64 inflation, no JSON string escaping, cheaper on both ends)
    def _frame_bytes(self, frame):
        w = self.cfg["frame_width"]
        h = int(frame.shape[0] * w / frame.shape[1])
        small = cv2.resize(frame, (w, h))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["jpeg_q"]])
        return buf.tobytes() if ok else None

    # legacy base64 path -- kept for API compatibility, unused by the Pi's
    # binary-frame pipeline (capture_loop uses _frame_bytes instead)
    def _frame_b64(self, frame):
        raw = self._frame_bytes(frame)
        return base64.b64encode(raw).decode("ascii") if raw is not None else None

    def _final_payload(self):
        s = self.s; n = NUM_CLASSES
        per, credit = [], 0.0
        for k in range(n):
            r = s["results"][k]
            c = 1.0 if (r["passed"] and not r.get("retry")) else (0.5 if r["passed"] else 0.0)
            credit += c
            per.append({"passed": bool(r["passed"]), "retry": bool(r.get("retry")),
                        "conf": round(float(r["conf"]), 2)})
        steps_frac = credit / n
        soap = 1.0 if s["soap_done"] else 0.0
        rinse = 1.0 if s["rinse_done"] else 0.0
        score = round(100 * (0.70 * steps_frac + 0.15 * soap + 0.15 * rinse))
        grade = ("A+" if score >= 95 else "A" if score >= 85 else
                 "B" if score >= 70 else "C" if score >= 55 else "D")
        passed = sum(1 for r in s["results"] if r["passed"])
        corrected = sum(1 for r in s["results"] if r["passed"] and r.get("retry"))
        order = ("Perfect" if passed == n and corrected == 0 else
                 "Corrected" if passed == n else "Incomplete")
        if self.run_start is not None:
            end = self.rinse_start if self.rinse_start is not None else (self.done_time or time.time())
            dur = end - self.run_start
        else:
            dur = (self.done_time or time.time()) - self.session_start
        return dict(finished=True, results=per, soap_done=s["soap_done"],
                    rinse_done=s["rinse_done"], score=score, grade=grade,
                    steps_done=passed, corrected=corrected, order=order,
                    duration_s=round(dur))

    def step(self, frame, now, encode_frame=True):
        s = self.s
        c = self.cfg
        n = NUM_CLASSES
        L = c["step_seconds"]
        total_s = L * n
        status = {"phase": s["phase"], "inference": False}  # frame sent separately as binary WS frame

        win_period = c["stride"] / max(self.fps_val, 1e-3)
        did_infer = False

        if s["phase"] != "DONE" and now - self.last_sample >= self.sample_dt:
            dt_sample = now - self.last_sample
            self.last_sample = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.lm.detect_for_video(img, int((now - self.prog_start) * 1000))
            s["raw_buf"].append(mediapipe_to_raw(res))

            raw_stack = np.stack(s["raw_buf"])
            s["state"], s["motion"] = activity_state(raw_stack, c["min_visibility"], c["motion_thresh"])
            washing = (s["state"] == "WASHING")
            buffering = len(s["raw_buf"]) < SEQ_LEN

            self.fps_n += 1
            if now - self.fps_t >= 0.5:
                self.fps_val = self.fps_n / (now - self.fps_t); self.fps_t, self.fps_n = now, 0

            if dt_sample < 0.5 and washing:
                if s["phase"] == "SOAP":  s["soap_s"]  += dt_sample
                if s["phase"] == "RINSE": s["rinse_s"] += dt_sample

            if s["phase"] in ("RUN", "REMEDIATE") and washing and not buffering \
                    and (s["si"] % c["stride"] == 0):
                window = build_features(np.stack(s["raw_buf"]))[-SEQ_LEN:]
                x = torch.from_numpy(((window - self.mean) / self.std).astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    p = F.softmax(self.model(x)[0], dim=-1).squeeze(0).cpu().numpy()
                s["prob_buf"].append(p)
                p_sm = np.mean(s["prob_buf"], axis=0)
                s["pred"], s["conf"] = int(np.argmax(p_sm)), float(np.max(p_sm))
                did_infer = True
                if s["phase"] == "RUN":
                    s["records"].append((now - s["start"], s["pred"], s["conf"]))
                    if os.environ.get("WASHIT_DEBUG_PREDICTIONS"):
                        _elapsed_dbg = now - s["start"]
                        _slot_dbg = min(NUM_CLASSES - 1, int(_elapsed_dbg // c["step_seconds"]))
                        _match = "OK " if s["pred"] == _slot_dbg else "!! "
                        print(f"   [debug] t={_elapsed_dbg:5.1f}s  in-slot={STEP_NAMES[_slot_dbg]:<22} "
                              f"predicted={STEP_NAMES[s['pred']]:<22} conf={s['conf']:.2f}  {_match}",
                              flush=True)
                else:
                    q = s["missed_queue"]
                    if s.get("retry_start") is not None and s["retry_idx"] < len(q) \
                            and s["pred"] == q[s["retry_idx"]]:
                        s["retry_hit"] += 1; s["retry_conf_sum"] += s["conf"]
            s["si"] += 1

        status["state"] = s["state"]
        status["fps"] = round(self.fps_val, 1)
        phase = s["phase"]

        if phase == "PRIME":
            buffering = len(s["raw_buf"]) < SEQ_LEN
            if (not buffering) and s["state"] != "NO_HANDS":
                s["phase"] = "SOAP"
            status.update(buffering=buffering,
                          buffer_fill=min(1.0, len(s["raw_buf"]) / SEQ_LEN),
                          visibility_ok=(s["state"] != "NO_HANDS"),
                          guidance=("Show both hands to begin" if s["state"] == "NO_HANDS"
                                    else "Hold still, starting…"))

        elif phase == "SOAP":
            rem = max(0.0, c["soap_seconds"] - s["soap_s"])
            if rem <= 0:
                s["soap_done"] = True; s["phase"] = "GETSET"; s["go_since"] = None
            status.update(phase_prompt="Wet hands & apply soap",
                          phase_remaining=round(rem, 1), state=s["state"],
                          guidance=gate_hint(s["state"], "rubbing soap in"))

        elif phase == "GETSET":
            if s["go_since"] is None: s["go_since"] = now
            left = PRIME_COUNTDOWN - (now - s["go_since"])
            if left <= 0:
                s["phase"] = "RUN"; s["start"] = now; s["prob_buf"].clear()
                self.run_start = now
            status.update(countdown=max(1, int(left) + 1), prompt="Get ready — Step 1")

        elif phase == "RUN":
            elapsed = now - s["start"]
            for k in range(n):
                if not s["revealed"][k] and elapsed >= (k + 1) * L + c["reveal_grace"]:
                    ok, cf, _, _ = step_verdict(s["records"], k, L, c["grace"], c["min_detect"], win_period)
                    s["revealed"][k] = True; s["live_pass"][k] = ok; s["live_conf"][k] = cf
            if elapsed >= total_s:
                s["results"] = score_ordered(s["records"], n, L, c["grace"], c["min_detect"], win_period)
                for k in range(n):
                    s["revealed"][k] = True
                    s["live_pass"][k] = s["results"][k]["passed"]
                    s["live_conf"][k] = s["results"][k]["conf"]
                missed = [k for k in range(n) if not s["results"][k]["passed"]]
                if missed:
                    s["missed_queue"] = missed; s["retry_idx"] = 0
                    s["redo_intro_since"] = now; s["redo_intro_done"] = False
                    s["redo_ready_since"] = now; s["retry_start"] = None
                    s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                    s["pred"] = -1; s["prob_buf"].clear(); s["phase"] = "REMEDIATE"
                else:
                    s["phase"] = "RINSE"
            else:
                slot = min(n - 1, int(elapsed // L)); s["cur_slot"] = slot
                if slot != self._last_slot:
                    print(f"   RUN: step {slot+1}/7  (t={elapsed:.0f}s, {self.fps_val:.1f} fps)")
                    self._last_slot = slot
                cur_det_s = sum(1 for (t, p, _) in s["records"]
                                if p == slot and slot * L <= t <= elapsed) * win_period
                completed = [k for k in range(n) if s["revealed"][k] and s["live_pass"][k]]
                failed = [k for k in range(n) if s["revealed"][k] and not s["live_pass"][k]]
                status.update(
                    inference=did_infer, current_step=slot, predicted_step=s["pred"],
                    confidence=round(s["conf"], 3),
                    progress=min(1.0, cur_det_s / max(c["min_detect"], 1e-3)),
                    cur_det_s=round(cur_det_s, 1), completed_steps=completed, failed_steps=failed,
                    slot_left=round(L - (elapsed - slot * L), 1),
                    total_left=round(total_s - elapsed, 1),
                    visibility_ok=(s["state"] != "NO_HANDS"),
                    buffering=(len(s["raw_buf"]) < SEQ_LEN),
                    buffer_fill=min(1.0, len(s["raw_buf"]) / SEQ_LEN),
                    is_compliant=(s["state"] == "WASHING" and s["pred"] == slot),
                    struggling=(s["state"] == "WASHING" and s["pred"] >= 0 and s["pred"] != slot),
                    guidance=run_hint(s["state"], s["pred"], slot))

        elif phase == "REMEDIATE":
            q = s["missed_queue"]
            if s["retry_idx"] >= len(q):
                s["phase"] = "RINSE"
            else:
                k = q[s["retry_idx"]]
                completed = [j for j in range(n) if s["results"][j]["passed"]]
                failed = [j for j in range(n) if not s["results"][j]["passed"]]

                if not s.get("redo_intro_done", False):
                    # one-time ~3s alert announcing which steps were missed
                    intro_left = REDO_INTRO - (now - s["redo_intro_since"])
                    if intro_left <= 0:
                        s["redo_intro_done"] = True; s["redo_ready_since"] = now
                    else:
                        status.update(inference=did_infer, redo_phase="intro",
                                      redo_missed=list(q), redo_intro_left=max(1, int(intro_left) + 1),
                                      completed_steps=completed, failed_steps=failed, state=s["state"])

                if s.get("redo_intro_done", False):
                    if s["retry_start"] is None:
                        # 3-second GET-READY before this redo step
                        left = REDO_GETSET - (now - s["redo_ready_since"])
                        if left <= 0:
                            s["retry_start"] = now; s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                            s["pred"] = -1; s["prob_buf"].clear()
                        else:
                            status.update(inference=did_infer, redo_step=k, redo_phase="getready",
                                          redo_countdown=max(1, int(left) + 1),
                                          completed_steps=completed, failed_steps=failed,
                                          redo_queue=q[s["retry_idx"]:], state=s["state"])

                    if s["retry_start"] is not None:
                        # PERFORM window (retry_seconds)
                        det_s = s["retry_hit"] * win_period
                        elapsed_r = now - s["retry_start"]
                        advanced = False
                        if det_s >= c["min_detect"]:
                            s["results"][k].update(passed=True, retry=True,
                                conf=s["retry_conf_sum"] / max(1, s["retry_hit"]))
                            print(f"   REDO: step {k+1} CORRECTED")
                            advanced = True
                        elif elapsed_r >= c["retry_seconds"]:
                            print(f"   REDO: step {k+1} still missed")
                            advanced = True
                        if advanced:
                            s["retry_idx"] += 1
                            s["redo_ready_since"] = now; s["retry_start"] = None
                            s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                            s["pred"] = -1; s["prob_buf"].clear()
                            if s["retry_idx"] >= len(q):
                                s["phase"] = "RINSE"
                        else:
                            completed = [j for j in range(n) if s["results"][j]["passed"]]
                            failed = [j for j in range(n) if not s["results"][j]["passed"]]
                            status.update(
                                inference=did_infer, redo_step=k, redo_phase="perform",
                                redo_remaining=round(max(0.0, c["retry_seconds"] - elapsed_r), 1),
                                predicted_step=s["pred"], confidence=round(s["conf"], 3),
                                detecting=(s["pred"] == k), completed_steps=completed,
                                failed_steps=failed, redo_queue=q[s["retry_idx"]:], state=s["state"])

        elif phase == "RINSE":
            if self.rinse_start is None:
                self.rinse_start = now
            rem = max(0.0, c["rinse_seconds"] - s["rinse_s"])
            if rem <= 0:
                s["rinse_done"] = True; s["phase"] = "DONE"; self.done_time = now
            status.update(phase_prompt="Rinse hands under water",
                          phase_remaining=round(rem, 1), state=s["state"],
                          guidance=gate_hint(s["state"], "rinsing"))

        if s["phase"] == "DONE":
            if self.final is None:
                self.final = self._final_payload()
            status["phase"] = "DONE"
            status.update(self.final)

        if s["phase"] != self._last_phase:
            print(f"  [phase] {self._last_phase} -> {s['phase']}  ({self.fps_val:.1f} fps)")
            self._last_phase = s["phase"]

        return status


# ── shared state between camera thread and websocket server ──────────────────
ENGINE = None
LOCK = threading.Lock()
LATEST = {"phase": "IDLE", "frame": None, "inference": False}
RESET = threading.Event()
STOP = threading.Event()


def idle_watchdog(dwell_s=25.0):
    """After a session reaches DONE, the results stay on screen for a
    while, then the engine automatically returns to IDLE -- so the PIR
    sensor can trigger the NEXT person's session without anyone needing to
    manually reset. Without this, DONE is a dead end and the PIR silently
    stops working after the very first use (it only fires while IDLE)."""
    announced_for = None
    while not STOP.is_set():
        time.sleep(1.0)
        if ENGINE is None:
            continue
        if ENGINE.s.get("phase") != "DONE":
            announced_for = None
            continue
        done_time = getattr(ENGINE, "done_time", None)
        if done_time is None:
            continue
        if announced_for != done_time:
            print(f"  [watchdog] session done -- returning to IDLE in {dwell_s:.0f}s", flush=True)
            announced_for = done_time
        if time.time() - done_time >= dwell_s:
            ENGINE.reset()
            ENGINE.s["phase"] = "IDLE"
            print("  [watchdog] back to IDLE -- ready for the next person", flush=True)


def pir_loop(pin):
    """Watches a PIR motion sensor on the given GPIO pin. When someone
    approaches while the engine is IDLE (nobody mid-session), pushes a
    'trigger' event over SSE to the browser -- which then runs the EXACT
    SAME code path a manual Start-button click does (triggerSession() ->
    startDemo() -> connectLiveBackend() -> sends 'reset'). This is
    deliberately the ONLY way a session gets started, whether by a human
    tap or the PIR -- avoids two different code paths fighting each other."""
    try:
        from gpiozero import MotionSensor
    except Exception as e:
        print(f"  [pir] gpiozero not available ({e}) -- PIR trigger disabled, "
              f"manual Start button/UI trigger still works.", flush=True)
        return
    try:
        pir = MotionSensor(pin)
    except Exception as e:
        print(f"  [pir] could not initialize sensor on GPIO{pin}: {e} -- "
              f"PIR trigger disabled, manual Start button/UI trigger still works.", flush=True)
        return
    warmup_s = 45.0
    print(f"  [pir] watching GPIO{pin} for motion "
          f"(warming up for {warmup_s:.0f}s -- ignoring triggers until then, "
          f"this is normal PIR sensor behavior right after power-on)", flush=True)
    ready_at = time.time() + warmup_s

    last_trigger = 0.0
    cooldown_s = 3.0   # basic debounce against rapid re-fires

    def on_motion():
        nonlocal last_trigger
        now = time.time()
        if now < ready_at:
            print("  [pir] motion detected but still warming up -- ignored")
            return
        if now - last_trigger < cooldown_s:
            print("  [pir] motion detected but within cooldown -- ignored")
            return
        if ENGINE is None or ENGINE.s["phase"] != "IDLE":
            phase = ENGINE.s["phase"] if ENGINE is not None else "no engine"
            print(f"  [pir] motion detected but engine is not IDLE "
                  f"(phase={phase}) -- ignored, not starting a session")
            return   # a session is already running -- don't interrupt it
        last_trigger = now
        print("  [pir] motion detected -- notifying UI to start session")
        push_sse_event("trigger")

    pir.when_motion = on_motion
    warmup_announced = False
    while not STOP.is_set():
        if not warmup_announced and time.time() >= ready_at:
            print("  [pir] warm-up complete -- now active", flush=True)
            warmup_announced = True
        time.sleep(0.5)


# ── SSE (Server-Sent Events) -- pushes the PIR "trigger" signal to the
#    browser. Deliberately separate from the WebSocket used for frames/
#    status; this is a one-way, fire-and-forget notification channel.
_sse_clients = []
_sse_lock = threading.Lock()


def push_sse_event(msg):
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(msg)
            except Exception:
                pass


def start_web_server(port):
    """Minimal Flask app serving ONLY the /events SSE stream the PIR uses
    to notify the browser. Does NOT serve the HTML itself -- the kiosk
    loads washit_v7_live.html directly from disk via file://, same as
    before; this is purely the PIR -> browser notification channel."""
    try:
        from flask import Flask, Response
    except Exception as e:
        print(f"  [web] Flask not available ({e}) -- PIR SSE trigger disabled, "
              f"manual Start button still works.", flush=True)
        return

    app = Flask(__name__)

    @app.after_request
    def add_cors(resp):
        # the kiosk loads the HTML via file://, a different "origin" than
        # http://localhost:5000 -- CORS header keeps EventSource happy
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.route("/events")
    def events():
        q = queue.Queue()
        with _sse_lock:
            # Only ONE legitimate kiosk browser should ever exist. Rather
            # than trust that a previous, possibly-zombie connection
            # (from a force-killed Chromium instance) gets cleaned up
            # promptly, just replace the whole list on every new connect --
            # guarantees at most one listener can ever react to a trigger,
            # even if an old one never got a chance to clean itself up.
            print(f"  [web] new /events subscriber -- replacing "
                  f"{len(_sse_clients)} previous listener(s)")
            _sse_clients.clear()
            _sse_clients.append(q)

        def stream():
            try:
                while not STOP.is_set():
                    try:
                        msg = q.get(timeout=1.0)
                        yield f"data: {msg}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"   # keeps the connection open
            finally:
                with _sse_lock:
                    if q in _sse_clients:
                        _sse_clients.remove(q)

        return Response(stream(), mimetype="text/event-stream")

    print(f"  [web] SSE endpoint serving at http://0.0.0.0:{port}/events", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


class Picamera2Capture:
    """Wraps picamera2 to look like cv2.VideoCapture (.read()/.release())."""
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

    def read(self):
        return True, self.picam2.capture_array()   # BGR-ordered despite "RGB888"

    def release(self):
        self.picam2.stop()


# shared raw frame handed from the capture thread to the inference thread
_RAW_FRAME = {"frame": None}
# shared JPEG bytes handed from the capture thread to the WS producer
# (binary frame channel -- kept separate from LATEST's JSON status dict)
_LATEST_JPEG = {"bytes": None}


def capture_loop(width, height, flip):
    """Fast thread: captures + JPEG-encodes frames continuously (~30fps on
    this board). Updates LATEST["frame"] in place so the browser's video
    preview stays smooth, independent of how often inference finishes."""
    global LATEST
    # Retry with backoff -- right at boot, the camera driver stack can take
    # a moment to be ready, and giving up on the first try left the service
    # "running" forever with a permanently dead camera thread (no crash for
    # systemd to notice and restart). Exit non-zero if it truly never comes
    # up, so Restart=on-failure actually keeps retrying at the service level.
    cap = None
    for attempt in range(1, 11):
        try:
            cap = Picamera2Capture(width, height)
            break
        except Exception as e:
            print(f"  [capture] camera open attempt {attempt}/10 failed: {e}")
            time.sleep(2.0)
    if cap is None:
        print("  x Could not open the CSI camera after 10 attempts -- exiting "
              "so systemd restarts the whole service.")
        os._exit(1)
    print(f"  Pi camera open ({width}x{height}) — streaming")
    n = 0
    encode_fail_streak = 0
    while not STOP.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        if flip: frame = cv2.flip(frame, 1)
        _RAW_FRAME["frame"] = frame
        if n == 0:
            print(f"  [capture] first frame: shape={frame.shape} dtype={frame.dtype} "
                  f"contiguous={frame.flags['C_CONTIGUOUS']}")
        try:
            jpeg_bytes = ENGINE._frame_bytes(frame)
        except Exception as e:
            jpeg_bytes = None
            if encode_fail_streak == 0:
                print(f"  [capture] _frame_bytes raised: {e!r}")
            encode_fail_streak += 1
        else:
            if jpeg_bytes is None:
                if encode_fail_streak == 0:
                    print("  [capture] _frame_bytes returned None (cv2.imencode failed)")
                encode_fail_streak += 1
            else:
                if encode_fail_streak > 0:
                    print(f"  [capture] recovered after {encode_fail_streak} failed encodes")
                encode_fail_streak = 0
        _LATEST_JPEG["bytes"] = jpeg_bytes
        n += 1


def inference_loop():
    """Slower thread: MediaPipe + model inference + phase machine (~5fps,
    the Pi's real throughput). Updates every LATEST field EXCEPT "frame",
    which the capture thread owns, so status text updates independently
    of video smoothness."""
    global LATEST
    sample_dt = ENGINE.sample_dt
    last = time.time()
    while not STOP.is_set():
        now = time.time()
        if now - last < sample_dt:
            time.sleep(0.005); continue
        last = now
        frame = _RAW_FRAME["frame"]
        if frame is None:
            continue
        if RESET.is_set():
            ENGINE.reset(); RESET.clear(); print("  -- session reset --")
        if ENGINE.s["phase"] == "IDLE":
            # waiting for the browser's "Start Handwashing Session" button --
            # no camera/MediaPipe/model work happens until then
            with LOCK:
                LATEST["phase"] = "IDLE"
            continue
        status = ENGINE.step(frame, now, encode_frame=False)
        status.pop("frame", None)   # never clobber the capture thread's frame
        with LOCK:
            LATEST.update(status)


def _json_default(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    return str(o)


async def ws_handler(ws, *args):
    print("  client connected")

    async def consumer():
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)): msg = msg.decode()
                if "emergency_stop" in msg.lower():
                    print("  !! EMERGENCY STOP requested from UI -- returning to desktop")
                    # fire-and-forget: stop the kiosk browser first (immediate
                    # visual effect), then this server itself. Needs the
                    # passwordless sudoers rule from pi_emergency_stop_setup.sh.
                    subprocess.Popen(["sudo", "systemctl", "stop", "washit-kiosk.service"])
                    subprocess.Popen(["sudo", "systemctl", "stop", "washit-server.service"])
                elif "reset" in msg.lower():
                    RESET.set()
        except Exception as e:
            print("  consumer ended:", repr(e))

    async def producer():
        send_dt = 1.0 / ARGS.send_fps
        try:
            while True:
                jpeg_bytes = _LATEST_JPEG["bytes"]
                if jpeg_bytes is not None:
                    await ws.send(jpeg_bytes)          # binary frame, no base64
                with LOCK:
                    st = dict(LATEST)
                await ws.send(json.dumps(st, default=_json_default))  # text status
                await asyncio.sleep(send_dt)
        except Exception as e:
            print("  producer ended:", repr(e))

    await asyncio.gather(consumer(), producer())
    print("  client disconnected")


async def main_async():
    async with websockets.serve(ws_handler, "0.0.0.0", ARGS.port, max_size=2 ** 22,
                                ping_interval=None):
        print(f"\nWASH-IT server ready ->  ws://localhost:{ARGS.port}/ws")
        print("(paste that into the UI's 'Live Camera Mode' box)\n")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WASH-IT WebSocket backend")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--pir-pin", type=int, default=None,
                     help="BCM GPIO pin number the PIR sensor is wired to (e.g. 4). Omit to disable PIR auto-start.")
    ap.add_argument("--web-port", type=int, default=5000,
                     help="Port for the PIR SSE notification endpoint (/events).")
    ap.add_argument("--done-dwell", type=float, default=25.0,
                     help="Seconds to show the DONE results screen before automatically returning to IDLE for the next person.")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--send-fps", type=float, default=25.0)
    ap.add_argument("--frame-width", type=int, default=320)
    ap.add_argument("--jpeg-quality", type=int, default=55)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--step-seconds", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=8.0)
    ap.add_argument("--min-detect", type=float, default=1.2)
    ap.add_argument("--min-visibility", type=float, default=0.2)
    ap.add_argument("--motion-thresh", type=float, default=0.006)
    ap.add_argument("--soap-seconds", type=float, default=5.0)
    ap.add_argument("--rinse-seconds", type=float, default=5.0)
    ap.add_argument("--retry-seconds", type=float, default=8.0)
    ap.add_argument("--reveal-grace", type=float, default=3.0)
    ARGS = ap.parse_args()

    print("=" * 60)
    print("WASH-IT backend — loading model")
    print("=" * 60)
    model, mean, std, val_f1 = load_model("cpu")
    print(f"  val_f1 = {val_f1:.4f}")

    cfg = dict(stride=ARGS.stride, step_seconds=ARGS.step_seconds, grace=ARGS.grace,
               min_detect=ARGS.min_detect, min_visibility=ARGS.min_visibility,
               motion_thresh=ARGS.motion_thresh, soap_seconds=ARGS.soap_seconds,
               rinse_seconds=ARGS.rinse_seconds, retry_seconds=ARGS.retry_seconds,
               reveal_grace=ARGS.reveal_grace, frame_width=ARGS.frame_width,
               jpeg_q=ARGS.jpeg_quality)

    lm_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS, min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF, min_tracking_confidence=MP_TRK_CONF)
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(lm_opts)

    ENGINE = Engine(model, mean, std, landmarker, cfg)
    ENGINE.s["phase"] = "IDLE"  # wait for the UI's Start button, not the camera

    t_cap = threading.Thread(target=capture_loop, args=(ARGS.width, ARGS.height, ARGS.flip), daemon=True)
    t_inf = threading.Thread(target=inference_loop, daemon=True)
    t_cap.start()
    t_inf.start()
    if ARGS.pir_pin is not None:
        t_web = threading.Thread(target=start_web_server, args=(ARGS.web_port,), daemon=True)
        t_web.start()
        t_pir = threading.Thread(target=pir_loop, args=(ARGS.pir_pin,), daemon=True)
        t_pir.start()
        t_watchdog = threading.Thread(target=idle_watchdog, args=(ARGS.done_dwell,), daemon=True)
        t_watchdog.start()
    else:
        print("  [pir] --pir-pin not set -- PIR + auto-idle-return disabled, "
              "manual Start button only (original behavior)")

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        STOP.set()
