"""
kids_server.py - WASH-IT EDUCATION MODE backend
==========================================================================
Runs the education (kids) UI against the SAME model, camera pipeline and
scoring as compliance mode, WITHOUT modifying a single line of server.py,
live_guided.py, washit_v7_live.html or washit_live_patch.js.

It does that by importing server.py and subclassing its Engine:

    server.Engine            <- compliance, untouched
      KidsEngine(server.Engine)  <- education, adds a TEACH gate only

Nothing is overridden that affects scoring, step order, thresholds or the
redo logic. Compliance sessions and education sessions grade identically
for identical behaviour.

WHY A TEACH GATE
----------------
server.py's RUN phase is one continuous clock:

    elapsed = now - s["start"]
    slot    = int(elapsed // step_seconds)

The 7 slots run back to back, and every inference record is timestamped
against s["start"] before score_ordered() grades it. A child needs to hear
the guide explain a step BEFORE the slot starts counting, otherwise they
are marked down while still being taught.

So this subclass freezes the clock instead of inserting a phase: while a
teach hold is active it advances s["start"] (and the REMEDIATE timers) by
each tick's delta, which holds `elapsed` still. Records already stored keep
valid timestamps, and when the hold releases the axis continues exactly
where it stopped with the narration time excised. Scoring never sees it.

PROTOCOL ADDITIONS (education only)
-----------------------------------
Server -> client:
    phase == "TEACH"
    teach_kind   "step" | "redo"
    teach_step   0-6, which step is being explained
    teach_left   seconds until the safety timeout fires
    (plus frame, state, visibility_ok, completed_steps, failed_steps)

Client -> server:
    "teach_done"   narration finished, start the slot clock
    "reset"        same as compliance

The client is NOT trusted to be alive: if "teach_done" never arrives (audio
blocked, tab hidden, browser closed) the hold self-releases after
--teach-timeout seconds so a child is never stranded.

RUN:
    python kids_server.py                    # cam 2, port 8002
    python kids_server.py --cam 0 --flip --port 8002

Then point the education UI at:  ws://<host>:8002/ws-kids

Compliance mode keeps running exactly as before on its own port:
    python server.py --port 8001
"""

import argparse, asyncio, json, threading, time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip install websockets")

import server as compliance
from test_continuous_sequence_local import (
    MP_TASK, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF, MAX_HANDS,
    TARGET_FPS, SEQ_LEN, NUM_CLASSES, load_model, mediapipe_to_raw,
)
from live_guided import activity_state


class KidsEngine(compliance.Engine):
    """Compliance Engine + a narration gate. Overrides nothing else."""

    def __init__(self, model, mean, std, landmarker, cfg,
                 teach_timeout=15.0, teach_redo=True):
        self.teach_timeout = float(teach_timeout)
        self.teach_redo = bool(teach_redo)
        super().__init__(model, mean, std, landmarker, cfg)

    # ---- state ----------------------------------------------------------
    def reset(self):
        super().reset()
        self._teach_active = False
        self._teach_kind = None
        self._teach_step = -1
        self._teach_since = 0.0
        self._teach_last_tick = None
        self._taught_steps = set()      # RUN slots already explained
        self._taught_redo = set()       # redo steps already explained
        self._release_requested = False

    def request_release(self):
        """Called from the websocket consumer when the UI reports the
        narration has finished playing."""
        self._release_requested = True

    # ---- keep the landmark buffer warm while the clock is frozen --------
    def _pump_sample(self, frame, now):
        """Mirror of the sampling half of Engine.step: feeds MediaPipe and
        the activity gate so the model window is fresh when the hold ends,
        but appends NO inference records and touches no phase timer."""
        c = self.cfg
        s = self.s
        if now - self.last_sample < self.sample_dt:
            return
        self.last_sample = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, int((now - self.prog_start) * 1000))
        s["raw_buf"].append(mediapipe_to_raw(res))
        raw_stack = np.stack(s["raw_buf"])
        s["state"], s["motion"] = activity_state(
            raw_stack, c["min_visibility"], c["motion_thresh"])
        self.fps_n += 1
        if now - self.fps_t >= 0.5:
            self.fps_val = self.fps_n / (now - self.fps_t)
            self.fps_t, self.fps_n = now, 0
        s["si"] += 1

    # ---- freeze every clock that is currently running -------------------
    def _freeze(self, dt):
        s = self.s
        if dt <= 0 or dt > 1.0:      # ignore absurd deltas after a stall
            return
        if s.get("start") is not None:
            s["start"] += dt                    # RUN slot clock
        if s.get("redo_intro_since") is not None:
            s["redo_intro_since"] += dt         # REMEDIATE intro
        if s.get("redo_ready_since") is not None:
            s["redo_ready_since"] += dt         # REMEDIATE get-ready
        if s.get("retry_start") is not None:
            s["retry_start"] += dt              # REMEDIATE perform window

    def _begin_teach(self, kind, step, now):
        self._teach_active = True
        self._teach_kind = kind
        self._teach_step = int(step)
        self._teach_since = now
        self._teach_last_tick = now
        self._release_requested = False
        print(f"  [teach] hold for {kind} step {int(step) + 1}")

    def _end_teach(self, why):
        print(f"  [teach] release ({why})")
        self._teach_active = False
        self._teach_kind = None
        self._teach_step = -1
        self._release_requested = False
        # a fresh window after the pause, so no stale probabilities leak in
        self.s["prob_buf"].clear()
        self.s["pred"] = -1

    def _teach_status(self, frame, now):
        s = self.s
        n = NUM_CLASSES
        completed, failed = [], []
        if s.get("results"):
            completed = [k for k in range(n) if s["results"][k]["passed"]]
            failed = [k for k in range(n) if not s["results"][k]["passed"]]
        else:
            completed = [k for k in range(n) if s["revealed"][k] and s["live_pass"][k]]
            failed = [k for k in range(n) if s["revealed"][k] and not s["live_pass"][k]]
        left = max(0.0, self.teach_timeout - (now - self._teach_since))
        return {
            "frame": self._frame_b64(frame),
            "phase": "TEACH",
            "inference": False,
            "teach_kind": self._teach_kind,
            "teach_step": self._teach_step,
            "teach_left": round(left, 1),
            "state": s["state"],
            "fps": round(self.fps_val, 1),
            "visibility_ok": (s["state"] != "NO_HANDS"),
            "buffering": len(s["raw_buf"]) < SEQ_LEN,
            "buffer_fill": min(1.0, len(s["raw_buf"]) / SEQ_LEN),
            "completed_steps": completed,
            "failed_steps": failed,
            "guidance": ("Show both hands to the camera"
                         if s["state"] == "NO_HANDS" else "Watch and listen"),
        }

    # ---- the only overridden behaviour ----------------------------------
    def step(self, frame, now):
        # (1) already holding: freeze the clocks, keep the buffer warm
        if self._teach_active:
            dt = now - (self._teach_last_tick or now)
            self._teach_last_tick = now
            self._freeze(dt)
            self._pump_sample(frame, now)
            if self._release_requested:
                self._end_teach("client")
            elif now - self._teach_since >= self.teach_timeout:
                self._end_teach("timeout")
            else:
                return self._teach_status(frame, now)
            # released this tick: fall through to a normal tick

        # (2) normal compliance tick, untouched
        status = compliance.Engine.step(self, frame, now)
        phase = status.get("phase")

        # (3) does this tick open a step that has not been explained yet?
        if phase == "RUN":
            k = status.get("current_step")
            if isinstance(k, int) and k not in self._taught_steps:
                self._taught_steps.add(k)
                self._begin_teach("step", k, now)
                self._teach_last_tick = now
                return self._teach_status(frame, now)

        elif phase == "REMEDIATE" and self.teach_redo:
            k = status.get("redo_step")
            if isinstance(k, int) and k not in self._taught_redo:
                self._taught_redo.add(k)
                self._begin_teach("redo", k, now)
                self._teach_last_tick = now
                return self._teach_status(frame, now)

        return status


# ── websocket layer (education route only) ───────────────────────────────
async def ws_handler(ws, *args):
    print("  kids client connected")

    async def consumer():
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    msg = msg.decode()
                low = msg.lower()
                if "teach_done" in low:
                    if compliance.ENGINE is not None:
                        compliance.ENGINE.request_release()
                elif "reset" in low:
                    compliance.RESET.set()
        except Exception as e:
            print("  consumer ended:", repr(e))

    async def producer():
        send_dt = 1.0 / ARGS.send_fps
        try:
            while True:
                with compliance.LOCK:
                    st = compliance.LATEST
                if st is not None:
                    await ws.send(json.dumps(st, default=compliance._json_default))
                await asyncio.sleep(send_dt)
        except Exception as e:
            print("  producer ended:", repr(e))

    await asyncio.gather(consumer(), producer())
    print("  kids client disconnected")


# ── static file serving on the SAME port ────────────────────────────────
# The websockets library lets us answer plain HTTP requests before the
# upgrade handshake, so the UI and its media are served from this one
# process. No second terminal, no separate http.server.
import os, mimetypes, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ALLOWED_EXT = {".html", ".mp4", ".mp3", ".js", ".css", ".png", ".jpg",
               ".jpeg", ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2"}


def _safe_path(url_path):
    """Map a URL path to a file inside HERE, blocking directory escapes."""
    rel = url_path.lstrip("/")
    if rel == "" or rel.endswith("/"):
        rel = (rel + "kids_live.html")
    target = (HERE / rel).resolve()
    try:
        target.relative_to(HERE)          # must stay inside the app folder
    except ValueError:
        return None
    if target.is_file() and target.suffix.lower() in ALLOWED_EXT:
        return target
    return None


def _serve_static(url):
    """Return (status, headers, body) for a URL, or None to let the
    WebSocket upgrade proceed. Shared by both signature styles."""
    clean = url.split("?")[0]
    if clean.rstrip("/") in ("/ws-kids", "/ws"):
        return None
    target = _safe_path(clean)
    if target is None:
        return (404, [("Content-Type", "text/plain")], b"Not found\n")
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    data = target.read_bytes()
    headers = [
        ("Content-Type", ctype),
        ("Content-Length", str(len(data))),
        ("Cache-Control", "no-cache"),
        ("Access-Control-Allow-Origin", "*"),
        ("Accept-Ranges", "bytes"),
    ]
    return (200, headers, data)


# websockets >=14 calls process_request(connection, request); older versions
# call process_request(path, request_headers). Support both so this runs on
# whatever version is installed.
async def process_request(*a):
    try:
        if len(a) == 2 and hasattr(a[1], "path"):
            # new API: (connection, request)
            connection, request = a
            result = _serve_static(request.path)
            if result is None:
                return None
            status, headers, body = result
            return connection.respond(status, body.decode("latin-1")) \
                if hasattr(connection, "respond") else (status, headers, body)
        else:
            # old API: (path, request_headers)
            path = a[0]
            headers = a[1] if len(a) > 1 else {}
            if "Upgrade" in headers or "upgrade" in headers:
                return None
            return _serve_static(path)
    except Exception as e:
        print("  static serve error:", repr(e))
        return (500, [("Content-Type", "text/plain")], b"error\n")


async def main_async():
    async with websockets.serve(ws_handler, "0.0.0.0", ARGS.port,
                                max_size=2 ** 22, ping_interval=None,
                                process_request=process_request):
        host = f"localhost:{ARGS.port}"
        ui_ok = (HERE / "kids_live.html").is_file()
        media_ok = (HERE / "kids_media").is_dir()
        print(f"\nWASH-IT EDUCATION server ready (single terminal)")
        print(f"  Open the UI ->  http://{host}/kids_live.html")
        print(f"  WebSocket    ->  ws://{host}/ws-kids  (already filled in for you)")
        if not ui_ok:
            print("  WARNING: kids_live.html not found next to kids_server.py")
        if not media_ok:
            print("  WARNING: kids_media/ folder not found next to kids_server.py")
        print("  compliance mode is untouched; run server.py separately if needed\n")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WASH-IT education-mode backend")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--send-fps", type=float, default=20.0)
    ap.add_argument("--frame-width", type=int, default=480)
    ap.add_argument("--jpeg-quality", type=int, default=70)
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
    # education-only knobs
    ap.add_argument("--teach-timeout", type=float, default=15.0,
                    help="seconds before a narration hold self-releases if the "
                         "UI never reports teach_done")
    ap.add_argument("--no-teach-redo", action="store_true",
                    help="do not pause for narration during the redo round")
    ARGS = ap.parse_args()

    print("=" * 60)
    print("WASH-IT EDUCATION backend - loading model")
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

    # the camera thread in server.py drives whatever engine is installed here,
    # so the education engine reuses that loop without editing it
    compliance.ENGINE = KidsEngine(model, mean, std, landmarker, cfg,
                                   teach_timeout=ARGS.teach_timeout,
                                   teach_redo=(not ARGS.no_teach_redo))

    t = threading.Thread(target=compliance.camera_loop,
                         args=(ARGS.cam, ARGS.flip), daemon=True)
    t.start()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        compliance.STOP.set()
