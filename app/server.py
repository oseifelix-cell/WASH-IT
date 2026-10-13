"""
server.py — WASH-IT WebSocket backend
==========================================================================
Wraps the SAME guided pipeline as live_guided.py (activity gate, latency-
aware ordered scoring, slot-end reveal, soap/rinse gates, redo loop) but
instead of an OpenCV window it:
  - opens the camera server-side (DroidCam id 2),
  - JPEG-encodes each frame to base64 (downscaled),
  - serves a WebSocket on ws://<host>:8000/ws,
  - emits one JSON status message per tick,
  - accepts the "reset" command the UI sends on connect.

The 7 step indices (0-6) map 1:1 onto the UI's STEPS array (same WHO order).

Run:
    pip install websockets
    python server.py                 # cam 2, port 8000
    python server.py --flip --port 8000

Then in the UI's "Live Camera Mode" box use:  ws://localhost:8000/ws

Message schema (every message has `frame` + `phase`):
  phase: PRIME | SOAP | GETSET | RUN | REMEDIATE | RINSE | DONE
  PRIME    : buffering, buffer_fill, visibility_ok, guidance
  SOAP     : phase_prompt, phase_remaining, state, guidance
  GETSET   : countdown, prompt
  RUN      : inference, current_step, predicted_step, confidence, progress,
             cur_det_s, completed_steps[], slot_left, total_left,
             is_compliant, struggling, visibility_ok, guidance, state
  REMEDIATE: redo_step, redo_remaining, redo_queue[], predicted_step,
             confidence, detecting, completed_steps[], state, guidance
  RINSE    : phase_prompt, phase_remaining, state, guidance
  DONE     : finished=true, results[], soap_done, rinse_done, score, grade,
             steps_done, corrected, order, duration_s
"""

import argparse, asyncio, base64, json, threading, time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

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
        self._kids_go = False        # UI signals "teach clip finished, start detecting"

    def kids_go(self):
        self._kids_go = True

    # encode a downscaled JPEG -> base64 string
    def _frame_b64(self, frame):
        w = self.cfg["frame_width"]
        h = int(frame.shape[0] * w / frame.shape[1])
        small = cv2.resize(frame, (w, h))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["jpeg_q"]])
        return base64.b64encode(buf).decode("ascii") if ok else None

    def _final_payload(self):
        s = self.s; n = NUM_CLASSES

        if self.cfg["mode"] == "kids":
            per, learned, beaten = [], 0, 0
            for k in range(n):
                r = s["kids_results"][k]
                lrn = bool(r and r.get("learned"))
                mercy = bool(r and r.get("mercy"))
                if r and r.get("defeated"): beaten += 1
                if lrn: learned += 1
                per.append({"defeated": bool(r and r.get("defeated")), "learned": lrn,
                            "mercy": mercy, "conf": (r or {}).get("conf", 0.0)})
            score = round(100 * learned / n)
            end = self.rinse_start if self.rinse_start is not None else (self.done_time or time.time())
            dur = end - (self.run_start or self.session_start)
            return dict(finished=True, mode="kids", kids_results=per,
                        germs_beaten=beaten, germs_learned=learned, score=score,
                        soap_done=s["soap_done"], rinse_done=s["rinse_done"], duration_s=round(dur))

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

    def step(self, frame, now, frame_b64=None):
        s = self.s
        c = self.cfg
        n = NUM_CLASSES
        L = c["step_seconds"]
        total_s = L * n
        # if the frame arrived already JPEG-encoded (remote camera), pass it
        # straight through instead of re-encoding it on this machine.
        status = {"frame": frame_b64 if frame_b64 is not None else self._frame_b64(frame),
                  "phase": s["phase"], "inference": False, "mode": c["mode"]}

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

            # both-hands presence over the recent window (raw idx 126=L, 127=R)
            _recent = raw_stack[-6:]
            _both = (_recent[:, 126] > 0.5) & (_recent[:, 127] > 0.5)
            s["both"] = bool(_both.mean() >= c["min_visibility"]) if len(_recent) else False

            self.fps_n += 1
            if now - self.fps_t >= 0.5:
                self.fps_val = self.fps_n / (now - self.fps_t); self.fps_t, self.fps_n = now, 0

            if dt_sample < 0.5 and washing:
                soap_ok = s["both"] if c["mode"] == "kids" else True   # kids: both hands
                if s["phase"] == "SOAP" and soap_ok:  s["soap_s"]  += dt_sample
                if s["phase"] == "RINSE" and soap_ok: s["rinse_s"] += dt_sample

            if s["phase"] in ("RUN", "REMEDIATE", "KIDS_RUN") and washing and not buffering \
                    and (s["si"] % c["stride"] == 0):
                window = build_features(np.stack(s["raw_buf"]))[-SEQ_LEN:]
                x = torch.from_numpy(((window - self.mean) / self.std).astype(np.float32)).unsqueeze(0)
                x = x.to(getattr(self, "device", "cpu"))
                with torch.no_grad():
                    p = F.softmax(self.model(x)[0], dim=-1).squeeze(0).cpu().numpy()
                s["prob_buf"].append(p)
                p_sm = np.mean(s["prob_buf"], axis=0)
                s["pred"], s["conf"] = int(np.argmax(p_sm)), float(np.max(p_sm))
                did_infer = True
                if s["phase"] == "RUN":
                    s["records"].append((now - s["start"], s["pred"], s["conf"]))
                elif s["phase"] == "KIDS_RUN":
                    if s.get("kids_sub") == "detect" and s["pred"] == s["kids_step"] \
                            and s["conf"] >= c["kids_min_conf"]:
                        s["kids_hit"] += 1; s["kids_conf_sum"] += s["conf"]
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
            hands_ok = s["both"] if c["mode"] == "kids" else (s["state"] != "NO_HANDS")
            if (not buffering) and hands_ok:
                s["phase"] = "SOAP"
            status.update(buffering=buffering,
                          buffer_fill=min(1.0, len(s["raw_buf"]) / SEQ_LEN),
                          visibility_ok=(s["state"] != "NO_HANDS"), both_hands=s["both"],
                          guidance=("Show both hands to begin" if not hands_ok
                                    else "Hold still, starting…"))

        elif phase == "SOAP":
            rem = max(0.0, c["soap_seconds"] - s["soap_s"])
            if rem <= 0:
                s["soap_done"] = True; s["phase"] = "GETSET"; s["go_since"] = None
            status.update(phase_prompt="Wet hands & apply soap",
                          phase_remaining=round(rem, 1), soap_total=c["soap_seconds"],
                          state=s["state"], both_hands=s["both"],
                          guidance=gate_hint(s["state"], "rubbing soap in"))

        elif phase == "GETSET":
            if s["go_since"] is None: s["go_since"] = now
            left = PRIME_COUNTDOWN - (now - s["go_since"])
            if left <= 0:
                s["start"] = now; s["prob_buf"].clear(); self.run_start = now
                if c["mode"] == "kids":
                    s["phase"] = "KIDS_RUN"
                    s["kids_step"] = 0; s["kids_sub"] = "teach"
                    s["kids_detect_start"] = None; s["kids_hit"] = 0; s["kids_conf_sum"] = 0.0
                    s["kids_defeated"] = []; s["kids_results"] = [None] * n
                    s["kids_attempt"] = 1; s["kids_saw_wrong"] = False; s["kids_miss_kind"] = "nothing"
                    self._kids_go = False
                else:
                    s["phase"] = "RUN"
            status.update(countdown=max(1, int(left) + 1),
                          prompt=("Get ready — here comes the first germ!" if c["mode"] == "kids"
                                  else "Get ready — Step 1"))

        elif phase == "KIDS_RUN":
            k = s["kids_step"]
            if k >= n:
                s["phase"] = "RINSE"
            else:
                if s["kids_sub"] == "teach":
                    if self._kids_go:                      # UI: teach clip finished
                        self._kids_go = False
                        s["kids_sub"] = "detect"; s["kids_attempt"] = 1
                        s["kids_detect_start"] = now; s["kids_hit"] = 0; s["kids_conf_sum"] = 0.0
                        s["kids_saw_wrong"] = False
                        s["pred"] = -1; s["prob_buf"].clear()
                    else:
                        status.update(inference=did_infer, kids_phase="teach", kids_step=k,
                                      germs_beaten=list(s["kids_defeated"]), state=s["state"])

                elif s["kids_sub"] == "retry_wait":
                    if self._kids_go:                      # UI: wrong clip finished -> attempt 2
                        self._kids_go = False
                        s["kids_sub"] = "detect"; s["kids_attempt"] = 2
                        s["kids_detect_start"] = now; s["kids_hit"] = 0; s["kids_conf_sum"] = 0.0
                        s["kids_saw_wrong"] = False
                        s["pred"] = -1; s["prob_buf"].clear()
                    else:
                        status.update(inference=did_infer, kids_phase="retry", kids_step=k,
                                      kids_miss_kind=s["kids_miss_kind"], kids_attempt=2,
                                      germs_beaten=list(s["kids_defeated"]), state=s["state"])

                if s["kids_sub"] == "detect":
                    elapsed_a = now - s["kids_detect_start"]
                    hold_s = s["kids_hit"] * win_period
                    needed = c["kids_hold"]
                    # remember if they did a DIFFERENT step's motion (for wrong_alert vs wrong_try)
                    if s["state"] == "WASHING" and s["pred"] >= 0 and s["pred"] != k:
                        s["kids_saw_wrong"] = True

                    def _advance(mercy, learned, conf):
                        s["kids_results"][k] = {"defeated": True, "mercy": mercy,
                                                "learned": learned, "conf": round(float(conf), 2)}
                        s["kids_defeated"].append(k)
                        print(f"   KIDS: germ {k+1} {'MERCY (not learned)' if mercy else 'BEATEN'} hold={s['kids_hit']*win_period:.1f}/{c['kids_hold']:.0f}s attempt={s['kids_attempt']}")
                        s["kids_step"] += 1; s["kids_sub"] = "teach"; s["kids_attempt"] = 1
                        s["kids_detect_start"] = None; s["kids_hit"] = 0; s["kids_conf_sum"] = 0.0
                        s["pred"] = -1; s["prob_buf"].clear()
                        status.update(inference=did_infer, kids_phase="defeated", kids_step=k,
                                      kids_mercy=mercy, kids_learned=learned,
                                      kids_miss_kind=s.get("kids_miss_kind", "nothing"),
                                      germs_beaten=list(s["kids_defeated"]), state=s["state"])
                        if s["kids_step"] >= n:
                            s["phase"] = "RINSE"

                    if hold_s >= needed:                    # got it
                        _advance(False, True, s["kids_conf_sum"] / max(1, s["kids_hit"]))
                    elif elapsed_a >= c["kids_attempt_seconds"]:
                        if s["kids_attempt"] >= 2:          # second miss -> mercy pass (not learned)
                            s["kids_miss_kind"] = "wrong" if s["kids_saw_wrong"] else "nothing"
                            _advance(True, False, 0.0)
                        else:                               # first miss -> wrong clip, then attempt 2
                            s["kids_miss_kind"] = "wrong" if s["kids_saw_wrong"] else "nothing"
                            s["kids_sub"] = "retry_wait"
                            s["pred"] = -1; s["prob_buf"].clear()
                            print(f"   KIDS: germ {k+1} attempt 1 missed ({s['kids_miss_kind']})")
                            status.update(inference=did_infer, kids_phase="retry", kids_step=k,
                                          kids_miss_kind=s["kids_miss_kind"], kids_attempt=2,
                                          germs_beaten=list(s["kids_defeated"]), state=s["state"])
                    else:                                    # attempt in progress
                        status.update(
                            inference=did_infer, kids_phase="detect", kids_step=k,
                            kids_attempt=s["kids_attempt"],
                            detecting=(s["state"] == "WASHING" and s["pred"] == k),
                            predicted_step=s["pred"], confidence=round(s["conf"], 3),
                            hold_s=round(hold_s, 1), needed=round(needed, 1),
                            progress=min(1.0, hold_s / max(needed, 1e-3)),
                            attempt_left=round(max(0.0, c["kids_attempt_seconds"] - elapsed_a), 1),
                            struggling=(s["state"] == "WASHING" and s["pred"] >= 0 and s["pred"] != k),
                            visibility_ok=(s["state"] != "NO_HANDS"),
                            buffering=(len(s["raw_buf"]) < SEQ_LEN),
                            buffer_fill=min(1.0, len(s["raw_buf"]) / SEQ_LEN),
                            germs_beaten=list(s["kids_defeated"]), state=s["state"])

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
            rlen = c["kids_rinse_seconds"] if c["mode"] == "kids" else c["rinse_seconds"]
            rem = max(0.0, rlen - s["rinse_s"])
            if rem <= 0:
                s["rinse_done"] = True; s["phase"] = "DONE"; self.done_time = now
            status.update(phase_prompt="Rinse hands under water",
                          phase_remaining=round(rem, 1), rinse_total=rlen,
                          state=s["state"], both_hands=s["both"],
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
LATEST = {"phase": "PRIME", "frame": None, "inference": False}
RESET = threading.Event()
STOP = threading.Event()
KIDS_GO = threading.Event()      # UI signals a teach clip finished (kids mode)
SET_MODE = threading.Event()     # UI requests a mode switch
MODE_REQ = {"mode": None}

# remote-camera inbox (Path B: the Pi captures, this machine infers)
INBOX = {"bgr": None, "b64": None, "seq": 0}
INBOX_LOCK = threading.Lock()
CAM_CONNECTED = threading.Event()


def push_remote_frame(b64_jpeg):
    """Called from the websocket thread when the Pi sends a frame."""
    try:
        buf = np.frombuffer(base64.b64decode(b64_jpeg), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
    except Exception:
        return
    with INBOX_LOCK:
        INBOX["bgr"] = bgr
        INBOX["b64"] = b64_jpeg
        INBOX["seq"] += 1


def remote_loop(flip):
    """Process the newest frame from the Pi. Stale frames are dropped, so a slow
    model never builds a backlog — it always works on the freshest image."""
    global LATEST
    print("  waiting for the Pi camera client to connect...")
    last_seq = -1
    while not STOP.is_set():
        with INBOX_LOCK:
            seq = INBOX["seq"]
            bgr = INBOX["bgr"]
            b64 = INBOX["b64"]
        if bgr is None or seq == last_seq:
            time.sleep(0.005)          # nothing new yet
            continue
        last_seq = seq
        if flip:
            bgr = cv2.flip(bgr, 1)
            b64 = None                 # flipped: must re-encode for the UI
        if RESET.is_set():
            ENGINE.reset(); RESET.clear(); print("  -- session reset --")
        if KIDS_GO.is_set():
            ENGINE.kids_go(); KIDS_GO.clear()
        if SET_MODE.is_set():
            m = MODE_REQ["mode"]
            if m: ENGINE.cfg["mode"] = m; print(f"  -- mode -> {m} --")
            ENGINE.reset(); SET_MODE.clear()
        status = ENGINE.step(bgr, time.time(), frame_b64=b64)
        with LOCK:
            LATEST = status


def camera_loop(cam_id, flip):
    global LATEST
    cap = open_camera(cam_id)
    if cap is None:
        print(f"  x Could not open camera {cam_id}. Try --cam 0/1/2."); STOP.set(); return
    print(f"  camera {cam_id} open — streaming")
    fails = 0
    while not STOP.is_set():
        ret, frame = cap.read()
        if not ret:
            fails += 1
            if fails > 60: print("  x lost camera feed"); break
            time.sleep(0.01); continue
        fails = 0
        if flip: frame = cv2.flip(frame, 1)
        if RESET.is_set():
            ENGINE.reset(); RESET.clear(); print("  -- session reset --")
        if KIDS_GO.is_set():
            ENGINE.kids_go(); KIDS_GO.clear()
        if SET_MODE.is_set():
            m = MODE_REQ["mode"]
            if m: ENGINE.cfg["mode"] = m; print(f"  -- mode -> {m} --")
            ENGINE.reset(); SET_MODE.clear()
        status = ENGINE.step(frame, time.time())
        with LOCK:
            LATEST = status
    cap.release()


def _json_default(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    return str(o)


async def ws_handler(ws, *args):
    """Two kinds of client:
       - the Pi camera client: sends {"role":"camera","frame":"<b64 jpeg>"} messages
       - the browser UI: sends "reset", receives the status stream
    """
    role = "ui"

    async def consumer():
        nonlocal role
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    msg = msg.decode(errors="ignore")
                stripped = msg.lstrip()
                if stripped.startswith("{"):
                    try:
                        d = json.loads(msg)
                    except Exception:
                        continue
                    if d.get("role") == "camera":
                        if role != "camera":
                            role = "camera"
                            CAM_CONNECTED.set()
                            print("  camera client connected (remote frames)")
                        if d.get("frame"):
                            push_remote_frame(d["frame"])
                        continue
                    if "reset" in str(d.get("cmd", "")).lower():
                        RESET.set()
                    if str(d.get("cmd", "")) == "start_detect":
                        KIDS_GO.set()
                    if str(d.get("cmd", "")) == "set_mode":
                        m = str(d.get("mode", "")).lower()
                        if m in ("compliance", "kids"):
                            MODE_REQ["mode"] = m; SET_MODE.set()
                    continue
                if "reset" in msg.lower():
                    RESET.set()
        except Exception as e:
            print("  consumer ended:", repr(e))
        finally:
            if role == "camera":
                CAM_CONNECTED.clear()
                print("  camera client disconnected")

    async def producer():
        # only UI clients get the status stream; the camera client doesn't need it
        send_dt = 1.0 / ARGS.send_fps
        try:
            await asyncio.sleep(0.2)          # let the role handshake land
            while True:
                if role == "camera":
                    await asyncio.sleep(0.5)
                    continue
                with LOCK:
                    st = LATEST
                if st is not None:
                    await ws.send(json.dumps(st, default=_json_default))
                await asyncio.sleep(send_dt)
        except Exception as e:
            print("  producer ended:", repr(e))

    print("  client connected")
    await asyncio.gather(consumer(), producer())
    print(f"  {role} client disconnected")


async def main_async():
    async with websockets.serve(ws_handler, "0.0.0.0", ARGS.port, max_size=2 ** 22,
                                ping_interval=None):
        # find this machine's LAN IP so the Pi knows where to point
        ip = "127.0.0.1"
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        except Exception:
            pass
        print(f"\nWASH-IT server ready")
        print(f"  local  :  ws://localhost:{ARGS.port}/ws")
        print(f"  LAN    :  ws://{ip}:{ARGS.port}/ws     <-- use this on the Pi")
        if ARGS.camera_source == "remote":
            print(f"\n  On the Pi run:")
            print(f"    ~/washit/bin/python pi_camera_client.py --server ws://{ip}:{ARGS.port}/ws")
            print(f"  Then open the UI on the Pi and set its address box to:")
            print(f"    ws://{ip}:{ARGS.port}/ws")
        print()
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WASH-IT WebSocket backend")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--send-fps", type=float, default=20.0)
    ap.add_argument("--frame-width", type=int, default=480)
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--step-seconds", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=8.0)
    ap.add_argument("--min-detect", type=float, default=1.2)
    ap.add_argument("--kids-min-conf", type=float, default=0.5,
                    help="kids: min model confidence for a frame to count toward beating a germ "
                         "(raise if wrong motions get credited; lower if right ones do not)")
    ap.add_argument("--min-visibility", type=float, default=0.2)
    ap.add_argument("--motion-thresh", type=float, default=0.006)
    ap.add_argument("--soap-seconds", type=float, default=5.0)
    ap.add_argument("--rinse-seconds", type=float, default=5.0)
    ap.add_argument("--kids-rinse-seconds", type=float, default=10.0,
                    help="kids mode: seconds of both-hands rinsing before the score")
    ap.add_argument("--retry-seconds", type=float, default=8.0)
    ap.add_argument("--reveal-grace", type=float, default=3.0)
    ap.add_argument("--camera-source", choices=["local", "remote"], default="local",
                    help="local = this machine's camera; remote = frames pushed by "
                         "pi_camera_client.py (Path B: Pi captures, this box infers)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                    help="inference device (auto uses CUDA when available)")
    ap.add_argument("--mode", choices=["compliance", "kids"], default="compliance",
                    help="compliance = timed WHO audit; kids = teach-until-correct game")
    ap.add_argument("--kids-hold", type=float, default=6.0,
                    help="seconds of ACCUMULATED correct motion to beat a germ "
                         "(enforces WHO ~duration; a single pose can't win)")
    ap.add_argument("--kids-attempt-seconds", type=float, default=10.0,
                    help="length of each attempt window (kids mode)")
    ap.add_argument("--kids-eased-hold", type=float, default=0.6,
                    help="(reserved) eased hold threshold")
    ap.add_argument("--kids-ease-after", type=float, default=20.0,
                    help="(reserved) legacy easing timer")
    ap.add_argument("--kids-mercy-after", type=float, default=40.0,
                    help="(reserved) legacy mercy timer")
    ARGS = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if ARGS.device == "auto" else ARGS.device

    print("=" * 60)
    print("WASH-IT backend — loading model")
    print("=" * 60)
    model, mean, std, val_f1 = load_model(dev)
    print(f"  val_f1 = {val_f1:.4f}   device = {dev}")
    print(f"  camera source: {ARGS.camera_source}   mode: {ARGS.mode}")

    cfg = dict(stride=ARGS.stride, step_seconds=ARGS.step_seconds, grace=ARGS.grace,
               min_detect=ARGS.min_detect, min_visibility=ARGS.min_visibility,
               kids_min_conf=ARGS.kids_min_conf,
               motion_thresh=ARGS.motion_thresh, soap_seconds=ARGS.soap_seconds,
               rinse_seconds=ARGS.rinse_seconds, kids_rinse_seconds=ARGS.kids_rinse_seconds,
               retry_seconds=ARGS.retry_seconds,
               reveal_grace=ARGS.reveal_grace, frame_width=ARGS.frame_width,
               jpeg_q=ARGS.jpeg_quality, mode=ARGS.mode,
               kids_hold=ARGS.kids_hold, kids_eased_hold=ARGS.kids_eased_hold,
               kids_ease_after=ARGS.kids_ease_after, kids_mercy_after=ARGS.kids_mercy_after,
               kids_attempt_seconds=ARGS.kids_attempt_seconds)

    lm_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS, min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF, min_tracking_confidence=MP_TRK_CONF)
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(lm_opts)

    ENGINE = Engine(model, mean, std, landmarker, cfg)
    ENGINE.device = dev

    if ARGS.camera_source == "remote":
        t = threading.Thread(target=remote_loop, args=(ARGS.flip,), daemon=True)
    else:
        t = threading.Thread(target=camera_loop, args=(ARGS.cam, ARGS.flip), daemon=True)
    t.start()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        STOP.set()
