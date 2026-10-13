"""
live_guided.py — Full guided WHO hand-hygiene protocol
==========================================================================
Phases: PRIME -> SOAP -> GETSET -> RUN(7x8s) -> REMEDIATE -> RINSE -> DONE

This version: a step's GREEN approval is revealed only AFTER its 8s slot
lapses (slot end + reveal_grace), not the moment it's first detected. During
the slot you see a neutral progress readout ("performing Step k ... Xs held")
so it's clear the model is accumulating evidence across the window rather than
stamping a single glimpse. reveal_grace (~3s) lets the trailing-window grace
detection settle so a real step doesn't briefly flash red at the 8s mark.

Other behaviour (activity gate, latency-aware ordered scoring, soap/rinse
gates, redo loop) is unchanged.

USAGE:
    python live_guided.py
    python live_guided.py --flip --motion-thresh 0.005
    python live_guided.py --reveal-grace 3

Controls:  [r] restart   [q] quit
"""

import argparse, time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

from test_continuous_sequence_local import (
    HERE, MP_TASK, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF, MAX_HANDS,
    TARGET_FPS, SEQ_LEN, NUM_CLASSES, STEP_NAMES, STEP_SHORT,
    mediapipe_to_raw, build_features, load_model,
)
from live_sequence_stream import STEP_BGR
from live_webcam import (
    open_camera, BUF_LEN, SMOOTH_WIN, VIS_WINDOW, PRESENT_L, PRESENT_R,
)

PRIME_COUNTDOWN = 3.0
MOTION_WIN = 15


def put(f, t, org, sc, col, th=2):
    cv2.putText(f, t, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY GATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion(raw_window):
    deltas = []
    for t in range(1, len(raw_window)):
        f0, f1 = raw_window[t - 1], raw_window[t]
        for off, flag in ((0, PRESENT_L), (63, PRESENT_R)):
            if f0[flag] > 0.5 and f1[flag] > 0.5:
                a = f0[off:off + 63].reshape(21, 3)[:, :2]
                b = f1[off:off + 63].reshape(21, 3)[:, :2]
                deltas.append(float(np.linalg.norm(b - a, axis=1).mean()))
    return float(np.mean(deltas)) if deltas else 0.0


def activity_state(raw_buf, min_visibility, motion_thresh):
    L = raw_buf[:, PRESENT_L] > 0.5
    R = raw_buf[:, PRESENT_R] > 0.5
    vis = (L | R)[-VIS_WINDOW:]
    if not len(vis) or vis.mean() < min_visibility:
        return "NO_HANDS", 0.0
    motion = compute_motion(raw_buf[-MOTION_WIN:])
    if motion < motion_thresh:
        return "IDLE", motion
    return "WASHING", motion


# ─────────────────────────────────────────────────────────────────────────────
# STEP SCORING — latency-aware
# ─────────────────────────────────────────────────────────────────────────────

def step_verdict(records, k, L, grace, min_detect, win_period):
    """Independent per-step verdict over [k*L, (k+1)*L + grace]."""
    krecs = [(t, c) for (t, p, c) in records if p == k and k * L <= t <= (k + 1) * L + grace]
    det_s = len(krecs) * win_period
    if det_s >= min_detect:
        return True, float(np.mean([c for _, c in krecs])), det_s, k
    slot = [p for (t, p, c) in records if k * L <= t <= (k + 1) * L]
    mode = int(np.bincount(slot, minlength=NUM_CLASSES).argmax()) if slot else -1
    conf = float(np.mean([c for _, c in krecs])) if krecs else 0.0
    return False, conf, det_s, mode


def score_ordered(records, n_steps, L, grace, min_detect, win_period):
    results = [None] * n_steps
    cursor = -1.0
    for k in range(n_steps):
        lo = max(k * L, cursor)
        hi = (k + 1) * L + grace
        krecs = [(t, c) for (t, p, c) in records if p == k and lo <= t <= hi]
        det_s = len(krecs) * win_period
        if det_s >= min_detect:
            conf = float(np.mean([c for _, c in krecs]))
            results[k] = {"passed": True, "retry": False, "conf": conf, "det_s": det_s, "mode": k}
            cursor = max(t for t, _ in krecs)
        else:
            slot = [p for (t, p, c) in records if k * L <= t <= (k + 1) * L]
            mode = int(np.bincount(slot, minlength=n_steps).argmax()) if slot else -1
            conf = float(np.mean([c for _, c in krecs])) if krecs else 0.0
            results[k] = {"passed": False, "retry": False, "conf": conf, "det_s": det_s, "mode": mode}
            cursor = max(cursor, k * L)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_strip(frame, n_steps, cur_slot, results):
    """results[i] is None (pending), or a dict with passed/conf/retry."""
    h, w = frame.shape[:2]
    bw = w // n_steps
    for i in range(n_steps):
        x0 = i * bw
        r = results[i]
        if r is None:
            col, label = (50, 50, 50), f"{STEP_SHORT[i]} .."
        elif r["passed"]:
            col = STEP_BGR[i]
            label = f"{STEP_SHORT[i]} {'r' if r.get('retry') else ''}{r['conf']:.2f}"
        else:
            col, label = (40, 40, 160), f"{STEP_SHORT[i]} X"
        cv2.rectangle(frame, (x0, h - 38), (x0 + bw - 2, h - 4), col, -1)
        if i == cur_slot:
            cv2.rectangle(frame, (x0, h - 38), (x0 + bw - 2, h - 4), (255, 255, 255), 2)
        put(frame, label, (x0 + 6, h - 14), 0.5, (255, 255, 255), 2)


def banner(frame, hgt=96):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (frame.shape[1], hgt), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)


def draw_run(frame, expected, cur_pred, state, motion, slot_left, total_left,
             disp, cur_slot, n_steps, fps, span, cur_det_s, min_detect):
    w = frame.shape[1]
    banner(frame)
    put(frame, f"DO NOW  ->  Step {expected+1}: {STEP_NAMES[expected].split(': ')[-1]}",
        (15, 34), 0.85, STEP_BGR[expected], 2)
    if state == "NO_HANDS":
        put(frame, "NO HANDS - show your hands", (15, 66), 0.6, (60, 160, 235), 2)
    elif state == "IDLE":
        put(frame, "IDLE HANDS - start rubbing", (15, 66), 0.6, (40, 200, 235), 2)
    elif cur_pred == expected:
        put(frame, f"performing Step {expected+1}  ({cur_det_s:.1f}s held)",
            (15, 66), 0.6, (210, 200, 120), 2)
    else:
        seen = STEP_SHORT[cur_pred] if cur_pred >= 0 else "-"
        put(frame, f"...seeing {seen}, do Step {expected+1}", (15, 66), 0.6, (200, 200, 200), 1)
    put(frame, f"slot {slot_left:2.0f}s", (w - 235, 30), 0.7, (230, 230, 230), 2)
    put(frame, f"total {total_left:2.0f}s", (w - 235, 54), 0.5, (200, 200, 200), 1)
    mcol = (150, 220, 150) if fps >= 12 else (90, 140, 235)
    put(frame, f"{fps:4.1f}fps win{span:3.1f}s m{motion:.3f}", (w - 235, 78), 0.45, mcol, 1)
    draw_strip(frame, n_steps, cur_slot, disp)
    return frame


def draw_remediate(frame, k, remaining, state, cur_pred, queue, idx, results):
    w = frame.shape[1]
    banner(frame)
    put(frame, f"REDO  Step {k+1}: {STEP_NAMES[k].split(': ')[-1]}", (15, 34), 0.85, STEP_BGR[k], 2)
    if state == "NO_HANDS":
        put(frame, "show your hands", (15, 66), 0.6, (60, 160, 235), 2)
    elif state == "IDLE":
        put(frame, f"start doing Step {k+1} now", (15, 66), 0.6, (40, 200, 235), 2)
    elif cur_pred == k:
        put(frame, "detecting - hold it", (15, 66), 0.6, (210, 200, 120), 2)
    else:
        seen = STEP_SHORT[cur_pred] if cur_pred >= 0 else "-"
        put(frame, f"...do Step {k+1} (seeing {seen})", (15, 66), 0.6, (200, 200, 200), 1)
    put(frame, f"{remaining:2.0f}s", (w - 110, 34), 0.9, (230, 230, 230), 2)
    pend = " ".join(STEP_SHORT[q] for q in queue[idx:])
    put(frame, f"to redo: {pend}", (w - 235, 64), 0.5, (200, 200, 200), 1)
    draw_strip(frame, NUM_CLASSES, k, results)
    return frame


def draw_center(frame, msg, sub, col=(240, 240, 240), shade=0.45):
    h, w = frame.shape[:2]
    ov = frame.copy(); cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, shade, frame, 1 - shade, 0, frame)
    put(frame, msg, (w // 2 - 220, h // 2 - 10), 1.3, col, 3)
    put(frame, sub, (w // 2 - 250, h // 2 + 40), 0.7, (210, 210, 210), 2)
    return frame


def draw_phase_gate(frame, title, remaining, state):
    if state == "WASHING":
        sub, col = "good - keep your hands moving", (140, 220, 140)
    elif state == "IDLE":
        sub, col = "rub your hands together", (40, 200, 235)
    else:
        sub, col = "show your hands to the camera", (60, 160, 235)
    return draw_center(frame, f"{title}  {remaining:0.0f}s", sub, col=col)


def draw_done(frame, results, n_steps, soap_done, rinse_done):
    h, w = frame.shape[:2]
    ov = frame.copy(); cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    passed    = sum(1 for r in results if r["passed"])
    corrected = sum(1 for r in results if r["passed"] and r.get("retry"))
    ok = soap_done and rinse_done and passed == n_steps
    put(frame, "COMPLIANT" if ok else "NON-COMPLIANT",
        (w // 2 - 210, h // 2 - 44), 1.5, (60, 220, 60) if ok else (60, 60, 230), 3)
    soap_c  = (90, 220, 90) if soap_done else (60, 60, 230)
    rinse_c = (90, 220, 90) if rinse_done else (60, 60, 230)
    put(frame, f"soap {'OK' if soap_done else 'X'}", (w // 2 - 235, h // 2 + 2), 0.7, soap_c, 2)
    put(frame, f"steps {passed}/{n_steps}", (w // 2 - 95, h // 2 + 2), 0.7, (230, 230, 230), 2)
    put(frame, f"rinse {'OK' if rinse_done else 'X'}", (w // 2 + 80, h // 2 + 2), 0.7, rinse_c, 2)
    if corrected:
        put(frame, f"({corrected} corrected on retry)", (w // 2 - 120, h // 2 + 34),
            0.55, (200, 200, 120), 1)
    put(frame, "[r] restart   [q] quit", (w // 2 - 150, h // 2 + 66), 0.6, (190, 190, 190), 2)
    draw_strip(frame, n_steps, -1, results)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

def fresh(n_steps):
    return {
        "raw_buf":  deque(maxlen=BUF_LEN),
        "prob_buf": deque(maxlen=SMOOTH_WIN),
        "records":  [], "results": None,
        "revealed": [False] * n_steps, "live_pass": [False] * n_steps,
        "live_conf": [0.0] * n_steps,
        "pred": -1, "conf": 0.0, "state": "NO_HANDS", "motion": 0.0,
        "soap_s": 0.0, "rinse_s": 0.0, "soap_done": False, "rinse_done": False,
        "missed_queue": [], "retry_idx": 0, "retry_start": None,
        "retry_hit": 0, "retry_conf_sum": 0.0,
        "si": 0, "phase": "PRIME", "go_since": None, "start": None, "cur_slot": 0,
    }


def print_summary(s, n_steps, soap_seconds, rinse_seconds, fps, span):
    print("\n  " + "-" * 60)
    print("  FULL HAND-HYGIENE RESULT  (soap + 7 steps + rinse)")
    print(f"  effective rate: {fps:.1f} fps   model window: {span:.1f}s")
    print("  " + "-" * 60)
    print(f"  SOAP   : {'DONE' if s['soap_done'] else 'NOT DONE':<10} "
          f"({s['soap_s']:.1f}s active / {soap_seconds:.0f}s needed)")
    print(f"  {'Step':<22}  {'Result':>12}  {'Conf':>6}")
    for k in range(n_steps):
        r = s["results"][k]
        res = ("CORRECTED" if (r["passed"] and r.get("retry"))
               else "PERFORMED" if r["passed"] else "MISSED")
        extra = "" if r["passed"] else (
            f"  (saw {STEP_SHORT[r['mode']]})" if r["mode"] >= 0 and r["mode"] != k else "")
        print(f"  {STEP_NAMES[k]:<22}  {res:>12}  {r['conf']:>6.2f}{extra}")
    print(f"  RINSE  : {'DONE' if s['rinse_done'] else 'NOT DONE':<10} "
          f"({s['rinse_s']:.1f}s active / {rinse_seconds:.0f}s needed)")
    passed    = sum(1 for r in s["results"] if r["passed"])
    corrected = sum(1 for r in s["results"] if r["passed"] and r.get("retry"))
    overall = s["soap_done"] and s["rinse_done"] and passed == n_steps
    print(f"\n  soap {'OK' if s['soap_done'] else 'X'}  |  steps {passed}/{n_steps}"
          f"{f' ({corrected} on retry)' if corrected else ''}  |  "
          f"rinse {'OK' if s['rinse_done'] else 'X'}")
    print(f"  ->  {'COMPLIANT' if overall else 'NON-COMPLIANT'}")


def run_guided(cam_id=2, flip=False, stride=4, step_seconds=8.0,
               grace=8.0, min_detect=1.2, min_visibility=0.2, motion_thresh=0.006,
               soap_seconds=5.0, rinse_seconds=5.0, retry_seconds=8.0, reveal_grace=3.0):
    n = NUM_CLASSES
    L = step_seconds
    total_s = L * n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 62)
    print(f"FULL WHO HAND-HYGIENE — soap + {n}x{L:.0f}s + redo + rinse")
    print("=" * 62)
    print(f"  Camera {cam_id}  flip={flip}  device={device}")
    print(f"  reveal_grace={reveal_grace}s  soap={soap_seconds}s rinse={rinse_seconds}s "
          f"retry={retry_seconds}s motion_thresh={motion_thresh}")

    print(f"\n[1] Loading model...");  model, mean, std, val_f1 = load_model(device)
    print(f"  val_f1 = {val_f1:.4f}")
    print(f"\n[2] Opening camera...");  cap = open_camera(cam_id)
    if cap is None:
        print(f"  x Could not open camera {cam_id}. Try --cam 0/1/2."); return

    sample_dt  = 1.0 / TARGET_FPS
    prog_start = time.time()
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )

    s = fresh(n)
    last_sample = time.time()
    fps_t, fps_n, fps_val = time.time(), 0, TARGET_FPS
    fails = 0
    print(f"\n[3] Put your hands in front of the camera to begin.\n")

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails > 60: print("  x Lost camera feed."); break
                continue
            fails = 0
            if flip: frame = cv2.flip(frame, 1)
            now = time.time()
            span = SEQ_LEN / max(fps_val, 1e-3)
            win_period = stride / max(fps_val, 1e-3)

            if s["phase"] != "DONE" and now - last_sample >= sample_dt:
                dt_sample = now - last_sample
                last_sample = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm.detect_for_video(img, int((now - prog_start) * 1000))
                s["raw_buf"].append(mediapipe_to_raw(res))

                raw_stack = np.stack(s["raw_buf"])
                s["state"], s["motion"] = activity_state(raw_stack, min_visibility, motion_thresh)
                washing = (s["state"] == "WASHING")
                buffering = len(s["raw_buf"]) < SEQ_LEN

                fps_n += 1
                if now - fps_t >= 0.5:
                    fps_val = fps_n / (now - fps_t); fps_t, fps_n = now, 0

                if dt_sample < 0.5 and washing:
                    if s["phase"] == "SOAP":  s["soap_s"]  += dt_sample
                    if s["phase"] == "RINSE": s["rinse_s"] += dt_sample

                if s["phase"] in ("RUN", "REMEDIATE") and washing and not buffering \
                        and (s["si"] % stride == 0):
                    window = build_features(np.stack(s["raw_buf"]))[-SEQ_LEN:]
                    x = torch.from_numpy(((window - mean) / std).astype(np.float32)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = F.softmax(model(x)[0], dim=-1).squeeze(0).cpu().numpy()
                    s["prob_buf"].append(p)
                    p_sm = np.mean(s["prob_buf"], axis=0)
                    s["pred"], s["conf"] = int(np.argmax(p_sm)), float(np.max(p_sm))
                    if s["phase"] == "RUN":
                        s["records"].append((now - s["start"], s["pred"], s["conf"]))
                    else:
                        q = s["missed_queue"]
                        if s["retry_idx"] < len(q) and s["pred"] == q[s["retry_idx"]]:
                            s["retry_hit"] += 1; s["retry_conf_sum"] += s["conf"]
                s["si"] += 1

            # ── phase machine ───────────────────────────────────────────────
            if s["phase"] == "PRIME":
                if (len(s["raw_buf"]) >= SEQ_LEN) and (s["state"] != "NO_HANDS"):
                    s["phase"] = "SOAP"; print("  --- SOAP: wet hands & apply soap ---"); hud = frame
                else:
                    msg = "buffering..." if len(s["raw_buf"]) < SEQ_LEN else "SHOW HANDS"
                    hud = draw_center(frame, msg, "place both hands in view")

            elif s["phase"] == "SOAP":
                rem = max(0.0, soap_seconds - s["soap_s"])
                if rem <= 0:
                    s["soap_done"] = True; s["phase"] = "GETSET"; s["go_since"] = None
                    print("  --- soap OK ---"); hud = frame
                else:
                    hud = draw_phase_gate(frame, "APPLY SOAP", rem, s["state"])

            elif s["phase"] == "GETSET":
                if s["go_since"] is None: s["go_since"] = now
                left = PRIME_COUNTDOWN - (now - s["go_since"])
                if left <= 0:
                    s["phase"] = "RUN"; s["start"] = now; s["prob_buf"].clear()
                    print("  --- START (Step 1) ---"); hud = frame
                else:
                    hud = draw_center(frame, f"GET READY  {int(left)+1}", "begin Step 1 (Rub Palms)")

            elif s["phase"] == "RUN":
                elapsed = now - s["start"]
                # reveal each step's verdict only after its slot lapses (+ grace)
                for k in range(n):
                    if not s["revealed"][k] and elapsed >= (k + 1) * L + reveal_grace:
                        ok, cf, _, _ = step_verdict(s["records"], k, L, grace, min_detect, win_period)
                        s["revealed"][k] = True; s["live_pass"][k] = ok; s["live_conf"][k] = cf
                        print(f"   Step {k+1}: {'PERFORMED' if ok else 'missed'} (reveal)")
                if elapsed >= total_s:
                    s["results"] = score_ordered(s["records"], n, L, grace, min_detect, win_period)
                    for k in range(n):            # sync any unrevealed (last step) to final
                        s["revealed"][k] = True
                        s["live_pass"][k] = s["results"][k]["passed"]
                        s["live_conf"][k] = s["results"][k]["conf"]
                    missed = [k for k in range(n) if not s["results"][k]["passed"]]
                    if missed:
                        s["missed_queue"] = missed; s["retry_idx"] = 0; s["retry_start"] = now
                        s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                        s["pred"] = -1; s["prob_buf"].clear(); s["phase"] = "REMEDIATE"
                        print(f"  --- REDO missed steps: {[k+1 for k in missed]} ---")
                    else:
                        s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---")
                    hud = frame
                else:
                    slot = min(n - 1, int(elapsed // L)); s["cur_slot"] = slot
                    disp = [({"passed": s["live_pass"][k], "conf": s["live_conf"][k], "retry": False}
                             if s["revealed"][k] else None) for k in range(n)]
                    cur_det_s = sum(1 for (t, p, c) in s["records"]
                                    if p == slot and slot * L <= t <= elapsed) * win_period
                    hud = draw_run(frame, slot, s["pred"], s["state"], s["motion"],
                                   L - (elapsed - slot * L), total_s - elapsed,
                                   disp, slot, n, fps_val, span, cur_det_s, min_detect)

            elif s["phase"] == "REMEDIATE":
                q = s["missed_queue"]
                if s["retry_idx"] >= len(q):
                    s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---"); hud = frame
                else:
                    k = q[s["retry_idx"]]
                    det_s = s["retry_hit"] * win_period
                    elapsed_r = now - s["retry_start"]
                    advanced = False
                    if det_s >= min_detect:
                        s["results"][k].update(passed=True, retry=True,
                            conf=s["retry_conf_sum"] / max(1, s["retry_hit"]))
                        print(f"   Step {k+1} ({STEP_NAMES[k].split(': ')[-1]}): CORRECTED")
                        advanced = True
                    elif elapsed_r >= retry_seconds:
                        print(f"   Step {k+1} ({STEP_NAMES[k].split(': ')[-1]}): still missed")
                        advanced = True
                    if advanced:
                        s["retry_idx"] += 1; s["retry_start"] = now
                        s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                        s["pred"] = -1; s["prob_buf"].clear()
                        if s["retry_idx"] >= len(q):
                            s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---")
                        hud = frame
                    else:
                        hud = draw_remediate(frame, k, max(0.0, retry_seconds - elapsed_r),
                                             s["state"], s["pred"], q, s["retry_idx"], s["results"])

            elif s["phase"] == "RINSE":
                rem = max(0.0, rinse_seconds - s["rinse_s"])
                if rem <= 0:
                    s["rinse_done"] = True; s["phase"] = "DONE"
                    print_summary(s, n, soap_seconds, rinse_seconds, fps_val, span); hud = frame
                else:
                    hud = draw_phase_gate(frame, "RINSE HANDS", rem, s["state"])

            else:  # DONE
                hud = draw_done(frame, s["results"], n, s["soap_done"], s["rinse_done"])

            cv2.imshow("WHO hand hygiene — guided", hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r"):
                print("\n  --- restart ---\n"); s = fresh(n); last_sample = time.time()

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full guided WHO hand-hygiene protocol")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--step-seconds", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=8.0)
    ap.add_argument("--min-detect", type=float, default=1.2)
    ap.add_argument("--min-visibility", type=float, default=0.2)
    ap.add_argument("--motion-thresh", type=float, default=0.006)
    ap.add_argument("--soap-seconds", type=float, default=5.0)
    ap.add_argument("--rinse-seconds", type=float, default=5.0)
    ap.add_argument("--retry-seconds", type=float, default=8.0)
    ap.add_argument("--reveal-grace", type=float, default=3.0,
                    help="seconds after a step's slot end before its verdict is "
                         "revealed (lets latency grace settle; set 0 to reveal at the 8s mark)")
    args = ap.parse_args()
    run_guided(cam_id=args.cam, flip=args.flip, stride=args.stride,
               step_seconds=args.step_seconds, grace=args.grace, min_detect=args.min_detect,
               min_visibility=args.min_visibility, motion_thresh=args.motion_thresh,
               soap_seconds=args.soap_seconds, rinse_seconds=args.rinse_seconds,
               retry_seconds=args.retry_seconds, reveal_grace=args.reveal_grace)
