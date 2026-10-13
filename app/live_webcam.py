"""
live_webcam.py — Real-time WHO compliance monitor from a live camera
==========================================================================
Same streaming model as live_compliance.py, but the source is a LIVE camera
(DroidCam at id 2 by default) instead of a video file. There's no ground
truth and no end — you control sessions from the keyboard:

    [r]  reset / start a new wash session (clears the checklist)
    [q]  quit (prints the final checklist for the current session)

The HUD shows the current predicted step, a 7-box strip that lights up as
each step is confirmed, the session timer, and which steps are still missing.

Reuses the proven pipeline from test_continuous_sequence_local.py and the
colour map from live_sequence_stream.py — both must be in the same folder.

USAGE:
    python live_webcam.py                 # DroidCam at id 2
    python live_webcam.py --cam 2 --flip  # try --flip if Left/Right look swapped
    python live_webcam.py --cam 0         # if id 2 isn't the right device

DROIDCAM TIPS (Windows):
  - Start the DroidCam client and connect the phone BEFORE running this.
  - If the camera won't open, try --cam 0 / 1 / 2, or run again (DSHOW
    backend is tried first automatically).
"""

import argparse, time, math, sys
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

BUF_LEN    = SEQ_LEN + 2
SMOOTH_WIN = 3
VIS_WINDOW = max(6, int(round(0.8 * TARGET_FPS)))   # ~0.8s of frames to judge presence
PRESENT_L  = 126   # left-hand presence flag in the raw 128-d frame
PRESENT_R  = 127   # right-hand presence flag


def open_camera(cam_id):
    """Open the camera, preferring the DirectShow backend on Windows."""
    for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "default")]:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            print(f"  Camera {cam_id} opened ({name} backend)")
            return cap
        cap.release()
    return None


def draw_hud(frame, pred, conf, t_now, confirmed, buffering, fps, hands_visible):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    if buffering:
        cv2.putText(frame, "buffering...", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    elif not hands_visible:
        cv2.putText(frame, "NO HANDS", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 160, 235), 2)
        cv2.putText(frame, "show your hands to the camera", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(frame, STEP_NAMES[pred], (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, STEP_BGR[pred], 2)
        cv2.putText(frame, f"conf {conf:0.2f}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    n_done = sum(confirmed)
    in_who = 40 <= t_now <= 60
    timer_col = (90, 230, 90) if in_who else (230, 230, 230)
    cv2.putText(frame, f"{t_now:4.0f}s   {n_done}/7", (w - 200, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_col, 2)
    cv2.putText(frame, "[r] reset  [q] quit", (w - 230, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"{fps:4.1f} fps", (w - 110, h - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # bottom 7-step strip
    box_w = w // 7
    for i in range(7):
        x0 = i * box_w
        col = STEP_BGR[i] if confirmed[i] else (55, 55, 55)
        cv2.rectangle(frame, (x0, h - 34), (x0 + box_w - 2, h - 4), col, -1)
        tcol = (255, 255, 255) if confirmed[i] else (140, 140, 140)
        cv2.putText(frame, STEP_SHORT[i], (x0 + box_w // 2 - 12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcol, 2)

    if n_done == 7:
        cv2.putText(frame, "COMPLIANT", (w // 2 - 110, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 220, 60), 3)
    return frame


def fresh_session():
    return {
        "raw_buf":  deque(maxlen=BUF_LEN),
        "prob_buf": deque(maxlen=SMOOTH_WIN),
        "total":    [0] * NUM_CLASSES,
        "longest":  [0] * NUM_CLASSES,
        "first_t":  [None] * NUM_CLASSES,
        "last_t":   [None] * NUM_CLASSES,
        "confirmed":[False] * NUM_CLASSES,
        "run_step": -1, "run_len": 0,
        "pred": -1, "conf": 0.0, "last_logged": -1,
        "hands_visible": False,
        "si": 0, "start": time.time(),
    }


def print_checklist(s, stride, min_run, min_sustained_s):
    performed = [s["longest"][i] >= min_run for i in range(NUM_CLASSES)]
    skipped   = [i for i in range(NUM_CLASSES) if not performed[i]]
    print(f"\n  WHO compliance checklist  (sustained >= {min_sustained_s:.1f}s):")
    print(f"  {'Step':<22}  {'Status':>8}  {'Seen at':>12}  {'~Dur':>7}")
    print("  " + "-" * 56)
    for i in range(NUM_CLASSES):
        dur = s["total"][i] * stride / TARGET_FPS
        if performed[i]:
            when, status = f"{s['first_t'][i]:4.0f}-{s['last_t'][i]:4.0f}s", "DONE"
        elif s["total"][i] > 0:
            when, status = f"~{s['first_t'][i]:4.0f}s", "brief?"
        else:
            when, status = "-", "SKIPPED"
        print(f"  {STEP_NAMES[i]:<22}  {status:>8}  {when:>12}  {dur:>5.1f}s")
    print(f"\n  Verdict: {NUM_CLASSES - len(skipped)}/7 steps  ->  "
          f"{'COMPLIANT' if not skipped else 'NON-COMPLIANT'}")
    if skipped:
        for i in skipped:
            note = "  (saw briefly — review)" if s["total"][i] > 0 else ""
            print(f"     missing: {STEP_NAMES[i]}{note}")


def run_webcam(cam_id=2, flip=False, stride=4, min_sustained_s=1.0,
               min_visibility=0.2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 62)
    print("LIVE WEBCAM WHO COMPLIANCE MONITOR")
    print("=" * 62)
    print(f"  Camera id : {cam_id}   flip={flip}   device={device}")

    print(f"\n[1] Loading model...")
    model, mean, std, val_f1 = load_model(device)
    print(f"  val_f1 = {val_f1:.4f}")

    print(f"\n[2] Opening camera (make sure DroidCam is connected)...")
    cap = open_camera(cam_id)
    if cap is None:
        print(f"  x Could not open camera {cam_id}. Is DroidCam running?")
        print(f"    Try a different id:  python live_webcam.py --cam 0")
        return

    MIN_RUN = max(1, math.ceil(min_sustained_s * TARGET_FPS / stride))
    sample_dt = 1.0 / TARGET_FPS
    prog_start = time.time()

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )

    s = fresh_session()
    last_sample = 0.0
    fps_t, fps_n, fps_val = time.time(), 0, 0.0
    fails = 0

    print(f"\n[3] Live. Wash your hands.  [r]=reset session  [q]=quit\n")
    print("  --- live step log -------------------------------------------")

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails > 60:
                    print("  x Lost the camera feed."); break
                continue
            fails = 0
            if flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            buffering = len(s["raw_buf"]) < SEQ_LEN

            # sample for the model at ~TARGET_FPS by wall clock
            if now - last_sample >= sample_dt:
                last_sample = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((now - prog_start) * 1000)     # monotonic across resets
                res = lm.detect_for_video(img, ts_ms)
                s["raw_buf"].append(mediapipe_to_raw(res))
                buffering = len(s["raw_buf"]) < SEQ_LEN

                # ── visibility gate: are hands actually in frame? ──────────
                raw_stack = np.stack(s["raw_buf"])
                present = (raw_stack[:, PRESENT_L] > 0.5) | \
                          (raw_stack[:, PRESENT_R] > 0.5)
                recent = present[-VIS_WINDOW:]
                hands_visible = bool(len(recent) and recent.mean() >= min_visibility)
                if s["hands_visible"] and not hands_visible:
                    print(f"   {now-s['start']:5.1f}s  ..  (no hands — paused)")
                    s["last_logged"] = -1
                s["hands_visible"] = hands_visible

                # classify ONLY when hands are visible
                if hands_visible and not buffering and (s["si"] % stride == 0):
                    window = build_features(np.stack(s["raw_buf"]))[-SEQ_LEN:]
                    norm   = ((window - mean) / std).astype(np.float32)
                    x = torch.from_numpy(norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = F.softmax(model(x)[0], dim=-1).squeeze(0).cpu().numpy()
                    s["prob_buf"].append(p)
                    p_sm = np.mean(s["prob_buf"], axis=0)
                    pred, conf = int(np.argmax(p_sm)), float(np.max(p_sm))
                    s["pred"], s["conf"] = pred, conf

                    t_mid = max(0.0, (now - s["start"]) - (SEQ_LEN/2)/TARGET_FPS)
                    s["total"][pred] += 1
                    if s["first_t"][pred] is None:
                        s["first_t"][pred] = t_mid
                    s["last_t"][pred] = t_mid
                    s["run_len"] = s["run_len"] + 1 if pred == s["run_step"] else 1
                    s["run_step"] = pred
                    s["longest"][pred] = max(s["longest"][pred], s["run_len"])
                    if s["longest"][pred] >= MIN_RUN:
                        s["confirmed"][pred] = True
                    if pred != s["last_logged"]:
                        print(f"   {now-s['start']:5.1f}s  ->  "
                              f"{STEP_NAMES[pred]:<22} (conf {conf:.2f})")
                        s["last_logged"] = pred
                s["si"] += 1

            # fps estimate
            fps_n += 1
            if now - fps_t >= 0.5:
                fps_val = fps_n / (now - fps_t); fps_t, fps_n = now, 0

            hud = draw_hud(frame, s["pred"], s["conf"], now - s["start"],
                           s["confirmed"], buffering, fps_val, s["hands_visible"])
            cv2.imshow("WHO handwash — live monitor", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("  --- session reset ---")
                s = fresh_session()

    cap.release()
    cv2.destroyAllWindows()

    print("  -------------------------------------------------------------")
    print_checklist(s, stride, MIN_RUN, min_sustained_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live webcam WHO compliance monitor")
    ap.add_argument("--cam", type=int, default=2, help="camera id (DroidCam = 2)")
    ap.add_argument("--flip", action="store_true",
                    help="horizontally flip the feed (try if Left/Right seem swapped)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-sustained", type=float, default=1.0)
    ap.add_argument("--min-visibility", type=float, default=0.2,
                    help="fraction of the recent ~0.8s that must show a hand "
                         "(raise toward 0.4 to be stricter)")
    args = ap.parse_args()

    run_webcam(cam_id=args.cam, flip=args.flip,
               stride=args.stride, min_sustained_s=args.min_sustained,
               min_visibility=args.min_visibility)
