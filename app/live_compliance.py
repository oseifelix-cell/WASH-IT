"""
live_compliance.py — Live WHO-step compliance check (no fixed timeline)
==========================================================================
Streams a handwash video into the v6 model frame-by-frame (like a webcam),
but unlike live_sequence_stream.py it makes NO assumption about which steps
are present or how long each lasts. This is the real deployment use case:
watch the wash, then report which of the 7 WHO steps the person actually
performed and which they SKIPPED.

Use this for videos where a step may be missing/out of order (e.g. the
"without step 2" clip). Use live_sequence_stream.py when you know all 7
steps are present in order and want timeline-aligned scoring.

A step only counts as "performed" if the model predicts it for a sustained
run (default >=1.0s), so a one-off flicker won't mask a genuine skip.

Reuses the proven pipeline from test_continuous_sequence_local.py and the
colour map from live_sequence_stream.py — both must be in the same folder.

USAGE:
    python live_compliance.py "without step 2.mp4"
    python live_compliance.py "without step 2.mp4" --display
    python live_compliance.py "without step 2.mp4" --no-realtime
"""

import argparse, time, math
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
from live_sequence_stream import STEP_BGR     # reuse the BGR colour map

BUF_LEN    = SEQ_LEN + 2
SMOOTH_WIN = 3


def draw_hud(frame, pred, conf, t_now, confirmed, buffering):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    if buffering:
        cv2.putText(frame, "buffering...", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    else:
        cv2.putText(frame, STEP_NAMES[pred], (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, STEP_BGR[pred], 2)
        cv2.putText(frame, f"conf {conf:0.2f}   t={t_now:0.1f}s", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    box_w = w // 7
    for i in range(7):
        x0 = i * box_w
        col = STEP_BGR[i] if confirmed[i] else (60, 60, 60)
        cv2.rectangle(frame, (x0, h - 34), (x0 + box_w - 2, h - 4), col, -1)
        tcol = (255, 255, 255) if confirmed[i] else (140, 140, 140)
        cv2.putText(frame, STEP_SHORT[i], (x0 + box_w // 2 - 12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcol, 2)
    return frame


def run_compliance(video_path, realtime=True, display=False,
                   stride=4, min_sustained_s=1.0):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  x Video not found: {video_path}"); return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 62)
    print("LIVE WHO COMPLIANCE CHECK  (v6 model — no timeline assumed)")
    print("=" * 62)
    print(f"  Video  : {video_path.name}")
    print(f"  Mode   : {'REAL-TIME' if realtime else 'fast'}"
          f"{'  + HUD' if display else ''}   device={device}")

    print(f"\n[1] Loading model...")
    model, mean, std, val_f1 = load_model(device)
    print(f"  val_f1 = {val_f1:.4f}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  x Could not open video"); return None
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total / native_fps
    interval   = max(1, int(round(native_fps / TARGET_FPS)))

    # sustained-run threshold (windows) so a flicker doesn't count as a step
    MIN_RUN = max(1, math.ceil(min_sustained_s * TARGET_FPS / stride))
    print(f"  Duration : {duration:.1f}s   "
          f"(a step counts as performed after ~{min_sustained_s:.1f}s "
          f"sustained = {MIN_RUN} windows)")
    print(f"  Warm-up  : {SEQ_LEN/TARGET_FPS:.1f}s before first prediction\n")

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )

    raw_buf  = deque(maxlen=BUF_LEN)
    prob_buf = deque(maxlen=SMOOTH_WIN)

    total_windows = [0] * NUM_CLASSES
    longest_run   = [0] * NUM_CLASSES
    first_t       = [None] * NUM_CLASSES
    last_t        = [None] * NUM_CLASSES
    run_step, run_len = -1, 0
    confirmed = [False] * NUM_CLASSES
    pred, conf, last_logged = -1, 0.0, -1

    print("  --- live step log -------------------------------------------")
    wall_start = time.time()
    fi = si = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % interval == 0:
                if realtime:
                    behind = (time.time() - wall_start) - si / TARGET_FPS
                    if behind < 0:
                        time.sleep(-behind)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm.detect_for_video(img, int(si / TARGET_FPS * 1000))
                raw_buf.append(mediapipe_to_raw(res))

                t_now     = si / TARGET_FPS
                buffering = len(raw_buf) < SEQ_LEN

                if not buffering and (si % stride == 0):
                    window = build_features(np.stack(raw_buf))[-SEQ_LEN:]
                    norm   = ((window - mean) / std).astype(np.float32)
                    x = torch.from_numpy(norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = F.softmax(model(x)[0], dim=-1).squeeze(0).cpu().numpy()
                    prob_buf.append(p)
                    p_sm = np.mean(prob_buf, axis=0)
                    pred, conf = int(np.argmax(p_sm)), float(np.max(p_sm))

                    mid_t = max(0.0, t_now - (SEQ_LEN / 2) / TARGET_FPS)
                    total_windows[pred] += 1
                    if first_t[pred] is None:
                        first_t[pred] = mid_t
                    last_t[pred] = mid_t

                    run_len = run_len + 1 if pred == run_step else 1
                    run_step = pred
                    longest_run[pred] = max(longest_run[pred], run_len)
                    if longest_run[pred] >= MIN_RUN:
                        confirmed[pred] = True

                    if pred != last_logged:
                        print(f"   {t_now:5.1f}s  ->  {STEP_NAMES[pred]:<22} "
                              f"(conf {conf:.2f})")
                        last_logged = pred

                if display:
                    hud = draw_hud(frame.copy(), pred, conf, t_now,
                                   confirmed, buffering)
                    cv2.imshow("WHO handwash — compliance", hud)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n  (quit by user)"); break
                si += 1
            fi += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # ── WHO compliance verdict ───────────────────────────────────────────────
    performed = [longest_run[i] >= MIN_RUN for i in range(NUM_CLASSES)]
    skipped   = [i for i in range(NUM_CLASSES) if not performed[i]]

    print("  -------------------------------------------------------------")
    print(f"\n[2] WHO compliance checklist:")
    print(f"  {'Step':<22}  {'Status':>10}  {'Seen at':>12}  {'~Duration':>10}")
    print("  " + "-" * 60)
    for i in range(NUM_CLASSES):
        approx_dur = total_windows[i] * stride / TARGET_FPS
        if performed[i]:
            when   = f"{first_t[i]:4.0f}-{last_t[i]:4.0f}s"
            status = "DONE"
        elif total_windows[i] > 0:
            when   = f"~{first_t[i]:4.0f}s"
            status = "brief?"     # flickered but not sustained
        else:
            when, status = "-", "SKIPPED"
        print(f"  {STEP_NAMES[i]:<22}  {status:>10}  {when:>12}  "
              f"{approx_dur:>8.1f}s")

    print(f"\n  -- Verdict --------------------------------------")
    print(f"     Steps performed : {NUM_CLASSES - len(skipped)}/{NUM_CLASSES}")
    if skipped:
        print(f"     NON-COMPLIANT — missing:")
        for i in skipped:
            note = ("  (model saw it briefly but not long enough — review)"
                    if total_windows[i] > 0 else "")
            print(f"        - {STEP_NAMES[i]}{note}")
    else:
        print(f"     COMPLIANT — all 7 WHO steps performed.")

    print(f"\n  (Times are approximate — each prediction reflects the trailing "
          f"~{SEQ_LEN/TARGET_FPS:.1f}s window.)")

    return {
        "video": str(video_path), "duration_s": duration,
        "performed": [STEP_NAMES[i] for i in range(NUM_CLASSES) if performed[i]],
        "skipped":   [STEP_NAMES[i] for i in skipped],
        "compliant": len(skipped) == 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live WHO compliance check on a video")
    ap.add_argument("video", nargs="?", default=str(HERE / "without step 2.mp4"))
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-sustained", type=float, default=1.0,
                    help="seconds a step must be sustained to count (default 1.0)")
    args = ap.parse_args()

    run_compliance(
        args.video,
        realtime=args.realtime,
        display=args.display,
        stride=args.stride,
        min_sustained_s=args.min_sustained,
    )
