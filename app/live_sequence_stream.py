"""
live_sequence_stream.py — Simulate a LIVE camera feed from a video file
==========================================================================
Feeds a video into the v6 student model FRAME BY FRAME, in real time, the
same way a webcam would. Instead of extracting the whole video then sliding
a window over it (the offline approach), this keeps a rolling buffer of the
most recent 64 feature frames and runs the model on that trailing window as
each new frame arrives.

Why this is valid for "live":
  - The TCN is causal (CausalConv1d) — outputs depend only on past frames.
  - Each inference runs on a complete, fixed 64-frame window made entirely of
    frames at or before "now". The BiGRU's backward pass only looks back
    inside that window — never into the real future.
  - Cost: ~4.3s warm-up (you need 64 frames @15fps before the first
    prediction), and each prediction reflects the trailing ~4.3s.

MediaPipe runs in VIDEO mode with the SAME timestamps as the offline script,
so the extracted landmarks — and therefore the predictions — match your
offline run very closely.

This REUSES the feature pipeline + model loader from
test_continuous_sequence_local.py (must be in the same folder).

USAGE (from inside the bundle folder):
    python live_sequence_stream.py "all complete.mp4"
    python live_sequence_stream.py "all complete.mp4" --display      # OpenCV HUD window
    python live_sequence_stream.py "all complete.mp4" --no-realtime  # run as fast as possible
"""

import argparse, time, sys
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

# ── Reuse everything proven in the offline script ────────────────────────────
from test_continuous_sequence_local import (
    HERE, MP_TASK, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF, MAX_HANDS,
    TARGET_FPS, SEQ_LEN, NUM_CLASSES, DEFAULT_TOLERANCE_S,
    STEP_NAMES, STEP_SHORT, STEP_COLORS,
    mediapipe_to_raw, build_features, load_model,
    build_timeline, expected_step_at, score_against_timeline,
)

BUF_LEN     = SEQ_LEN + 2     # +2 so velocity/accel at window edges are exact
SMOOTH_WIN  = 3              # causal running-mean over last N probability vectors


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

STEP_BGR = [hex_to_bgr(c) for c in STEP_COLORS]


def draw_hud(frame, pred, conf, t_now, expected, detected, buffering):
    """Overlay current prediction + a 7-step progress strip onto the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top banner
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if buffering:
        cv2.putText(frame, "buffering...", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    else:
        color = STEP_BGR[pred]
        cv2.putText(frame, f"{STEP_NAMES[pred]}", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"conf {conf:0.2f}   t={t_now:0.1f}s", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # Bottom 7-step strip (lights up as each step gets detected)
    box_w = w // 7
    for i in range(7):
        x0 = i * box_w
        col = STEP_BGR[i] if detected[i] else (60, 60, 60)
        cv2.rectangle(frame, (x0, h - 34), (x0 + box_w - 2, h - 4), col, -1)
        is_exp = i in expected
        txt_col = (255, 255, 255) if (detected[i] or is_exp) else (150, 150, 150)
        cv2.putText(frame, STEP_SHORT[i], (x0 + box_w // 2 - 12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 2)
    return frame


def run_live(video_path, step_durations=None, tolerance=DEFAULT_TOLERANCE_S,
             realtime=True, display=False, stride=4):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  x Video not found: {video_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 62)
    print("LIVE STREAM SIMULATION  (v6 model)")
    print("=" * 62)
    print(f"  Video     : {video_path.name}")
    print(f"  Mode      : {'REAL-TIME paced' if realtime else 'fast-as-possible'}"
          f"{'  + HUD window' if display else ''}")
    print(f"  Device    : {device}")

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
    timeline   = build_timeline(duration, step_durations)

    print(f"  Duration  : {duration:.1f}s  ({total} frames @ {native_fps:.0f}FPS"
          f" -> sampling every {interval} to ~{TARGET_FPS}fps)")
    print(f"  Warm-up   : first prediction after {SEQ_LEN/TARGET_FPS:.1f}s "
          f"(need {SEQ_LEN} frames in buffer)\n")

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )

    raw_buf   = deque(maxlen=BUF_LEN)
    prob_buf  = deque(maxlen=SMOOTH_WIN)
    preds_list, mids_list = [], []
    detected  = [False] * NUM_CLASSES
    pred, conf = -1, 0.0
    last_logged = -1
    latencies = []
    lagged    = 0

    print("  --- live step log -------------------------------------------")
    wall_start = time.time()
    fi = si = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fi % interval == 0:
                # real-time pacing: make sampled frame si land at si/TARGET_FPS sec
                if realtime:
                    target = si / TARGET_FPS
                    behind = (time.time() - wall_start) - target
                    if behind < 0:
                        time.sleep(-behind)
                    elif behind > 0.05:
                        lagged += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm.detect_for_video(img, int(si / TARGET_FPS * 1000))
                raw_buf.append(mediapipe_to_raw(res))

                t_now    = si / TARGET_FPS
                expected = expected_step_at(t_now, timeline, tolerance)
                buffering = len(raw_buf) < SEQ_LEN

                if not buffering and (si % stride == 0):
                    t0 = time.time()
                    window = build_features(np.stack(raw_buf))[-SEQ_LEN:]
                    norm   = ((window - mean) / std).astype(np.float32)
                    x = torch.from_numpy(norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)[0]
                        p = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    prob_buf.append(p)
                    p_sm = np.mean(prob_buf, axis=0)
                    pred = int(np.argmax(p_sm)); conf = float(p_sm[pred])
                    latencies.append(time.time() - t0)

                    detected[pred] = True
                    mid = max(0, si - SEQ_LEN // 2)   # window-centre, as offline
                    preds_list.append(pred); mids_list.append(mid)

                    if pred != last_logged:
                        print(f"   {t_now:5.1f}s  ->  {STEP_NAMES[pred]:<22} "
                              f"(conf {conf:.2f})")
                        last_logged = pred

                if display:
                    hud = draw_hud(frame.copy(), pred, conf, t_now,
                                   expected, detected, buffering)
                    cv2.imshow("WHO handwash — live", hud)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n  (quit by user)")
                        break
                si += 1
            fi += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    wall = time.time() - wall_start

    # ── Performance ─────────────────────────────────────────────────────────
    print("  -------------------------------------------------------------")
    print(f"\n[2] Stream performance:")
    print(f"  Wall-clock time   : {wall:.1f}s   (video is {duration:.1f}s)")
    if latencies:
        ms = np.array(latencies) * 1000
        print(f"  Inference latency : {ms.mean():.1f}ms avg, {ms.max():.1f}ms max "
              f"({len(latencies)} predictions)")
        print(f"  Sustainable rate  : ~{1000/ms.mean():.0f} predictions/sec")
    if realtime:
        kept = "yes" if lagged < max(1, si // 20) else f"NO ({lagged} laggy frames)"
        print(f"  Real-time kept up : {kept}")

    # ── Did streaming still catch all 7 steps? (same scoring as offline) ─────
    if not preds_list:
        print("\n  No predictions produced (video too short?)."); return None

    score = score_against_timeline(np.array(preds_list), np.array(mids_list),
                                   timeline, tolerance)
    print(f"\n[3] Did the live stream catch every step?")
    print(f"  Overall window accuracy (within tolerance): "
          f"{score['overall_accuracy']:.1%}\n")
    print(f"  {'Step':<22}  {'Slot':>12}  {'Detected?':>10}  {'Hit/Total':>12}")
    print("  " + "-" * 62)
    missed = []
    for start, end, idx in timeline:
        det = score["step_detected"][idx]
        if not det: missed.append(idx)
        print(f"  {STEP_NAMES[idx]:<22}  {start:4.0f}-{end:4.0f}s  "
              f"{'YES' if det else 'MISSED':>10}  "
              f"{score['step_window_count'][idx]:>4d} /"
              f"{score['step_total_windows'][idx]:<5d}")

    print(f"\n  Steps detected : {7 - len(missed)}/7")
    if missed:
        print(f"  Steps MISSED   : {', '.join(STEP_NAMES[i] for i in missed)}")
    else:
        print(f"  All 7 WHO steps detected live, in order.")

    return {
        "video": str(video_path), "duration_s": duration,
        "wall_clock_s": wall, "overall_accuracy": score["overall_accuracy"],
        "steps_detected": 7 - len(missed),
        "steps_missed": [STEP_NAMES[i] for i in missed],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live-stream a video through the v6 model")
    ap.add_argument("video", nargs="?", default=str(HERE / "all complete.mp4"))
    ap.add_argument("--display", action="store_true",
                    help="Show an OpenCV HUD window")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="Process as fast as possible instead of real-time pacing")
    ap.add_argument("--stride", type=int, default=4,
                    help="Run inference every N sampled frames (default 4)")
    ap.add_argument("--tolerance", type=float, default=3.0)
    args = ap.parse_args()

    run_live(
        args.video,
        step_durations=[8, 8, 8, 8, 8, 8, 8],   # same 56s layout as your offline run
        tolerance=args.tolerance,
        realtime=args.realtime,
        display=args.display,
        stride=args.stride,
    )
