"""
pi_benchmark.py — Run this FIRST on the Raspberry Pi, before building the
full standalone device. It answers the one question that determines every
other decision: how fast can this Pi actually run MediaPipe + the model?

CAMERA: this version targets the Pi Camera Module (CSI ribbon) via
picamera2/libcamera — confirmed via `v4l2-ctl --list-devices` showing
`unicam (platform:fe801000.csi)` and no USB UVC webcam present. If you
later add a USB webcam instead, ask for the cv2.VideoCapture variant.

Copy these files into the same folder on the Pi before running:
    pi_benchmark.py
    test_continuous_sequence_local.py
    model_tcn_attention.py
    student_best.pt
    norm_stats.npz
    hand_landmarker.task

USAGE:
    python3 pi_benchmark.py                # 30s test at 640x480
    python3 pi_benchmark.py --secs 20
    python3 pi_benchmark.py --width 480 --height 360   # try a lower res if slow

WHAT IT MEASURES:
    1. Raw camera read FPS (no processing)
    2. MediaPipe hand-landmark FPS (the usual bottleneck on Pi CPU)
    3. Model inference latency
    4. Recommends a TARGET_FPS / stride combination based on what it measures
"""

import argparse, time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
from picamera2 import Picamera2

from test_continuous_sequence_local import (
    HERE, MP_TASK, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF, MAX_HANDS,
    SEQ_LEN, mediapipe_to_raw, build_features, load_model,
)


def open_camera_pi(width, height):
    """Open the CSI camera via picamera2. Note: picamera2's 'RGB888' format
    is a historical misnomer -- it actually returns BGR-ordered arrays (for
    OpenCV compatibility), so we convert BGR->RGB before handing frames to
    MediaPipe, exactly like the cv2.VideoCapture path did on Windows."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let auto-exposure/white-balance settle
    return picam2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--secs", type=float, default=30.0)
    args = ap.parse_args()

    print("=" * 62)
    print("PI BENCHMARK — camera / MediaPipe / model throughput")
    print("=" * 62)
    print(f"  Device: CPU (Pi has no CUDA)")
    print(f"  Camera: Pi Camera Module (CSI, via picamera2)")
    print(f"  Resolution requested: {args.width}x{args.height}")

    try:
        cap = open_camera_pi(args.width, args.height)
    except Exception as e:
        print(f"  x Could not open the CSI camera: {e}")
        print(f"    Check the ribbon cable is seated correctly and try "
              f"`libcamera-hello --list-cameras` to confirm it's detected.")
        return
    print(f"  Camera opened.")

    # ── 1. Raw camera FPS ────────────────────────────────────────────────
    print("\n[1] Measuring raw camera read speed (5s)...")
    n, t0 = 0, time.time()
    while time.time() - t0 < 5.0:
        _ = cap.capture_array()
        n += 1
    cam_fps = n / (time.time() - t0)
    print(f"    Raw camera FPS: {cam_fps:.1f}")

    # ── 2. MediaPipe FPS ─────────────────────────────────────────────────
    print("\n[2] Measuring MediaPipe hand-landmark FPS (10s)...")
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )
    mp_times = []
    raw_buf = deque(maxlen=SEQ_LEN + 2)
    t_start = time.time()
    ts_ms = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while time.time() - t_start < 10.0:
            frame = cap.capture_array()   # BGR-ordered despite "RGB888" name
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.time()
            res = lm.detect_for_video(img, ts_ms)
            mp_times.append(time.time() - t0)
            raw_buf.append(mediapipe_to_raw(res))
            ts_ms += 33  # advance timestamp regardless of true elapsed time

        mp_fps = 1.0 / np.mean(mp_times)
        print(f"    MediaPipe FPS: {mp_fps:.1f}  "
              f"(avg {np.mean(mp_times)*1000:.1f}ms/frame, "
              f"max {np.max(mp_times)*1000:.1f}ms)")

        # ── 3. Model inference latency ──────────────────────────────────
        print("\n[3] Loading model and measuring inference latency...")
        model, mean, std, val_f1 = load_model("cpu")
        print(f"    val_f1 = {val_f1:.4f}")

        while len(raw_buf) < SEQ_LEN:
            frame = cap.capture_array()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lm.detect_for_video(img, ts_ms); ts_ms += 33
            raw_buf.append(mediapipe_to_raw(res))

        infer_times = []
        for _ in range(30):
            window = build_features(np.stack(raw_buf))[-SEQ_LEN:]
            norm = ((window - mean) / std).astype(np.float32)
            x = torch.from_numpy(norm).unsqueeze(0)
            t0 = time.time()
            with torch.no_grad():
                logits = model(x)[0]
                _ = F.softmax(logits, dim=-1)
            infer_times.append(time.time() - t0)
        infer_fps = 1.0 / np.mean(infer_times)
        print(f"    Model inference: {np.mean(infer_times)*1000:.1f}ms/window "
              f"({infer_fps:.0f} windows/sec possible)")

    cap.stop()

    # ── Recommendation ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RECOMMENDATION")
    print("=" * 62)
    pipeline_fps = min(cam_fps, mp_fps)
    print(f"  Bottleneck stage: {'MediaPipe' if mp_fps < cam_fps else 'camera'} "
          f"(~{pipeline_fps:.1f} FPS ceiling)")
    if pipeline_fps >= 12:
        print("  -> Close to desktop performance. Keep TARGET_FPS=15, stride=4 "
              "as-is and just confirm smoothness in the guided-flow test.")
    elif pipeline_fps >= 6:
        print(f"  -> Set TARGET_FPS={int(pipeline_fps)} in "
              f"test_continuous_sequence_local.py (both the extraction script "
              f"AND retrain-time value must match, OR keep 15 and accept "
              f"the model sees fewer real frames per window — test both).")
        print("  -> Increase stride to 6-8 so inference doesn't run every "
              "sampled frame, only the display updates each frame.")
    else:
        print("  -> Pipeline is too slow for smooth real-time at 640x480.")
        print("  -> Try --width 480 --height 360 (rerun this benchmark).")
        print("  -> Consider MediaPipe 'lite' hand landmark model variant, "
              "or reduce running_mode overhead further.")
    print(f"\n  Full report: camera={cam_fps:.1f}fps  mediapipe={mp_fps:.1f}fps "
          f" model={infer_fps:.0f}win/s")


if __name__ == "__main__":
    main()
