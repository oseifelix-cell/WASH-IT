"""
pi_missing_step_test.py — Does the checklist correctly flag a SKIPPED step?
==========================================================================
Unlike pi_5fps_test.py, this does NOT assume all 7 WHO steps occur in
sequence -- it makes no assumptions about which steps are present at all.
It slides a window across the whole video (retimed to the Pi's real
throughput: TARGET_FPS=5, SEQ_LEN=22 by default, matching what worked in
the last test) and tracks, per step, how long the model sustained a
confident detection -- exactly the same "confirmed" logic live_webcam.py
uses for the real-time compliance checklist.

The point: feed it a video where a step was deliberately left out, and see
whether the checklist correctly reports that step as SKIPPED rather than
hallucinating it (a false "step was performed" is worse for a compliance
device than a missed one -- it would let genuinely non-compliant
handwashing pass as compliant).

USAGE:
    python3 pi_missing_step_test.py "without step 2.mp4"
    python3 pi_missing_step_test.py "without step 2.mp4" --target-fps 5 --seq-len 22
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import test_continuous_sequence_local as tcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--seq-len", type=int, default=22)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-sustained-s", type=float, default=1.0,
                     help="Minimum continuous detection time to count a "
                          "step as CONFIRMED (matches live_webcam.py default)")
    ap.add_argument("--smooth-win", type=int, default=3)
    args = ap.parse_args()

    tcs.TARGET_FPS = args.target_fps
    tcs.SEQ_LEN = args.seq_len
    tcs.STRIDE = args.stride

    print(f">>> TARGET_FPS={tcs.TARGET_FPS}  SEQ_LEN={tcs.SEQ_LEN}  "
          f"STRIDE={tcs.STRIDE}  (window = {tcs.SEQ_LEN/tcs.TARGET_FPS:.1f}s)")
    print(f">>> min_sustained_s={args.min_sustained_s} "
          f"(a step must hold this long continuously to count as CONFIRMED)\n")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"  x Video not found: {video_path}")
        return

    device = "cpu"
    print("[1] Loading model...")
    model, mean, std, val_f1 = tcs.load_model(device)
    print(f"    val_f1 = {val_f1:.4f}")

    print("\n[2] Extracting MediaPipe landmarks (this is the slow part)...")
    base, duration = tcs.extract_base(video_path)
    if base is None:
        print("  x Could not extract frames"); return

    print("\n[3] Building features + running continuous sliding-window inference...")
    feat = tcs.build_features(base)
    norm = ((feat - mean) / std).astype(np.float32)
    T = norm.shape[0]

    if T < tcs.SEQ_LEN:
        pad = np.tile(norm[-1:], (tcs.SEQ_LEN - T, 1))
        norm = np.concatenate([norm, pad], axis=0)
        T = tcs.SEQ_LEN

    prob_hist = []   # smoothed prob vector per window
    times = []       # window-centre time in seconds
    from collections import deque
    smooth_buf = deque(maxlen=args.smooth_win)

    with torch.no_grad():
        for start in range(0, T - tcs.SEQ_LEN + 1, tcs.STRIDE):
            w = norm[start:start + tcs.SEQ_LEN]
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            logits = model(x)[0]
            p = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            smooth_buf.append(p)
            p_sm = np.mean(smooth_buf, axis=0)
            prob_hist.append(p_sm)
            mid = start + tcs.SEQ_LEN // 2
            times.append(mid / tcs.TARGET_FPS)

    preds = [int(np.argmax(p)) for p in prob_hist]
    confs = [float(p[pred]) for p, pred in zip(prob_hist, preds)]

    # ── Track sustained runs per class, same logic as live_webcam.py ────────
    win_period = tcs.STRIDE / tcs.TARGET_FPS   # seconds between inference windows
    MIN_RUN = max(1, int(round(args.min_sustained_s / win_period)))

    total = [0] * tcs.NUM_CLASSES
    longest = [0] * tcs.NUM_CLASSES
    first_t = [None] * tcs.NUM_CLASSES
    last_t = [None] * tcs.NUM_CLASSES
    run_step, run_len = -1, 0

    for t, pred in zip(times, preds):
        total[pred] += 1
        if first_t[pred] is None:
            first_t[pred] = t
        last_t[pred] = t
        run_len = run_len + 1 if pred == run_step else 1
        run_step = pred
        longest[pred] = max(longest[pred], run_len)

    confirmed = [longest[i] >= MIN_RUN for i in range(tcs.NUM_CLASSES)]

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n  Video duration: {duration:.1f}s   Windows evaluated: {len(preds)}")
    print(f"  MIN_RUN threshold: {MIN_RUN} consecutive windows "
          f"(~{MIN_RUN*win_period:.1f}s)\n")

    print("=" * 70)
    print("CHECKLIST")
    print("=" * 70)
    print(f"  {'Step':<22}  {'Status':>10}  {'Seen at':>14}  "
          f"{'Longest run':>12}  {'Total hits':>10}")
    print("  " + "-" * 66)
    for i in range(tcs.NUM_CLASSES):
        dur = longest[i] * win_period
        if confirmed[i]:
            when = f"{first_t[i]:5.1f}-{last_t[i]:5.1f}s"
            status = "CONFIRMED"
        elif total[i] > 0:
            when = f"~{first_t[i]:5.1f}s"
            status = "brief?"
        else:
            when = "-"
            status = "SKIPPED"
        print(f"  {tcs.STEP_NAMES[i]:<22}  {status:>10}  {when:>14}  "
              f"{dur:>10.1f}s  {total[i]:>10d}")

    n_confirmed = sum(confirmed)
    print(f"\n  Steps confirmed: {n_confirmed}/7")
    print(f"  Steps SKIPPED / not sustained: "
          f"{', '.join(tcs.STEP_NAMES[i] for i in range(tcs.NUM_CLASSES) if not confirmed[i]) or 'none'}")


if __name__ == "__main__":
    main()
