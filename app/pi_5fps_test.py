"""
pi_5fps_test.py — Does the existing model survive native Pi frame rate?
==========================================================================
Answers one question: if we sample the video at ~5 FPS (the Pi's real
MediaPipe throughput, measured by pi_benchmark.py) instead of resampling
to the trained 15 FPS, does the model still correctly identify all 7 WHO
steps?

This reuses test_continuous_sequence_local.py's extraction, feature
pipeline, model, and scoring UNCHANGED -- the only thing that changes is
TARGET_FPS, which controls how densely extract_base() samples frames from
the video. Everything downstream (features, SEQ_LEN=64 window, scoring
against the timeline) automatically adapts because those functions read
TARGET_FPS from the module at call time.

Run this on the Pi (it's a pure offline/no-camera test -- reads the video
file, doesn't need picamera2 at all -- so it also works fine on your PC
for a side-by-side comparison if you want one).

Copy alongside the existing bundle:
    pi_5fps_test.py
    test_continuous_sequence_local.py   (already there)
    model_tcn_attention.py              (already there)
    student_best.pt                     (already there)
    norm_stats.npz                      (already there)
    hand_landmarker.task                (already there)
    all complete.mp4                    (the video -- transfer this)

USAGE:
    python3 pi_5fps_test.py "all complete.mp4"
    python3 pi_5fps_test.py "all complete.mp4" --target-fps 5
    python3 pi_5fps_test.py "all complete.mp4" --target-fps 15   # sanity check:
        should reproduce your original desktop result almost exactly
"""

import argparse
import test_continuous_sequence_local as tcs


def main():
    ap = argparse.ArgumentParser(description="Test model at reduced native FPS")
    ap.add_argument("video", help="Path to the test video")
    ap.add_argument("--target-fps", type=float, default=5.0,
                     help="Simulated native sampling rate (Pi benchmark showed "
                          "~5.3 FPS for MediaPipe on this board)")
    ap.add_argument("--seq-len", type=int, default=None,
                     help="Override SEQ_LEN (window length in frames). Default "
                          "keeps the original 64. Use this to hold the window's "
                          "REAL-TIME duration constant across FPS changes -- "
                          "e.g. at 5 FPS, --seq-len 22 gives ~4.3s of context, "
                          "matching the original 64-frames-at-15FPS window.")
    ap.add_argument("--tolerance", type=float, default=3.0)
    args = ap.parse_args()

    print(f"\n>>> Patching TARGET_FPS: {tcs.TARGET_FPS} -> {args.target_fps}")
    if args.seq_len is not None:
        print(f">>> Patching SEQ_LEN: {tcs.SEQ_LEN} -> {args.seq_len}")
        tcs.SEQ_LEN = args.seq_len
    print(f">>> Window now spans {tcs.SEQ_LEN/args.target_fps:.1f}s of real time "
          f"(original: {64/15:.1f}s at 64 frames / 15 FPS)\n")

    tcs.TARGET_FPS = args.target_fps

    result = tcs.test_sequence(
        args.video,
        step_durations=[8, 8, 8, 8, 8, 8, 8],
        tolerance=args.tolerance,
    )

    print("\n" + "=" * 62)
    print(f"VERDICT @ {args.target_fps} FPS")
    print("=" * 62)
    if result is None:
        print("  Test did not complete -- see errors above.")
        return
    print(f"  Overall window accuracy : {result['overall_accuracy']:.1%}")
    print(f"  Steps detected          : {result['steps_detected']}/7")
    if result["steps_missed"]:
        print(f"  Steps missed            : {', '.join(result['steps_missed'])}")
    print()
    if result["steps_detected"] == 7 and result["overall_accuracy"] >= 0.7:
        print("  -> Looks tolerant of reduced FPS. Retiming (option 2) is "
              "viable with little or no retraining -- worth also comparing "
              "against the --target-fps 15 sanity-check run above to see "
              "how much accuracy actually dropped.")
    elif result["steps_detected"] >= 5:
        print("  -> Partially degraded. Likely needs the norm_stats.npz "
              "recompute + short fine-tune (Step 2 from earlier), not a "
              "full retrain.")
    else:
        print("  -> Substantially broken at this FPS. Points toward needing "
              "either a real fine-tune on native-rate data, or leaning "
              "more on the Coral accelerator option instead.")


if __name__ == "__main__":
    main()
