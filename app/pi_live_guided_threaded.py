"""
pi_live_guided_threaded.py — Smooth video + guided WHO protocol, live on the Pi
==============================================================================
Fixes the lag from pi_live_guided.py by splitting the single blocking loop
into two threads:

  DISPLAY THREAD (this process's main thread): captures camera frames and
  redraws the HUD continuously at full camera speed (~30 FPS on this
  board). It never waits on MediaPipe -- it just reads whatever the most
  recently computed state is and draws it on top of a FRESH camera frame
  every iteration. This is what makes the video look smooth.

  INFERENCE THREAD: runs MediaPipe + the model at whatever pace this CPU
  actually supports (~5 FPS, per pi_benchmark.py), and updates shared
  state (predicted step, confidence, soap/rinse timers, step records).
  It never touches the display.

Reuses live_guided.py's drawing functions (draw_run, draw_remediate,
draw_center, draw_phase_gate, draw_done) and scoring functions
(step_verdict, score_ordered, activity_state) completely unchanged --
only the orchestration loop is rewritten to split across two threads.

REQUIRES A DISPLAY (same as pi_live_guided.py) -- run with DISPLAY=:0 on
the Pi's own desktop session, not headless SSH.

USAGE:
    DISPLAY=:0 python3 pi_live_guided_threaded.py
    DISPLAY=:0 python3 pi_live_guided_threaded.py --flip

Controls: [r] restart   [q] quit
"""

import argparse, threading, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from picamera2 import Picamera2

import test_continuous_sequence_local as tcs

WINDOW_TITLE = "WHO hand hygiene — guided"


class Picamera2Capture:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

    def read(self):
        return True, self.picam2.capture_array()   # BGR-ordered despite "RGB888"

    def release(self):
        self.picam2.stop()


def inference_worker(s, lock, stop_event, args, model, mean, std, device, lm,
                      prog_start, live_guided):
    """Runs MediaPipe + model inference at its own pace. Never touches display."""
    sample_dt = 1.0 / args.target_fps
    last_sample = time.time()

    while not stop_event.is_set():
        now = time.time()
        if now - last_sample < sample_dt:
            time.sleep(0.005)
            continue

        with lock:
            phase_check = s["phase"]
        if phase_check == "DONE":
            time.sleep(0.05)
            continue

        frame = s["frame"]
        if frame is None:
            last_sample = now
            continue

        dt_sample = now - last_sample
        last_sample = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, int((now - prog_start) * 1000))
        raw = tcs.mediapipe_to_raw(res)

        do_infer = False
        with lock:
            s["raw_buf"].append(raw)
            raw_stack = np.stack(s["raw_buf"])
            state, motion = live_guided.activity_state(
                raw_stack, args.min_visibility, args.motion_thresh)
            s["state"], s["motion"] = state, motion
            washing = (state == "WASHING")
            buffering = len(s["raw_buf"]) < tcs.SEQ_LEN

            if dt_sample < 0.5 and washing:
                if s["phase"] == "SOAP":  s["soap_s"]  += dt_sample
                if s["phase"] == "RINSE": s["rinse_s"] += dt_sample

            do_infer = (s["phase"] in ("RUN", "REMEDIATE") and washing
                        and not buffering and (s["si"] % args.stride == 0))
            s["si"] += 1
            if do_infer:
                buf_snapshot = np.stack(s["raw_buf"])
                phase_now = s["phase"]

        if do_infer:
            window = tcs.build_features(buf_snapshot)[-tcs.SEQ_LEN:]
            x = torch.from_numpy(((window - mean) / std).astype(np.float32)) \
                     .unsqueeze(0).to(device)
            with torch.no_grad():
                p = F.softmax(model(x)[0], dim=-1).squeeze(0).cpu().numpy()
            with lock:
                s["prob_buf"].append(p)
                p_sm = np.mean(s["prob_buf"], axis=0)
                s["pred"], s["conf"] = int(np.argmax(p_sm)), float(np.max(p_sm))
                if phase_now == "RUN" and s["phase"] == "RUN":
                    s["records"].append((now - s["start"], s["pred"], s["conf"]))
                elif phase_now == "REMEDIATE" and s["phase"] == "REMEDIATE":
                    q = s["missed_queue"]
                    if s["retry_idx"] < len(q) and s["pred"] == q[s["retry_idx"]]:
                        s["retry_hit"] += 1
                        s["retry_conf_sum"] += s["conf"]


def display_loop(s, lock, stop_event, args, n, L, total_s, win_period, span,
                  live_guided):
    """Captures frames + draws the HUD continuously. Never touches MediaPipe."""
    cap = Picamera2Capture(args.width, args.height)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.width, args.height)
    cv2.moveWindow(WINDOW_TITLE, 0, 0)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if args.flip:
            frame = cv2.flip(frame, 1)
        s["frame"] = frame
        now = time.time()

        with lock:
            phase = s["phase"]

        if phase == "PRIME":
            with lock:
                buf_len = len(s["raw_buf"]); state = s["state"]
            if buf_len >= tcs.SEQ_LEN and state != "NO_HANDS":
                with lock:
                    if s["phase"] == "PRIME":
                        s["phase"] = "SOAP"
                        print("  --- SOAP: wet hands & apply soap ---")
                hud = frame
            else:
                msg = "buffering..." if buf_len < tcs.SEQ_LEN else "SHOW HANDS"
                hud = live_guided.draw_center(frame, msg, "place both hands in view")

        elif phase == "SOAP":
            with lock:
                soap_s = s["soap_s"]; state = s["state"]
            rem = max(0.0, args.soap_seconds - soap_s)
            if rem <= 0:
                with lock:
                    if s["phase"] == "SOAP":
                        s["soap_done"] = True; s["phase"] = "GETSET"; s["go_since"] = None
                        print("  --- soap OK ---")
                hud = frame
            else:
                hud = live_guided.draw_phase_gate(frame, "APPLY SOAP", rem, state)

        elif phase == "GETSET":
            with lock:
                if s["go_since"] is None:
                    s["go_since"] = now
                go_since = s["go_since"]
            left = live_guided.PRIME_COUNTDOWN - (now - go_since)
            if left <= 0:
                with lock:
                    if s["phase"] == "GETSET":
                        s["phase"] = "RUN"; s["start"] = now; s["prob_buf"].clear()
                        print("  --- START (Step 1) ---")
                hud = frame
            else:
                hud = live_guided.draw_center(
                    frame, f"GET READY  {int(left)+1}", "begin Step 1 (Rub Palms)")

        elif phase == "RUN":
            with lock:
                start = s["start"]; pred = s["pred"]; state = s["state"]; motion = s["motion"]
            elapsed = now - start

            with lock:
                for k in range(n):
                    if not s["revealed"][k] and elapsed >= (k + 1) * L + args.reveal_grace:
                        ok, cf, _, _ = live_guided.step_verdict(
                            s["records"], k, L, args.grace, args.min_detect, win_period)
                        s["revealed"][k] = True; s["live_pass"][k] = ok; s["live_conf"][k] = cf
                        print(f"   Step {k+1}: {'PERFORMED' if ok else 'missed'} (reveal)")

            if elapsed >= total_s:
                with lock:
                    if s["phase"] == "RUN":
                        s["results"] = live_guided.score_ordered(
                            s["records"], n, L, args.grace, args.min_detect, win_period)
                        for k in range(n):
                            s["revealed"][k] = True
                            s["live_pass"][k] = s["results"][k]["passed"]
                            s["live_conf"][k] = s["results"][k]["conf"]
                        missed = [k for k in range(n) if not s["results"][k]["passed"]]
                        if missed:
                            s["missed_queue"] = missed; s["retry_idx"] = 0
                            s["retry_start"] = now; s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                            s["pred"] = -1; s["prob_buf"].clear(); s["phase"] = "REMEDIATE"
                            print(f"  --- REDO missed steps: {[k+1 for k in missed]} ---")
                        else:
                            s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---")
                hud = frame
            else:
                slot = min(n - 1, int(elapsed // L))
                with lock:
                    s["cur_slot"] = slot
                    disp = [({"passed": s["live_pass"][k], "conf": s["live_conf"][k], "retry": False}
                             if s["revealed"][k] else None) for k in range(n)]
                    cur_det_s = sum(1 for (t, p, c) in s["records"]
                                    if p == slot and slot * L <= t <= elapsed) * win_period
                hud = live_guided.draw_run(frame, slot, pred, state, motion,
                                            L - (elapsed - slot * L), total_s - elapsed,
                                            disp, slot, n, args.target_fps, span,
                                            cur_det_s, args.min_detect)

        elif phase == "REMEDIATE":
            with lock:
                q = list(s["missed_queue"]); retry_idx = s["retry_idx"]
                retry_start = s["retry_start"]; retry_hit = s["retry_hit"]
                retry_conf_sum = s["retry_conf_sum"]; pred = s["pred"]; state = s["state"]
            if retry_idx >= len(q):
                with lock:
                    if s["phase"] == "REMEDIATE":
                        s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---")
                hud = frame
            else:
                k = q[retry_idx]
                det_s = retry_hit * win_period
                elapsed_r = now - retry_start
                advanced = False
                with lock:
                    if det_s >= args.min_detect:
                        s["results"][k].update(passed=True, retry=True,
                            conf=retry_conf_sum / max(1, retry_hit))
                        print(f"   Step {k+1} ({tcs.STEP_NAMES[k].split(': ')[-1]}): CORRECTED")
                        advanced = True
                    elif elapsed_r >= args.retry_seconds:
                        print(f"   Step {k+1} ({tcs.STEP_NAMES[k].split(': ')[-1]}): still missed")
                        advanced = True
                    if advanced:
                        s["retry_idx"] += 1; s["retry_start"] = now
                        s["retry_hit"] = 0; s["retry_conf_sum"] = 0.0
                        s["pred"] = -1; s["prob_buf"].clear()
                        if s["retry_idx"] >= len(q):
                            s["phase"] = "RINSE"; print("  --- RINSE: rinse hands ---")
                    results_snapshot = s["results"]
                hud = frame if advanced else live_guided.draw_remediate(
                    frame, k, max(0.0, args.retry_seconds - elapsed_r),
                    state, pred, q, retry_idx, results_snapshot)

        elif phase == "RINSE":
            with lock:
                rinse_s = s["rinse_s"]; state = s["state"]
            rem = max(0.0, args.rinse_seconds - rinse_s)
            if rem <= 0:
                with lock:
                    if s["phase"] == "RINSE":
                        s["rinse_done"] = True; s["phase"] = "DONE"
                        live_guided.print_summary(s, n, args.soap_seconds,
                                                   args.rinse_seconds, args.target_fps, span)
                hud = frame
            else:
                hud = live_guided.draw_phase_gate(frame, "RINSE HANDS", rem, state)

        else:  # DONE
            with lock:
                results = s["results"]; soap_done = s["soap_done"]; rinse_done = s["rinse_done"]
            hud = live_guided.draw_done(frame, results, n, soap_done, rinse_done)

        cv2.imshow(WINDOW_TITLE, hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break
        if key == ord("r"):
            with lock:
                fresh_s = live_guided.fresh(n)
                fresh_s["frame"] = s["frame"]
                s.clear(); s.update(fresh_s)
            print("\n  --- restart ---\n")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Threaded guided WHO protocol, live on the Pi")
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--seq-len", type=int, default=22)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--step-seconds", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=8.0)
    ap.add_argument("--min-detect", type=float, default=1.2)
    ap.add_argument("--min-visibility", type=float, default=0.2)
    ap.add_argument("--motion-thresh", type=float, default=0.006)
    ap.add_argument("--soap-seconds", type=float, default=5.0)
    ap.add_argument("--rinse-seconds", type=float, default=5.0)
    ap.add_argument("--retry-seconds", type=float, default=8.0)
    ap.add_argument("--reveal-grace", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    tcs.TARGET_FPS = args.target_fps
    tcs.SEQ_LEN = args.seq_len
    tcs.STRIDE = args.stride
    print(f">>> TARGET_FPS={tcs.TARGET_FPS}  SEQ_LEN={tcs.SEQ_LEN}  STRIDE={tcs.STRIDE}")

    import live_guided
    print(f">>> live_guided BUF_LEN={live_guided.BUF_LEN}  VIS_WINDOW={live_guided.VIS_WINDOW}\n")

    n = tcs.NUM_CLASSES
    L = args.step_seconds
    total_s = L * n
    win_period = args.stride / args.target_fps
    span = args.seq_len / args.target_fps
    device = "cpu"

    print("[1] Loading model...")
    model, mean, std, val_f1 = tcs.load_model(device)
    print(f"    val_f1 = {val_f1:.4f}")

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(tcs.MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=tcs.MAX_HANDS,
        min_hand_detection_confidence=tcs.MP_DET_CONF,
        min_hand_presence_confidence=tcs.MP_PRES_CONF,
        min_tracking_confidence=tcs.MP_TRK_CONF,
    )

    s = live_guided.fresh(n)
    s["frame"] = None
    lock = threading.Lock()
    stop_event = threading.Event()
    prog_start = time.time()

    print("[2] Starting inference thread + display window...")
    print("    Put your hands in front of the camera to begin.\n")

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        t = threading.Thread(
            target=inference_worker,
            args=(s, lock, stop_event, args, model, mean, std, device, lm,
                  prog_start, live_guided),
            daemon=True,
        )
        t.start()
        display_loop(s, lock, stop_event, args, n, L, total_s, win_period,
                     span, live_guided)
        stop_event.set()
        t.join(timeout=2.0)

    print("\nDone.")


if __name__ == "__main__":
    main()
