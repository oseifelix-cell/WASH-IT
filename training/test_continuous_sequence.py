"""
test_continuous_sequence.py — Test v6 model on a full WHO 7-step sequence
============================================================================
For testing a single continuous video that performs all 7 WHO handwash
steps IN ORDER (Step 1 -> Step 2 -> ... -> Step 7), matching the WHO
guideline that the full sequence takes ~40-60 seconds.

Ground truth assumption
------------------------
The 7 steps are assumed to occur strictly in order. You provide EITHER:
  (a) equal_duration=True  -> video duration split into 7 equal slices, or
  (b) custom per-step durations as a list of 7 numbers (seconds)

A tolerance window (default ±3s) is applied at each step boundary so
windows landing near a transition are not unfairly penalised — they are
counted as correct if they match EITHER the step before or after the
boundary within the tolerance.

Output
------
  - Per-step recall: was each step detected during its expected window?
  - Per-step "missed" flag: steps with near-zero coverage in their slot
  - Timeline plot: predicted (top) vs ground truth (bottom) side by side
  - Overall sequence-following score
"""

import sys, time, json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    from IPython.display import display, Image as IPImage
    from io import BytesIO
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False

sys.path.insert(0, "/kaggle/working")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = Path("/kaggle/working/handwash_tcn_dataset_v6")
CKPT_PATH = BASE_DIR / "checkpoints" / "student_best.pt"
NORM_PATH = BASE_DIR / "metadata"    / "norm_stats.npz"
MP_TASK   = Path("/kaggle/working/models/hand_landmarker.task")
OUT_DIR   = BASE_DIR / "sequence_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS  = 15
SEQ_LEN     = 64
STRIDE      = 4
FEATURE_DIM = 403
NUM_CLASSES = 7

MP_DET_CONF  = 0.5
MP_TRK_CONF  = 0.5
MP_PRES_CONF = 0.5
MAX_HANDS    = 2

DEFAULT_TOLERANCE_S = 3.0   # ±3 seconds at boundaries

STEP_NAMES = [
    "Step 1: Rub Palms",  "Step 2: Rub Backs",
    "Step 3: Interlace",  "Step 4: Back Fingers",
    "Step 5: Thumbs",     "Step 6: Fingertips",
    "Step 7: Wrists",
]
STEP_SHORT  = ["S1","S2","S3","S4","S5","S6","S7"]
STEP_COLORS = ["#3B82F6","#10B981","#F59E0B","#EF4444",
               "#8B5CF6","#EC4899","#14B8A6"]

WRIST=0; MIDDLE_MCP=9; INDEX_MCP=5; PINKY_MCP=17
INDEX_TIP=8; MIDDLE_TIP=12; RING_TIP=16; PINKY_TIP=20
HAND_COORDS=63


# ─────────────────────────────────────────────────────────────────────────────
# 1. GROUND TRUTH TIMELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(duration_s, step_durations=None):
    """
    Build a strict sequential ground-truth timeline for 7 WHO steps.

    Parameters
    ----------
    duration_s     : total video duration in seconds
    step_durations : list of 7 numbers (seconds) for each step's duration.
                      If None, splits duration_s into 7 equal slices.
                      If provided but doesn't sum to duration_s, it is
                      rescaled proportionally to fit the actual video length.

    Returns
    -------
    list of (start_s, end_s, step_idx) tuples, step_idx is 0-based
    """
    if step_durations is None:
        slice_len = duration_s / 7
        durations = [slice_len] * 7
    else:
        assert len(step_durations) == 7, "Must provide exactly 7 durations"
        total = sum(step_durations)
        # Rescale to match actual video duration
        durations = [d * duration_s / total for d in step_durations]

    timeline = []
    t = 0.0
    for i, d in enumerate(durations):
        timeline.append((t, t + d, i))
        t += d
    return timeline


def expected_step_at(t, timeline, tolerance=DEFAULT_TOLERANCE_S):
    """
    Return the set of step indices considered "correct" at time t,
    given the tolerance window. Near a boundary, both adjacent steps
    are accepted.
    """
    accepted = set()
    for start, end, step_idx in timeline:
        if (start - tolerance) <= t <= (end + tolerance):
            accepted.add(step_idx)
    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE PIPELINE  (v6 — palm-scaled, 403-dim)
# ─────────────────────────────────────────────────────────────────────────────

def _get_lm(frame, hand):
    if hand=="left": return frame[0:63].reshape(21,3), float(frame[126])
    return frame[63:126].reshape(21,3), float(frame[127])

def _palm_scale(lm,p):
    if p<0.5: return 1.0
    s=float(np.linalg.norm(lm[MIDDLE_MCP]-lm[WRIST]))
    return s if s>1e-6 else 1.0

def _palm_normal(lm):
    n=np.cross(lm[INDEX_MCP]-lm[WRIST],lm[PINKY_MCP]-lm[WRIST]).astype(np.float32)
    nm=np.linalg.norm(n); return n/nm if nm>1e-6 else n

def _scale_frame(frame):
    out=frame.copy()
    ll=frame[0:63].reshape(21,3); rl=frame[63:126].reshape(21,3)
    lp=float(frame[126]); rp=float(frame[127])
    out[0:63]=(ll/_palm_scale(ll,lp)).reshape(-1)
    out[63:126]=(rl/_palm_scale(rl,rp)).reshape(-1)
    return out

def _orient(sf):
    ll,lp=_get_lm(sf,"left"); rl,rp=_get_lm(sf,"right")
    ln=_palm_normal(ll) if lp>0.5 else np.zeros(3,np.float32)
    rn=_palm_normal(rl) if rp>0.5 else np.zeros(3,np.float32)
    def zsp(lm,p):
        z=lm[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP],2]
        return float(z.max()-z.min()) if p>0.5 else 0.
    return np.array([*ln,*rn,float(np.dot(ln,rn)),zsp(ll,lp),zsp(rl,rp)],np.float32)

def _spread(lm,p):
    if p<0.5: return 0.,0.
    t=lm[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP]]
    d=np.array([np.linalg.norm(t[i]-t[i+1]) for i in range(3)],np.float32)
    return float(d.mean()),float(d.std())

def _cross(own,op,pp,opp):
    if pp<0.5 or opp<0.5: return 0.
    t=own[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP]]
    return float(np.linalg.norm(t-op[WRIST],axis=1).mean())

def _s6v4(sf):
    ll,lp=_get_lm(sf,"left"); rl,rp=_get_lm(sf,"right")
    lm,ls=_spread(ll,lp); rm,rs=_spread(rl,rp)
    return np.array([lm,ls,rm,rs,_cross(ll,rl,lp,rp),_cross(rl,ll,rp,lp)],np.float32)

def _curl(lm,p):
    if p<0.5: return 0.
    tc=lm[[INDEX_TIP,MIDDLE_TIP,RING_TIP,PINKY_TIP]].mean(0)
    w2t=tc-lm[WRIST]; n=np.linalg.norm(w2t)
    if n<1e-6: return 0.
    pn=_palm_normal(lm); pnn=np.linalg.norm(pn)
    if pnn<1e-6: return 0.
    return float(np.arccos(np.clip(np.dot(w2t/n,pn),-1.,1.)))

def _s6v5(rf):
    ll,lp=_get_lm(rf,"left"); rl,rp=_get_lm(rf,"right")
    return np.array([_curl(ll,lp),_curl(rl,rp)],np.float32)

def mediapipe_to_raw(result):
    left=np.zeros((21,3),np.float32); right=np.zeros((21,3),np.float32)
    lp=rp=0.0
    if result.hand_landmarks:
        for i,lms in enumerate(result.hand_landmarks):
            label=""
            if i<len(result.handedness) and result.handedness[i]:
                label=result.handedness[i][0].category_name
            c=np.array([[lm.x,lm.y,lm.z] for lm in lms],np.float32)
            if label=="Left": left,lp=c,1.0
            else:             right,rp=c,1.0
    return np.concatenate([left.reshape(-1),right.reshape(-1),
                           np.array([lp,rp],np.float32)])

def build_features(base_seq):
    T=base_seq.shape[0]
    sc=np.stack([_scale_frame(base_seq[t]) for t in range(T)])
    wr=sc.copy()
    lw=wr[:,0:3].copy(); rw=wr[:,HAND_COORDS:HAND_COORDS+3].copy()
    for lm in range(21):
        wr[:,lm*3:lm*3+3]-=lw
        wr[:,HAND_COORDS+lm*3:HAND_COORDS+lm*3+3]-=rw
    vel=np.diff(wr,axis=0,prepend=wr[0:1])
    acc=np.diff(vel,axis=0,prepend=vel[0:1])
    kin=np.concatenate([wr,vel,acc],axis=1)
    orient=np.stack([_orient(sc[t]) for t in range(T)])
    s6v4=np.stack([_s6v4(sc[t]) for t in range(T)])
    curl=np.stack([_s6v5(base_seq[t]) for t in range(T)])
    lsv=np.diff(s6v4[:,0],prepend=s6v4[0,0])
    rsv=np.diff(s6v4[:,2],prepend=s6v4[0,2])
    spv=np.stack([lsv,rsv],axis=1)
    return np.concatenate([kin,orient,s6v4,curl,spv],axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_base(video_path):
    import mediapipe as mp
    opts=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )
    cap=cv2.VideoCapture(str(video_path))
    orig_fps=cap.get(cv2.CAP_PROP_FPS) or 30.
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur=total/orig_fps
    interval=max(1,int(round(orig_fps/TARGET_FPS)))
    print(f"  Duration   : {dur:.1f}s  ({total} frames @ {orig_fps:.0f}FPS)")
    print(f"  WHO target : 40-60s  →  "
          f"{'✓ within range' if 40<=dur<=60 else '⚠ outside WHO 40-60s range'}")

    feats=[]; fi=0; si=0
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ret,frame=cap.read()
            if not ret: break
            if fi%interval==0:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                res=lm.detect_for_video(img,int(si/TARGET_FPS*1000))
                feats.append(mediapipe_to_raw(res)); si+=1
            fi+=1
    cap.release()

    if not feats: return None, dur

    # ── Detailed hand-detection diagnostics ──────────────────────────────────
    left_present  = np.array([f[126] for f in feats])
    right_present = np.array([f[127] for f in feats])
    both_present  = (left_present > 0) & (right_present > 0)
    one_present   = (left_present > 0) ^ (right_present > 0)
    none_present  = (left_present == 0) & (right_present == 0)

    n_hands = int((left_present + right_present > 0).sum())
    print(f"  Extracted  : {len(feats)} frames  |  "
          f"Hands detected: {n_hands} ({100*n_hands/len(feats):.0f}%)")
    print(f"    Both hands  : {both_present.sum():4d} frames ({100*both_present.mean():.0f}%)")
    print(f"    One hand    : {one_present.sum():4d} frames ({100*one_present.mean():.0f}%)")
    print(f"    No hands    : {none_present.sum():4d} frames ({100*none_present.mean():.0f}%)")

    # Per-10-second breakdown — shows WHERE detection drops, not just overall %
    print(f"\n  Hand detection by time segment (10s buckets):")
    n = len(feats)
    seg_len = int(TARGET_FPS * 10)  # frames per 10s bucket
    for seg_start in range(0, n, seg_len):
        seg_end = min(seg_start + seg_len, n)
        t0, t1 = seg_start/TARGET_FPS, seg_end/TARGET_FPS
        seg_both = both_present[seg_start:seg_end].mean() * 100
        seg_one  = one_present[seg_start:seg_end].mean()  * 100
        seg_none = none_present[seg_start:seg_end].mean() * 100
        print(f"    {t0:5.1f}s-{t1:5.1f}s  |  "
              f"both:{seg_both:5.1f}%  one:{seg_one:5.1f}%  none:{seg_none:5.1f}%")

    return np.stack(feats), dur


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    from model_tcn_attention import TCNAttentionHAR
    ckpt=torch.load(CKPT_PATH,map_location=device,weights_only=False)
    cfg=ckpt["model_config"]
    model=TCNAttentionHAR(
        input_dim=cfg["input_dim"],num_classes=cfg["num_classes"],
        branch_kernels=tuple(cfg["branch_kernels"]),
        branch_channels=cfg["branch_channels"],
        dilations=tuple(cfg["dilations"]),fusion_dim=cfg["fusion_dim"],
        gru_hidden=cfg["gru_hidden"],dropout=cfg["dropout"],
        use_feature_attn=True,use_layer_norm=False,
    )
    try:    model.load_state_dict(ckpt["model"],strict=True)
    except: model.load_state_dict(ckpt["model"],strict=False)
    model.to(device).eval()
    stats=np.load(NORM_PATH)
    return (model, stats["mean"].astype(np.float32),
            stats["std"].astype(np.float32), ckpt.get("best_val_f1",0))


@torch.no_grad()
def infer(model, feat_seq, mean, std, device):
    T=feat_seq.shape[0]
    norm=((feat_seq-mean)/std).astype(np.float32)
    preds,probs,mids=[],[],[]
    if T < SEQ_LEN:
        pad=np.tile(norm[-1:], (SEQ_LEN-T,1))
        norm=np.concatenate([norm,pad],axis=0); T=SEQ_LEN
    for start in range(0,T-SEQ_LEN+1,STRIDE):
        w=norm[start:start+SEQ_LEN]
        x=torch.from_numpy(w).unsqueeze(0).to(device)
        out=model(x); logits=out[0]
        p=F.softmax(logits,dim=-1).squeeze(0).cpu().numpy()
        preds.append(int(np.argmax(p))); probs.append(p)
        mids.append(start+SEQ_LEN//2)
    return np.array(preds),np.array(probs),np.array(mids)

def smooth(probs,w=3):
    if len(probs)<=1: return probs
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(probs,size=min(w,len(probs)),axis=0,mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCORING — per-step recall against the sequential timeline
# ─────────────────────────────────────────────────────────────────────────────

def score_against_timeline(preds, mids, timeline, tolerance):
    """
    For each prediction window, check if it matches the expected step(s)
    at that timestamp (within tolerance). Returns per-window correctness
    and per-step recall statistics.

    Also returns a "what did the model predict instead" breakdown for
    each GT slot — essential for diagnosing whether a missed step is a
    timeline mismatch or a genuine confusion with a specific other step.
    """
    times = mids / TARGET_FPS
    window_correct  = []
    window_expected = []

    for t, pred in zip(times, preds):
        accepted = expected_step_at(t, timeline, tolerance)
        window_expected.append(sorted(accepted))
        window_correct.append(pred in accepted)

    # Per-step recall: for each GT step's exclusive (non-overlap) time range,
    # did the model predict that step in AT LEAST one window?
    step_detected = [False] * NUM_CLASSES
    step_window_count = [0] * NUM_CLASSES
    step_total_windows = [0] * NUM_CLASSES
    step_confusion = [None] * NUM_CLASSES   # what was predicted instead

    for start, end, step_idx in timeline:
        # Windows whose time falls within [start, end] (core region, no tolerance)
        mask = (times >= start) & (times <= end)
        n_windows_in_slot = int(mask.sum())
        step_total_windows[step_idx] = n_windows_in_slot
        if n_windows_in_slot > 0:
            slot_preds = preds[mask]
            n_correct  = int((slot_preds == step_idx).sum())
            step_window_count[step_idx] = n_correct
            step_detected[step_idx] = n_correct > 0
            # Breakdown of what WAS predicted in this slot
            counts = np.bincount(slot_preds, minlength=NUM_CLASSES)
            step_confusion[step_idx] = counts.tolist()

    return {
        "window_correct":     np.array(window_correct),
        "window_expected":    window_expected,
        "step_detected":      step_detected,
        "step_window_count":  step_window_count,
        "step_total_windows": step_total_windows,
        "step_confusion":     step_confusion,
        "overall_accuracy":   float(np.mean(window_correct)) if len(window_correct)>0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_sequence_result(preds, probs, mids, timeline, tolerance,
                         video_name, save_path):
    t = mids / TARGET_FPS
    W = len(preds)
    duration = timeline[-1][1]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                              gridspec_kw={"height_ratios":[1,1,2]})
    fig.suptitle(
        f"Sequential WHO Step Test: {video_name}\n"
        f"Duration={duration:.1f}s  ·  Tolerance=±{tolerance:.0f}s  ·  "
        f"WHO target: 40-60s",
        fontsize=12, fontweight="bold",
    )

    # ── Row 0: GROUND TRUTH timeline (expected steps in order) ────────────────
    ax0 = axes[0]
    for start, end, step_idx in timeline:
        ax0.fill_betweenx([0,1], start, end,
                          color=STEP_COLORS[step_idx], alpha=0.9)
        ax0.text((start+end)/2, 0.5, STEP_SHORT[step_idx],
                 ha="center", va="center", fontsize=10,
                 color="white", fontweight="bold")
        # Tolerance shading at boundaries
        ax0.axvspan(max(0,start-tolerance), start+tolerance,
                    color="grey", alpha=0.15, zorder=2)
    ax0.set_xlim(0, duration)
    ax0.set_ylim(0,1); ax0.set_yticks([])
    ax0.set_title("Expected (Ground Truth) — WHO Steps in Strict Order",
                  fontweight="bold")

    # ── Row 1: PREDICTED timeline ──────────────────────────────────────────
    ax1 = axes[1]
    for i in range(W-1):
        ax1.fill_betweenx([0,1], t[i], t[i+1],
                          color=STEP_COLORS[preds[i]], alpha=0.9)
    if W>0:
        ax1.fill_betweenx([0,1], t[-1], duration,
                          color=STEP_COLORS[preds[-1]], alpha=0.9)
    ax1.set_xlim(0, duration)
    ax1.set_ylim(0,1); ax1.set_yticks([])
    ax1.set_title("Predicted Step Timeline", fontweight="bold")
    patches=[mpatches.Patch(color=STEP_COLORS[i],label=STEP_SHORT[i])
             for i in range(NUM_CLASSES)]
    ax1.legend(handles=patches, loc="upper right", ncol=7, fontsize=8)

    # ── Row 2: confidence heatmap with GT boundaries overlaid ────────────────
    ax2 = axes[2]
    im = ax2.imshow(probs.T, aspect="auto", origin="lower", cmap="RdYlGn",
                    vmin=0, vmax=1,
                    extent=[0, duration, -0.5, NUM_CLASSES-0.5],
                    interpolation="nearest")
    for start, end, step_idx in timeline:
        ax2.axvline(start, color="white", lw=1, ls="--", alpha=0.6)
        ax2.axhline(step_idx, color="black", lw=0.3, alpha=0.2)
        ax2.plot([start, end], [step_idx, step_idx],
                 color="black", lw=3, alpha=0.5, solid_capstyle="butt")
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(STEP_SHORT, fontsize=10)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Step", fontsize=10)
    ax2.set_title("Confidence Heatmap (black bar = expected GT step at that time)",
                  fontweight="bold")
    plt.colorbar(im, ax=ax2, label="Probability", pad=0.01)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    print(f"\n  Plot saved → {save_path}")
    if _IN_NOTEBOOK:
        buf=BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",dpi=115)
        buf.seek(0); plt.close(fig); display(IPImage(data=buf.read()))
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def test_sequence(video_path, step_durations=None,
                  tolerance=DEFAULT_TOLERANCE_S):
    """
    Test the v6 model on a full continuous WHO 7-step handwash video.

    Ground truth assumption: steps occur strictly in order, Step1 -> Step7.

    Parameters
    ----------
    video_path     : path to the video (must show all 7 steps in order)
    step_durations : list of 7 numbers (seconds), or None for equal slices.
                      E.g. [5,5,7,6,6,9,4] for 42s total — will be rescaled
                      to match the actual video duration.
    tolerance       : seconds of buffer at each step boundary (default 3.0)
    """
    VIDEO_EXTS = {".mp4",".avi",".mov",".MP4",".AVI",".MOV",".mkv",".MKV"}
    video_path = Path(video_path)

    # Auto-resolve folders to the video file(s) inside them
    if video_path.is_dir():
        videos = sorted(v for v in video_path.rglob("*")
                        if v.suffix in VIDEO_EXTS and v.is_file())
        if not videos:
            print(f"  ✗ No video files found in folder: {video_path}")
            return None
        if len(videos) > 1:
            print(f"  Found {len(videos)} videos in {video_path}:")
            for i, v in enumerate(videos):
                print(f"    [{i}] {v.name}")
            print(f"  Using [0]: {videos[0].name}")
        video_path = videos[0]
    elif not video_path.exists():
        print(f"  ✗ Video not found: {video_path}")
        return None

    print("="*62)
    print("SEQUENTIAL WHO STEP TEST  (v6 model)")
    print("="*62)
    print(f"  Video     : {video_path.name}")
    print(f"  Tolerance : ±{tolerance}s at boundaries")

    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] Loading model ({device})...")
    model,mean,std,val_f1=load_model(device)
    print(f"  val_f1 = {val_f1:.4f}")

    print(f"\n[2] Extracting MediaPipe landmarks...")
    t0=time.time()
    base,duration=extract_base(video_path)
    print(f"  Done in {time.time()-t0:.1f}s")

    if base is None:
        print("  ✗ Could not extract frames")
        return None

    # Build ground truth timeline
    timeline = build_timeline(duration, step_durations)
    print(f"\n  Ground truth timeline (strict order, Step1→Step7):")
    for start,end,idx in timeline:
        print(f"    {STEP_NAMES[idx]:<22}  {start:6.1f}s — {end:6.1f}s  "
              f"({end-start:.1f}s)")

    print(f"\n[3] Computing v6 features...")
    feat=build_features(base)

    print(f"\n[4] Running inference (stride={STRIDE})...")
    preds,probs,mids=infer(model,feat,mean,std,device)
    probs_s=smooth(probs,w=3)
    preds_s=np.argmax(probs_s,axis=1)

    print(f"\n[5] Scoring against sequential timeline...")
    score = score_against_timeline(preds_s, mids, timeline, tolerance)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"RESULTS")
    print(f"{'='*62}")
    print(f"\n  Overall window accuracy (within tolerance) : "
          f"{score['overall_accuracy']:.1%}")

    print(f"\n  {'Step':<22}  {'Expected slot':>14}  {'Detected?':>10}  "
          f"{'Hit/Total windows':>18}")
    print("  "+"─"*72)
    missed_steps = []
    for start,end,idx in timeline:
        detected = score["step_detected"][idx]
        hit  = score["step_window_count"][idx]
        tot  = score["step_total_windows"][idx]
        flag = "✓ YES" if detected else "✗ MISSED"
        if not detected: missed_steps.append(idx)
        print(f"  {STEP_NAMES[idx]:<22}  {start:5.1f}-{end:5.1f}s  "
              f"{flag:>10}  {hit:>6d} / {tot:<6d}")

    # ── Confusion breakdown for MISSED steps — what did it predict instead? ──
    if missed_steps:
        print(f"\n  ── What the model predicted instead (missed steps) ──────")
        for idx in missed_steps:
            confusion = score["step_confusion"][idx]
            if confusion is None:
                continue
            print(f"\n  During {STEP_NAMES[idx]} slot, model predicted:")
            ranked = sorted(range(NUM_CLASSES), key=lambda i: -confusion[i])
            tot = sum(confusion)
            for rank_i in ranked:
                if confusion[rank_i] == 0:
                    continue
                pct = 100*confusion[rank_i]/tot if tot>0 else 0
                marker = " ← actually predicted this" if rank_i!=idx else ""
                print(f"      {STEP_NAMES[rank_i]:<22}: "
                      f"{confusion[rank_i]:>3d} windows ({pct:>5.1f}%){marker}")


    print(f"\n  ── Summary ──────────────────────────────────────")
    print(f"     Steps detected : {7-len(missed_steps)}/7")
    if missed_steps:
        print(f"     Steps MISSED   : "
              f"{', '.join(STEP_NAMES[i] for i in missed_steps)}")
        print(f"\n  ⚠  Recommendation: review video for the missed steps —")
        print(f"     either the person did not perform them clearly,")
        print(f"     or the model failed to recognise them.")
    else:
        print(f"     ✓ All 7 WHO steps were detected in their expected time slot")

    # Duration compliance
    print(f"\n  WHO duration compliance:")
    print(f"     Video duration : {duration:.1f}s")
    print(f"     WHO guideline  : 40-60s")
    print(f"     Compliant      : "
          f"{'✓ YES' if 40<=duration<=60 else '✗ NO'}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    stem=video_path.stem[:40]
    save_path=OUT_DIR/f"sequence_{stem}.png"
    print(f"\n[6] Generating plot...")
    plot_sequence_result(preds_s,probs_s,mids,timeline,tolerance,
                         video_path.name,save_path)

    return {
        "video":            str(video_path),
        "duration_s":       duration,
        "who_compliant":    bool(40<=duration<=60),
        "overall_accuracy": score["overall_accuracy"],
        "steps_detected":   7-len(missed_steps),
        "steps_missed":     [STEP_NAMES[i] for i in missed_steps],
        "timeline":         [(s,e,STEP_NAMES[i]) for s,e,i in timeline],
    }


if __name__ == "__main__":
    # Example usage — edit VIDEO_PATH
    result = test_sequence(
        "/kaggle/input/your-dataset/full_handwash.mp4",
        step_durations=None,   # None = equal 1/7 slices
        tolerance=3.0,
    )