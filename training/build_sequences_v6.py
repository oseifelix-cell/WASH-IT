"""
build_sequences_v6.py — Feature extraction pipeline v6
=======================================================
Changes from v5:

1. PALM-SCALE NORMALIZATION for distance-like Step 6 features
   Problem: raw MediaPipe coordinate distances (fingertip spread,
   fingertip-to-wrist, z-spread) vary with hand size, camera distance,
   and camera angle. Per-feature z-score normalization removes global
   mean/variance but cannot remove within-subject correlation between
   hand size and feature value. This is the most likely explanation
   for G05 LOSO collapse — G05's hand geometry sits outside the
   training distribution.

   Fix: before computing any distance-like feature, divide raw
   landmark coordinates by a palm scale factor:
       scale = ||landmarks[MIDDLE_MCP] - landmarks[WRIST]||
   This makes all distance features invariant to hand size and
   camera distance. Angular features (curl angle, palm normal dot
   product) and presence flags are left unscaled — they are already
   scale-invariant.

   Affected feature dims (all distance-based):
     [0:126]   wrist-relative coords  ← scaled before wrist-subtraction
     [126:128] presence flags         ← unscaled
     [384:393] orientation: palm normals (unit vectors, scale-invariant),
               dot product (scale-invariant), z-spread (scaled)
     [393:399] step6-v4: spread mean/std, fingertip-to-wrist (scaled)
     [399:401] step6-v5: curl angles (scale-invariant, angle-based)
     [401:403] step6-v5: spread velocity (scaled, derived from spread)

   Implementation: _palm_scale(landmarks, present) returns the
   wrist-to-middle-MCP distance, or 1.0 if hand absent (no-op).
   Applied in build_full_feature_sequence_v6() before all feature
   computation on raw coords.

2. Step 6 added to MIRROR_LABELS (was {0, 2}, now {0, 2, 5})
   Step 6 involves fingertips pressing into the opposite palm.
   Left-hand-dominant and right-hand-dominant versions are
   geometrically distinct. Mirroring adds free augmentation
   diversity for the weakest class.

3. OUT_DIR changed to handwash_tcn_dataset_v6 to keep v5 intact
   so you can compare LOSO results between versions.

All other pipeline logic (kinematic block, orientation, step6-v4/v5
features, normalization, sequence windowing) is identical to v5.
FEATURE_DIM remains 403.
"""

import sys
_WORKING_DIR = "/kaggle/working"
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

import json
import re
import random
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, SEQ_LEN, STRIDE, SEED, TARGET_FPS,
    MP_TASK_MODEL, MAX_NUM_HANDS, MP_DET_CONF, MP_TRK_CONF, MP_PRES_CONF,
    ALL_SUBJECTS, LOSO_TEST_SUBJECT, FIXED_VAL_SUBJECT, FIXED_TEST_SUBJECT,
)

# ─────────────────────────────────────────────────────────────────────────────
# V6 OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────

# Keep v5 output intact — write v6 to a separate directory
OUT_DIR_V6 = Path("/kaggle/working/handwash_tcn_dataset_v6")
SEQ_DIR_V6 = OUT_DIR_V6 / "sequences"
META_DIR_V6 = OUT_DIR_V6 / "metadata"
CKPT_DIR_V6 = OUT_DIR_V6 / "checkpoints"

# Step 6 now mirrored alongside minority classes
MIRROR_LABELS_V6 = {0, 2, 5}   # Steps 1, 3, 6

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
HAND_COORDS   = 63
BASE_DIM      = 128
_FLAG_OFFSETS = [126, 127]

# MediaPipe landmark indices
WRIST      = 0
MIDDLE_MCP = 9     # used for palm scale
INDEX_MCP  = 5
PINKY_MCP  = 17
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20
THUMB_TIP  = 4

KINEMATIC_DIM  = 384
ORIENT_DIM     = 9
STEP6_V4_DIM   = 6
STEP6_V5_DIM   = 4
FEATURE_DIM    = KINEMATIC_DIM + ORIENT_DIM + STEP6_V4_DIM + STEP6_V5_DIM  # 403

ORIENT_START   = 384
ORIENT_END     = 393
STEP6_V4_START = 393
STEP6_V4_END   = 399
STEP6_V5_START = 399
STEP6_V5_END   = 403
PROTECT_START  = ORIENT_START  # 384


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _flag_indices():
    return [b * 128 + f for b in range(3) for f in _FLAG_OFFSETS]


def extract_subject_id(path_or_name):
    m = re.search(r"G_?(\d+)", str(path_or_name), re.IGNORECASE)
    if m:
        return f"G{int(m.group(1)):02d}"
    raise ValueError(f"Could not extract subject ID from: {path_or_name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO LISTING  (same as v5)
# ─────────────────────────────────────────────────────────────────────────────

def list_videos():
    items = []
    for class_id in range(1, 8):
        class_dir = RAW_DATA_DIR / f"class_{class_id}"
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix in VIDEO_EXTS:
                items.append({
                    "video_path": str(p),
                    "video_name": p.stem,
                    "label":      class_id - 1,
                    "subject":    extract_subject_id(p),
                })
    return items


def split_videos(video_items, loso_test_subject,
                 fixed_val_subject, fixed_test_subject):
    if loso_test_subject is not None:
        test_subj = loso_test_subject
        val_subj  = next(s for s in ALL_SUBJECTS if s != test_subj)
        train_subj = [s for s in ALL_SUBJECTS
                      if s not in (test_subj, val_subj)]
        print(f"\nLOSO fold  test={test_subj}  val={val_subj}  train={train_subj}")
    else:
        test_subj  = fixed_test_subject
        val_subj   = fixed_val_subject
        train_subj = [s for s in ALL_SUBJECTS
                      if s not in (test_subj, val_subj)]
        print(f"\nFixed split  test={test_subj}  val={val_subj}  train={train_subj}")

    def _assign(s):
        if s == test_subj: return "test"
        if s == val_subj:  return "val"
        return "train"

    split      = {i: _assign(v["subject"]) for i, v in enumerate(video_items)}
    labels_arr = np.array([v["label"]   for v in video_items])
    groups_arr = np.array([v["subject"] for v in video_items])

    for sname in ("train", "val", "test"):
        idxs   = [i for i, s in split.items() if s == sname]
        subjs  = sorted(set(groups_arr[idxs]))
        counts = np.bincount(labels_arr[idxs], minlength=7)
        print(f"  {sname:5s}: {len(idxs):4d} videos | "
              f"subjects {subjs} | class counts {counts.tolist()}")
    return split


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE EXTRACTION  (same as v5)
# ─────────────────────────────────────────────────────────────────────────────

def result_to_feature(result):
    left  = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    left_present, right_present = 0.0, 0.0

    if result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            hand_label = ""
            if i < len(result.handedness) and len(result.handedness[i]) > 0:
                hand_label = result.handedness[i][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                               dtype=np.float32)
            if hand_label == "Left":
                left, left_present = coords, 1.0
            else:
                right, right_present = coords, 1.0

    return np.concatenate([
        left.reshape(-1), right.reshape(-1),
        np.array([left_present, right_present], dtype=np.float32),
    ])


def extract_base_features(video_path, target_fps=15):
    if not Path(MP_TASK_MODEL).exists():
        raise FileNotFoundError(f"HandLandmarker not found: {MP_TASK_MODEL}")

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_TASK_MODEL),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MP_DET_CONF,
        min_hand_presence_confidence=MP_PRES_CONF,
        min_tracking_confidence=MP_TRK_CONF,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    original_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(original_fps / target_fps)))
    feats, frame_idx, sampled_idx = [], 0, 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms    = int((sampled_idx / target_fps) * 1000.0)
                result   = landmarker.detect_for_video(mp_image, ts_ms)
                feats.append(result_to_feature(result))
                sampled_idx += 1
            frame_idx += 1

    cap.release()
    if len(feats) < SEQ_LEN:
        return None
    return np.stack(feats, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# HAND MIRRORING  (same as v5)
# ─────────────────────────────────────────────────────────────────────────────

def mirror_raw_frame(frame):
    m = frame.copy()
    for lm in range(21):
        m[lm * 3]               = -frame[lm * 3]
        m[HAND_COORDS + lm * 3] = -frame[HAND_COORDS + lm * 3]
    left_block, right_block = m[0:63].copy(), m[63:126].copy()
    m[0:63], m[63:126] = right_block, left_block
    m[126], m[127] = m[127], m[126]
    return m


def mirror_base_sequence(base_seq):
    return np.stack(
        [mirror_raw_frame(base_seq[t]) for t in range(base_seq.shape[0])],
        axis=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PALM SCALE  — NEW IN V6
# ─────────────────────────────────────────────────────────────────────────────

def _palm_scale(landmarks: np.ndarray, present: float) -> float:
    """
    Returns the Euclidean distance from WRIST to MIDDLE_MCP in raw
    MediaPipe coordinates. Used to normalize all distance-like features
    so they become invariant to hand size and camera distance.

    Returns 1.0 (no-op) when the hand is absent, preserving zero-padding.

    Why WRIST → MIDDLE_MCP:
      This segment is the most stable across all hand poses — it does not
      change length regardless of finger configuration, unlike fingertip
      distances which shorten when fingers curl. It captures hand size
      and camera distance reliably.

    Typical values in MediaPipe normalized coords:
      ~0.05–0.15 depending on camera distance and hand size.
    """
    if present < 0.5:
        return 1.0
    scale = float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST]))
    return scale if scale > 1e-6 else 1.0


def _scale_landmarks(landmarks: np.ndarray,
                     present: float) -> tuple:
    """
    Returns (scaled_landmarks, scale_factor).
    scaled_landmarks has the same shape as landmarks but all coordinates
    divided by the palm scale. When hand is absent, returns landmarks
    unchanged and scale=1.0.
    """
    s = _palm_scale(landmarks, present)
    return landmarks / s, s


# ─────────────────────────────────────────────────────────────────────────────
# HAND LANDMARK ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def _get_hand_landmarks(frame, hand):
    if hand == "left":
        return frame[0:63].reshape(21, 3), float(frame[126])
    return frame[63:126].reshape(21, 3), float(frame[127])


def _palm_normal(landmarks):
    v1     = landmarks[INDEX_MCP] - landmarks[WRIST]
    v2     = landmarks[PINKY_MCP] - landmarks[WRIST]
    normal = np.cross(v1, v2).astype(np.float32)
    norm   = np.linalg.norm(normal)
    return normal / norm if norm > 1e-6 else normal


# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION FEATURES  (9 dims — angles are scale-invariant, z-spread scaled)
# ─────────────────────────────────────────────────────────────────────────────

def _fingertip_z_spread(landmarks, present):
    if present < 0.5:
        return 0.0
    tips_z = landmarks[[INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP], 2]
    return float(tips_z.max() - tips_z.min())


def compute_orientation_features_v6(raw_frame):
    """
    Same as v5 but uses palm-scaled landmarks for z-spread.
    Palm normals and dot product are computed on unit vectors — inherently
    scale-invariant — so they do not need scaling.
    """
    left_lm,  lp = _get_hand_landmarks(raw_frame, "left")
    right_lm, rp = _get_hand_landmarks(raw_frame, "right")

    # Scale landmarks for distance-based features
    left_scaled,  _ = _scale_landmarks(left_lm,  lp)
    right_scaled, _ = _scale_landmarks(right_lm, rp)

    # Palm normals: computed on raw (unit vectors anyway)
    ln = _palm_normal(left_lm)  if lp > 0.5 else np.zeros(3, np.float32)
    rn = _palm_normal(right_lm) if rp > 0.5 else np.zeros(3, np.float32)

    return np.array([
        *ln, *rn,
        float(np.dot(ln, rn)),
        _fingertip_z_spread(left_scaled,  lp),   # scaled
        _fingertip_z_spread(right_scaled, rp),   # scaled
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 V4 FEATURES  (6 dims — all distance-based, now palm-scaled)
# ─────────────────────────────────────────────────────────────────────────────

def _inter_fingertip_spread(landmarks, present):
    if present < 0.5:
        return 0.0, 0.0
    tips = landmarks[[INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]
    dists = np.array([
        np.linalg.norm(tips[0] - tips[1]),
        np.linalg.norm(tips[1] - tips[2]),
        np.linalg.norm(tips[2] - tips[3]),
    ], dtype=np.float32)
    return float(dists.mean()), float(dists.std())


def _fingertip_to_opposite_wrist(own_lm, own_p, opp_lm, opp_p):
    if own_p < 0.5 or opp_p < 0.5:
        return 0.0
    tips  = own_lm[[INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]
    dists = np.linalg.norm(tips - opp_lm[WRIST], axis=1)
    return float(dists.mean())


def compute_step6_v4_features_v6(raw_frame):
    """
    Same as v5 but ALL distance features are computed on palm-scaled
    landmarks. This makes fingertip spread and fingertip-to-wrist
    distances invariant to hand size and camera distance.
    """
    left_lm,  lp = _get_hand_landmarks(raw_frame, "left")
    right_lm, rp = _get_hand_landmarks(raw_frame, "right")

    # Scale each hand independently by its own palm scale
    left_scaled,  _ = _scale_landmarks(left_lm,  lp)
    right_scaled, _ = _scale_landmarks(right_lm, rp)

    l_mean, l_std = _inter_fingertip_spread(left_scaled,  lp)
    r_mean, r_std = _inter_fingertip_spread(right_scaled, rp)

    # Cross-hand distance: use each hand scaled by its OWN palm
    # (opposite wrist is from the other hand's raw coords divided by
    # that hand's scale — both hands normalized independently)
    l_to_r = _fingertip_to_opposite_wrist(
        left_scaled, lp, right_scaled, rp)
    r_to_l = _fingertip_to_opposite_wrist(
        right_scaled, rp, left_scaled, lp)

    return np.array([l_mean, l_std, r_mean, r_std, l_to_r, r_to_l],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 V5 FEATURES  (4 dims — curl angles scale-invariant, spread vel scaled)
# ─────────────────────────────────────────────────────────────────────────────

def _fingertip_curl_angle(landmarks, present):
    """
    Angle between wrist-to-fingertip-centroid vector and palm normal.
    This is computed from unit vectors so it is scale-invariant.
    No change needed from v5.
    """
    if present < 0.5:
        return 0.0
    tip_indices  = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    tip_centroid = landmarks[tip_indices].mean(axis=0)
    w2t          = tip_centroid - landmarks[WRIST]
    norm_w2t     = np.linalg.norm(w2t)
    if norm_w2t < 1e-6:
        return 0.0
    w2t    = w2t / norm_w2t
    palm_n = _palm_normal(landmarks)
    norm_pn = np.linalg.norm(palm_n)
    if norm_pn < 1e-6:
        return 0.0
    cos_angle = float(np.clip(np.dot(w2t, palm_n), -1.0, 1.0))
    return float(np.arccos(cos_angle))


def compute_step6_v5_per_frame_v6(raw_frame):
    """Curl angles — scale-invariant, same as v5."""
    left_lm,  lp = _get_hand_landmarks(raw_frame, "left")
    right_lm, rp = _get_hand_landmarks(raw_frame, "right")
    return np.array([
        _fingertip_curl_angle(left_lm,  lp),
        _fingertip_curl_angle(right_lm, rp),
    ], dtype=np.float32)


def compute_spread_velocity(step6_v4_seq):
    """
    Temporal diff of spread mean — same as v5.
    Since spread mean is now palm-scaled, velocity is also scale-invariant.
    """
    left_spread_mean  = step6_v4_seq[:, 0]
    right_spread_mean = step6_v4_seq[:, 2]
    left_vel  = np.diff(left_spread_mean,  prepend=left_spread_mean[0])
    right_vel = np.diff(right_spread_mean, prepend=right_spread_mean[0])
    return np.stack([left_vel, right_vel], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KINEMATIC BLOCK  — v6: wrist-relative computed on palm-scaled coords
# ─────────────────────────────────────────────────────────────────────────────

def _apply_palm_scale_to_base_frame(frame: np.ndarray) -> np.ndarray:
    """
    Divide each hand's 21 landmarks by that hand's palm scale factor.
    Presence flags (dims 126, 127) are left unchanged.

    This operates on the raw 128-dim base frame before any kinematic
    feature engineering.
    """
    out = frame.copy()

    left_lm  = frame[0:63].reshape(21, 3)
    right_lm = frame[63:126].reshape(21, 3)
    lp       = float(frame[126])
    rp       = float(frame[127])

    sl = _palm_scale(left_lm,  lp)
    sr = _palm_scale(right_lm, rp)

    out[0:63]   = (left_lm  / sl).reshape(-1)
    out[63:126] = (right_lm / sr).reshape(-1)
    # flags 126, 127 unchanged

    return out


def _apply_palm_scale_to_base_sequence(base_seq: np.ndarray) -> np.ndarray:
    """Apply palm scale to every frame in a (T, 128) base sequence."""
    return np.stack(
        [_apply_palm_scale_to_base_frame(base_seq[t])
         for t in range(base_seq.shape[0])],
        axis=0,
    )


def _make_wrist_relative(base_seq):
    seq         = base_seq.copy()
    left_wrist  = seq[:, 0:3].copy()
    right_wrist = seq[:, HAND_COORDS: HAND_COORDS + 3].copy()
    for lm in range(21):
        seq[:, lm * 3: lm * 3 + 3]                              -= left_wrist
        seq[:, HAND_COORDS + lm * 3: HAND_COORDS + lm * 3 + 3] -= right_wrist
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# FULL FEATURE SEQUENCE  — v6 main change
# ─────────────────────────────────────────────────────────────────────────────

def build_full_feature_sequence_v6(base_seq: np.ndarray) -> np.ndarray:
    """
    Full feature engineering pipeline — v6.

    Changes from v5:
      1. Palm-scale normalization applied to raw base_seq BEFORE all
         feature computation. This makes all distance-based features
         (kinematic coords, spread, fingertip-to-wrist) scale-invariant.
      2. Step 6 v4/v5 features computed on palm-scaled landmarks.
      3. Orientation z-spread computed on palm-scaled landmarks.
      4. Curl angles and dot products are unchanged (inherently scale-inv).

    Pipeline:
      raw (T, 128)
        → palm-scale each frame (÷ wrist-to-middle-MCP distance per hand)
        → wrist-relative coords
        → velocity (Δ wrist-relative)
        → acceleration (Δ velocity)
        → orientation features (9 dims)     from palm-scaled raw coords
        → step6 v4 features    (6 dims)     from palm-scaled raw coords
        → step6 v5 curl angles (2 dims)     from raw coords (scale-inv)
        → step6 v5 spread vel  (2 dims)     from palm-scaled step6_v4 seq
        → concatenate → (T, 403)

    FEATURE_DIM = 403 (unchanged from v5).
    """
    T = base_seq.shape[0]

    # ── Apply palm scale to raw coordinates ──────────────────────────────
    scaled_seq = _apply_palm_scale_to_base_sequence(base_seq)

    # ── Kinematic block (384 dims) from palm-scaled coords ───────────────
    wr_seq    = _make_wrist_relative(scaled_seq)
    vel       = np.diff(wr_seq, axis=0, prepend=wr_seq[0:1])
    acc       = np.diff(vel,    axis=0, prepend=vel[0:1])
    kinematic = np.concatenate([wr_seq, vel, acc], axis=1)     # (T, 384)

    # ── Orientation block (9 dims) — v6 uses palm-scaled raw ─────────────
    orient = np.stack(
        [compute_orientation_features_v6(scaled_seq[t]) for t in range(T)],
        axis=0,
    )   # (T, 9)

    # ── Step6 v4 block (6 dims) — v6 uses palm-scaled raw ────────────────
    step6_v4 = np.stack(
        [compute_step6_v4_features_v6(scaled_seq[t]) for t in range(T)],
        axis=0,
    )   # (T, 6)

    # ── Step6 v5 curl angle block (2 dims) — scale-invariant ─────────────
    curl = np.stack(
        [compute_step6_v5_per_frame_v6(base_seq[t]) for t in range(T)],
        axis=0,
    )   # (T, 2)

    # ── Step6 v5 spread velocity (2 dims) — derived from scaled step6_v4 ─
    spread_vel = compute_spread_velocity(step6_v4)   # (T, 2)

    return np.concatenate(
        [kinematic, orient, step6_v4, curl, spread_vel], axis=1
    )   # (T, 403)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION  (same as v5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_train_stats(train_windows):
    if not train_windows:
        raise ValueError("No training windows provided.")
    feature_dim = train_windows[0].shape[1]
    flag_idx    = set(_flag_indices())
    coord_mask  = np.ones(feature_dim, dtype=bool)
    for fi in flag_idx:
        coord_mask[fi] = False

    all_frames = np.concatenate(
        [w.reshape(-1, feature_dim) for w in train_windows], axis=0
    ).astype(np.float32)

    mean = np.zeros(feature_dim, dtype=np.float32)
    std  = np.ones(feature_dim,  dtype=np.float32)
    mean[coord_mask] = all_frames[:, coord_mask].mean(axis=0)
    std[coord_mask]  = all_frames[:, coord_mask].std(axis=0) + 1e-6
    for fi in flag_idx:
        mean[fi] = 0.0
        std[fi]  = 1.0
    return {"mean": mean, "std": std}


def normalize_sequence(seq, mean, std):
    return ((seq - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    SEQ_DIR_V6.mkdir(parents=True, exist_ok=True)
    META_DIR_V6.mkdir(parents=True, exist_ok=True)
    CKPT_DIR_V6.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("BUILD SEQUENCES v6")
    print("=" * 62)
    print(f"  Changes from v5:")
    print(f"    1. Palm-scale normalization for all distance features")
    print(f"    2. Step 6 added to mirror labels (was {{0,2}}, now {{0,2,5}})")
    print(f"  FEATURE_DIM : {FEATURE_DIM}  (unchanged)")
    print(f"  MIRROR_LABELS: {MIRROR_LABELS_V6}")
    print(f"  Output dir  : {OUT_DIR_V6}")

    videos    = list_videos()
    split_map = split_videos(
        videos, LOSO_TEST_SUBJECT, FIXED_VAL_SUBJECT, FIXED_TEST_SUBJECT,
    )
    print(f"\n  Found {len(videos)} videos")

    # ── Pass 1: extract and engineer features ─────────────────────────────
    print(f"\nPass 1/2 — extracting features (FPS={TARGET_FPS})...")

    video_feats        = {}
    video_feats_mirror = {}
    train_windows      = []
    mirror_eligible    = 0

    for i, v in enumerate(tqdm(videos, desc="Extracting")):
        base = extract_base_features(Path(v["video_path"]),
                                     target_fps=TARGET_FPS)
        if base is None:
            video_feats[i] = video_feats_mirror[i] = None
            continue

        feat = build_full_feature_sequence_v6(base)   # (T, 403)
        video_feats[i] = feat

        is_train    = (split_map[i] == "train")
        is_minority = (v["label"] in MIRROR_LABELS_V6)   # now includes Step 6

        if is_train and is_minority:
            base_mirror           = mirror_base_sequence(base)
            feat_mirror           = build_full_feature_sequence_v6(base_mirror)
            video_feats_mirror[i] = feat_mirror
            mirror_eligible      += 1
        else:
            video_feats_mirror[i] = None

        if is_train:
            n = feat.shape[0]
            for start in range(0, n - SEQ_LEN + 1, STRIDE):
                train_windows.append(feat[start: start + SEQ_LEN])
            if video_feats_mirror[i] is not None:
                for start in range(0, n - SEQ_LEN + 1, STRIDE):
                    train_windows.append(
                        video_feats_mirror[i][start: start + SEQ_LEN]
                    )

    print(f"  Collected {len(train_windows)} train windows")
    print(f"  Videos with mirroring : {mirror_eligible}")
    print(f"  Feature dim           : {train_windows[0].shape[1]}")

    # ── Normalisation stats ───────────────────────────────────────────────
    print("  Computing per-feature normalisation statistics...")
    norm_stats = compute_train_stats(train_windows)
    np.savez(META_DIR_V6 / "norm_stats.npz",
             mean=norm_stats["mean"], std=norm_stats["std"])
    print(f"  Saved norm_stats.npz  (dim={norm_stats['mean'].shape[0]})")

    # ── Pass 2: normalize and save ────────────────────────────────────────
    print("\nPass 2/2 — normalising and saving sequences...")

    index     = []
    seq_id    = skipped = n_original = n_mirrored = 0

    for i, v in enumerate(tqdm(videos, desc="Saving")):
        feat = video_feats[i]
        if feat is None:
            skipped += 1
            continue

        n, split = feat.shape[0], split_map[i]

        for start in range(0, n - SEQ_LEN + 1, STRIDE):
            window   = normalize_sequence(
                feat[start: start + SEQ_LEN],
                norm_stats["mean"], norm_stats["std"],
            )
            filename = f"seq_{seq_id:07d}.npy"
            np.save(SEQ_DIR_V6 / filename, window)
            index.append({
                "seq_id":      seq_id,
                "file":        filename,
                "label":       int(v["label"]),
                "video_name":  v["video_name"],
                "subject":     v["subject"],
                "start_frame": int(start),
                "seq_len":     int(SEQ_LEN),
                "feature_dim": int(window.shape[1]),
                "split":       split,
                "mirrored":    False,
            })
            seq_id += 1; n_original += 1

        feat_m = video_feats_mirror[i]
        if feat_m is not None:
            for start in range(0, n - SEQ_LEN + 1, STRIDE):
                window   = normalize_sequence(
                    feat_m[start: start + SEQ_LEN],
                    norm_stats["mean"], norm_stats["std"],
                )
                filename = f"seq_{seq_id:07d}.npy"
                np.save(SEQ_DIR_V6 / filename, window)
                index.append({
                    "seq_id":      seq_id,
                    "file":        filename,
                    "label":       int(v["label"]),
                    "video_name":  v["video_name"] + "_mirror",
                    "subject":     v["subject"],
                    "start_frame": int(start),
                    "seq_len":     int(SEQ_LEN),
                    "feature_dim": int(window.shape[1]),
                    "split":       split,
                    "mirrored":    True,
                })
                seq_id += 1; n_mirrored += 1

    # ── Save metadata ─────────────────────────────────────────────────────
    with open(META_DIR_V6 / "dataset_index.json", "w") as f:
        json.dump(index, f, indent=2)

    split_counts  = {}
    class_counts  = {}
    mirror_counts = {}
    for item in index:
        split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
        if item["split"] == "train":
            lb = item["label"]
            if item["mirrored"]:
                mirror_counts[lb] = mirror_counts.get(lb, 0) + 1
            else:
                class_counts[lb]  = class_counts.get(lb, 0) + 1

    with open(META_DIR_V6 / "config.json", "w") as f:
        json.dump({
            "version":             "v6_palm_scale_mirror_s6",
            "num_sequences":       len(index),
            "num_original":        n_original,
            "num_mirrored":        n_mirrored,
            "num_videos_skipped":  skipped,
            "feature_dim":         FEATURE_DIM,
            "mirror_labels":       sorted(MIRROR_LABELS_V6),
            "palm_scale_applied":  True,
            "split_counts":        split_counts,
            "feature_dim_layout": {
                "wrist_relative_palm_scaled": "0:128",
                "velocity":                  "128:256",
                "acceleration":              "256:384",
                "orientation_v6":            "384:393",
                "step6_v4_palm_scaled":      "393:399",
                "step6_v5_curl":             "399:401",
                "step6_v5_spread_vel":       "401:403",
            },
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("DATASET READY  (v6)")
    print(f"{'='*62}")
    print(f"  Total sequences : {len(index)}")
    print(f"    Original      : {n_original}")
    print(f"    Mirrored      : {n_mirrored}")
    print(f"  Videos skipped  : {skipped}")
    print(f"  Feature dim     : {FEATURE_DIM}  (unchanged)")
    print(f"  Palm scale      : APPLIED to all distance features")
    print(f"  Split counts    : {split_counts}")

    step_names = ["Step 1","Step 2","Step 3","Step 4","Step 5","Step 6","Step 7"]
    print(f"\n  Train class breakdown:")
    for lb in range(7):
        orig   = class_counts.get(lb, 0)
        mirror = mirror_counts.get(lb, 0)
        tag    = "  ← mirrored" if mirror > 0 else ""
        print(f"    class {lb} {step_names[lb]:8s}: "
              f"{orig:4d} + {mirror:4d} mirror = {orig+mirror:4d} total{tag}")

    print(f"\n  Output → {OUT_DIR_V6}")
    print(f"\n  Next: update config.py paths to v6, then run")
    print(f"  train_teacher_v6.py and train_student_kd_v6.py")


if __name__ == "__main__":
    main()