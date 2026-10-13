"""
config.py — centralised configuration for the HandWash TCN pipeline
=====================================================================
v5 changes:
  - FEATURE_DIM updated 399 → 403  (+4 Step 6 v5 disambiguation dims)
    New dims: curl angle × 2 (399:401) + spread velocity × 2 (401:403)
  - Added STEP6_V5_DIM, STEP6_V5_START, STEP6_V5_END constants
  - PROTECT_START unchanged at 384 — already covers 384:403 by range
  - LR_DUALSTREAM added for dual-stream fast training
  - DROPOUT_DUALSTREAM added
  - All other settings carried forward from v4
"""

import sys
import os

_WORKING_DIR = "/kaggle/working"
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

from pathlib import Path

# ============================================================================
# DATA PATHS
# ============================================================================

RAW_DATA_DIR = Path("/kaggle/working/handwash_7class_remapped")
OUT_DIR      = Path("/kaggle/working/handwash_tcn_dataset")
SEQ_DIR      = OUT_DIR / "sequences"
META_DIR     = OUT_DIR / "metadata"
CKPT_DIR     = OUT_DIR / "checkpoints"

# ============================================================================
# FEATURE DIMENSIONS  (v5)
# ============================================================================

KINEMATIC_DIM   = 384    # 3 × 128  (wrist-relative + velocity + acceleration)
ORIENT_DIM      = 9      # palm normals (6) + dot product (1) + z-spread (2)
STEP6_V4_DIM    = 6      # fingertip spread (4) + fingertip-to-wrist (2)
STEP6_V5_DIM    = 4      # curl angle (2) + spread velocity (2)  [NEW in v5]
FEATURE_DIM     = KINEMATIC_DIM + ORIENT_DIM + STEP6_V4_DIM + STEP6_V5_DIM  # 403

# Slice boundaries within the 403-dim vector
ORIENT_SLICE_START = 384
ORIENT_SLICE_END   = 393
STEP6_V4_START     = 393
STEP6_V4_END       = 399
STEP6_V5_START     = 399    # NEW
STEP6_V5_END       = 403    # NEW

# All engineered dims protected from feature dropout (covers 384:403)
PROTECT_START = 384

# ============================================================================
# SEQUENCE PARAMETERS
# ============================================================================

SEQ_LEN    = 64
STRIDE     = 16
SEED       = 42
TARGET_FPS = 15

# ============================================================================
# SUBJECT CONFIGURATION & CROSS-VALIDATION
# ============================================================================

ALL_SUBJECTS = ["G01", "G02", "G03", "G04", "G05"]

LOSO_TEST_SUBJECT = None   # set to "G01".."G05" for LOSO-CV

FIXED_VAL_SUBJECT  = "G03"
FIXED_TEST_SUBJECT = "G04"

NUM_CLASSES = 7

# ============================================================================
# HAND MIRRORING AUGMENTATION
# ============================================================================

MIRROR_LABELS = {0, 2}   # Steps 1 and 3 — minority classes

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================

MP_TASK_MODEL = "/kaggle/working/models/hand_landmarker.task"
MAX_NUM_HANDS = 2
MP_DET_CONF   = 0.5
MP_TRK_CONF   = 0.5
MP_PRES_CONF  = 0.5

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

BATCH_SIZE  = 64
NUM_WORKERS = 2

EPOCHS_TEACHER = 150
EPOCHS_STUDENT = 150

LR_TEACHER        = 1e-3
LR_STUDENT        = 8e-4
LR_DUALSTREAM     = 5e-4
WEIGHT_DECAY      = 1e-4

# ============================================================================
# KNOWLEDGE DISTILLATION PARAMETERS
# ============================================================================

KD_T     = 3.0
KD_ALPHA = 0.6
KD_BETA  = 0.35
KD_GAMMA = 0.35

# ============================================================================
# CLASS WEIGHTS  (sqrt-inverse-frequency — used as alpha in focal loss)
# ============================================================================

CLASS_WEIGHTS = [
    1.414,  # Step 1  — minority
    1.000,  # Step 2
    1.414,  # Step 3  — minority
    1.000,  # Step 4
    1.000,  # Step 5
    1.000,  # Step 6
    1.000,  # Step 7
]

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

USE_AUGMENTATION   = True
MIXUP_ALPHA        = 0.4
TEMPORAL_SHIFT_MAX = 8
LANDMARK_NOISE_STD = 0.01
HAND_DROP_PROB     = 0.1

# Protect all engineered dims (384:403) from feature dropout.
# Covers: orientation (384:393), step6-v4 (393:399), step6-v5 (399:403)
AUG_ORIENT_PROTECT = True

# ============================================================================
# REGULARIZATION
# ============================================================================

LABEL_SMOOTHING     = 0.1
DROPOUT_TEACHER     = 0.3
DROPOUT_STUDENT     = 0.25
DROPOUT_DUALSTREAM  = 0.3
EARLY_STOP_PATIENCE = 25
GRAD_CLIP           = 1.0

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Teacher model
TEACHER_BRANCH_KERNELS  = (3, 5, 7)
TEACHER_BRANCH_CHANNELS = 128
TEACHER_DILATIONS       = (1, 2, 4, 8)
TEACHER_FUSION_DIM      = 192
TEACHER_GRU_HIDDEN      = 128

# Student model (~63% of teacher capacity)
STUDENT_BRANCH_KERNELS  = (3, 5, 7)
STUDENT_BRANCH_CHANNELS = 96
STUDENT_DILATIONS       = (1, 2, 4, 8)
STUDENT_FUSION_DIM      = 144
STUDENT_GRU_HIDDEN      = 96