"""
train_student_kd_v6.py — Student KD training for v6 pipeline
=============================================================
Completely self-contained — no imports from train_teacher_v5.py.
Requires only config.py, model_tcn_attention.py, and train_teacher_v6.py
(for FocalLoss, AugmentedSeqDataset, set_seed which are now defined there).
"""

import sys
_WORKING_DIR = "/kaggle/working"
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    precision_score, recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    BATCH_SIZE, NUM_WORKERS, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, GRAD_CLIP,
    STUDENT_BRANCH_KERNELS, STUDENT_BRANCH_CHANNELS,
    STUDENT_DILATIONS, STUDENT_FUSION_DIM, STUDENT_GRU_HIDDEN,
    TEACHER_BRANCH_KERNELS, TEACHER_BRANCH_CHANNELS,
    TEACHER_DILATIONS, TEACHER_FUSION_DIM, TEACHER_GRU_HIDDEN,
    DROPOUT_TEACHER, FEATURE_DIM, PROTECT_START,
    MIXUP_ALPHA, AUG_ORIENT_PROTECT, NUM_CLASSES,
)
from model_tcn_attention import TCNAttentionHAR

# Import shared components from train_teacher_v6 (self-contained)
from train_teacher_v6 import (
    AugmentedSeqDataset,
    FocalLoss,
    set_seed,
    run_epoch_detailed,
)

# ─────────────────────────────────────────────────────────────────────────────
# V6 PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR_V6 = Path("/kaggle/working/handwash_tcn_dataset_v6")
META_DIR_V6 = BASE_DIR_V6 / "metadata"
SEQ_DIR_V6  = BASE_DIR_V6 / "sequences"
CKPT_DIR_V6 = BASE_DIR_V6 / "checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# V6 HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

KD_T            = 3.0
KD_ALPHA        = 0.4
KD_BETA         = 0.20
KD_GAMMA        = 0.35
AUX_LOSS_WEIGHT = 0.1
FOCAL_GAMMA     = 1.5
LABEL_SMOOTHING = 0.1

CLASS_WEIGHTS_V6 = [
    1.414, 1.000, 1.414, 1.000, 1.000,
    1.250,   # Step 6
    1.000,
]

EPOCHS_STUDENT     = 150
LR_STUDENT         = 8e-4
DROPOUT_STUDENT    = 0.20
STEP6_LABEL        = 5
STEP6_RECALL_FLOOR = 0.88
CKPT_S6_WEIGHT     = 0.4

STEP_NAMES = [
    "Rub Palms", "Rub Backs", "Interlace", "Back Fingers",
    "Thumbs", "Fingertips", "Wrists",
]

# ─────────────────────────────────────────────────────────────────────────────
# KD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class FeatProjection(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
    def forward(self, x): return self.proj(x)


def cosine_feat_loss(projected, target):
    B, T, D = projected.shape
    p = projected.reshape(B*T, D)
    t = target.reshape(B*T, D)
    return (1. - F.cosine_similarity(p, t, dim=1)).mean()


def kd_soft_loss(s_logits, t_logits, T):
    return F.kl_div(F.log_softmax(s_logits/T, dim=1),
                    F.softmax(t_logits/T, dim=1),
                    reduction="batchmean") * T**2


def build_hard_criterion(device):
    alpha = torch.tensor(CLASS_WEIGHTS_V6, dtype=torch.float32, device=device)
    return FocalLoss(alpha, gamma=FOCAL_GAMMA,
                     smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────

def run_train_epoch(student, teacher, projector, loader,
                    hard_criterion, optimizer, device):
    student.train(); projector.train(); teacher.eval()
    losses, ys, ps = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            tl, _ = teacher(x)
            tf     = teacher.get_intermediate_features(x)

        if np.random.random() < 0.5:
            lam  = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            idx  = torch.randperm(x.size(0), device=device)
            xm   = lam*x + (1-lam)*x[idx]
            ya, yb = y, y[idx]
            ts_mix = lam*F.softmax(tl/KD_T,1) + (1-lam)*F.softmax(tl[idx]/KD_T,1)
            with torch.no_grad():
                tfm = teacher.get_intermediate_features(xm)
            sl, sfl, _ = student(xm, return_frame_logits=True)
            sf = student.get_intermediate_features(xm)
            Ls = F.kl_div(F.log_softmax(sl/KD_T,1), ts_mix,
                           reduction="batchmean") * KD_T**2
            Lh = lam*hard_criterion(sl,ya) + (1-lam)*hard_criterion(sl,yb)
            La = lam*hard_criterion(sfl.mean(1),ya) + (1-lam)*hard_criterion(sfl.mean(1),yb)
            Lf = cosine_feat_loss(projector(sf), tfm)
        else:
            sl, sfl, _ = student(x, return_frame_logits=True)
            sf = student.get_intermediate_features(x)
            Ls = kd_soft_loss(sl, tl, KD_T)
            Lh = hard_criterion(sl, y)
            La = hard_criterion(sfl.mean(1), y)
            Lf = cosine_feat_loss(projector(sf), tf)

        loss = KD_ALPHA*Ls + KD_BETA*Lh + KD_GAMMA*Lf + AUX_LOSS_WEIGHT*La
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(projector.parameters()), GRAD_CLIP)
        optimizer.step()

        losses.append(loss.item())
        ps.extend(torch.argmax(sl, 1).cpu().numpy().tolist())
        ys.extend(y.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)),
        "acc":  float(accuracy_score(ys, ps)),
        "f1":   float(f1_score(ys, ps, average="macro", zero_division=0)),
    }


def run_eval_epoch_kd(student, teacher, projector, loader,
                      hard_criterion, device, detailed=False):
    student.eval(); projector.eval(); teacher.eval()
    losses, ys, ps = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            tl, _ = teacher(x)
            tf     = teacher.get_intermediate_features(x)
            sl, sfl, _ = student(x, return_frame_logits=True)
            sf = student.get_intermediate_features(x)
            Ls = kd_soft_loss(sl, tl, KD_T)
            Lh = hard_criterion(sl, y)
            La = hard_criterion(sfl.mean(1), y)
            Lf = cosine_feat_loss(projector(sf), tf)
            loss = KD_ALPHA*Ls + KD_BETA*Lh + KD_GAMMA*Lf + AUX_LOSS_WEIGHT*La
            losses.append(loss.item())
            ps.extend(torch.argmax(sl, 1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())

    result = {
        "loss": float(np.mean(losses)),
        "acc":  float(accuracy_score(ys, ps)),
        "f1":   float(f1_score(ys, ps, average="macro", zero_division=0)),
    }
    if detailed:
        result["macro_precision"]     = float(precision_score(ys, ps, average="macro", zero_division=0))
        result["macro_recall"]        = float(recall_score(ys, ps, average="macro", zero_division=0))
        result["per_class_f1"]        = f1_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist()
        result["per_class_precision"] = precision_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist()
        result["per_class_recall"]    = recall_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist()
        result["confusion_matrix"]    = confusion_matrix(ys, ps, labels=range(NUM_CLASSES)).tolist()
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    CKPT_DIR_V6.mkdir(parents=True, exist_ok=True)

    print("="*62)
    print("STUDENT KD TRAINING  v6  (self-contained)")
    print("="*62)
    print(f"  FEATURE_DIM     : {FEATURE_DIM}")
    print(f"  FOCAL_GAMMA     : {FOCAL_GAMMA}")
    print(f"  KD T/α/β/γ      : {KD_T}/{KD_ALPHA}/{KD_BETA}/{KD_GAMMA}")
    print(f"  DROPOUT_STUDENT : {DROPOUT_STUDENT}")
    print(f"  S6 weight       : {CLASS_WEIGHTS_V6[5]}")
    print(f"  CKPT score      : macro_f1 + {CKPT_S6_WEIGHT} × s6_recall")
    print(f"  Dataset         : {BASE_DIR_V6}")

    with open(META_DIR_V6 / "dataset_index.json") as f:
        index = json.load(f)

    train_ds = AugmentedSeqDataset(index, "train", SEQ_DIR_V6, augment=True)
    val_ds   = AugmentedSeqDataset(index, "val",   SEQ_DIR_V6, augment=False)
    test_ds  = AugmentedSeqDataset(index, "test",  SEQ_DIR_V6, augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher = TCNAttentionHAR(
        input_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
        branch_kernels=TEACHER_BRANCH_KERNELS,
        branch_channels=TEACHER_BRANCH_CHANNELS,
        dilations=TEACHER_DILATIONS,
        fusion_dim=TEACHER_FUSION_DIM,
        gru_hidden=TEACHER_GRU_HIDDEN,
        dropout=DROPOUT_TEACHER,
        use_feature_attn=True, use_layer_norm=False,
    ).to(device)

    t_ckpt_path = CKPT_DIR_V6 / "teacher_best.pt"
    if not t_ckpt_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {t_ckpt_path}\n"
            "Run train_teacher_v6.py first.")
    t_ckpt = torch.load(t_ckpt_path, map_location=device)
    teacher.load_state_dict(t_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    print(f"\n  Teacher: ep={t_ckpt['best_epoch']}  "
          f"val_f1={t_ckpt['best_val_f1']:.4f}  "
          f"s6r={t_ckpt['best_s6_recall']:.4f}")

    # ── Student ───────────────────────────────────────────────────────────
    student = TCNAttentionHAR(
        input_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
        branch_kernels=STUDENT_BRANCH_KERNELS,
        branch_channels=STUDENT_BRANCH_CHANNELS,
        dilations=STUDENT_DILATIONS,
        fusion_dim=STUDENT_FUSION_DIM,
        gru_hidden=STUDENT_GRU_HIDDEN,
        dropout=DROPOUT_STUDENT,
        use_feature_attn=True, use_layer_norm=False,
    ).to(device)

    projector  = FeatProjection(STUDENT_FUSION_DIM, TEACHER_FUSION_DIM).to(device)
    s_params   = student.count_parameters()
    t_params   = sum(p.numel() for p in teacher.parameters())
    p_params   = sum(p.numel() for p in projector.parameters())
    print(f"  Student  : {s_params:,} params  ({s_params/t_params*100:.1f}% of teacher)")
    print(f"  Projector: {p_params:,} params")

    hard_criterion = build_hard_criterion(device)
    all_params     = list(student.parameters()) + list(projector.parameters())
    optimizer      = AdamW(all_params, lr=LR_STUDENT, weight_decay=WEIGHT_DECAY)
    scheduler      = CosineAnnealingLR(optimizer, T_max=EPOCHS_STUDENT, eta_min=1e-6)

    print("\n"+"="*62+"\nTRAINING\n"+"="*62+"\n")

    best_ckpt_score  = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    s6_floor_active  = False

    for epoch in range(1, EPOCHS_STUDENT + 1):
        tr = run_train_epoch(student, teacher, projector,
                             train_dl, hard_criterion, optimizer, device)
        va = run_eval_epoch_kd(student, teacher, projector,
                               val_dl, hard_criterion, device, detailed=True)
        scheduler.step()

        s6_recall  = va["per_class_recall"][STEP6_LABEL]
        ckpt_score = va["f1"] + CKPT_S6_WEIGHT * s6_recall

        print(f"Ep {epoch:03d} | "
              f"Train L:{tr['loss']:.4f} F1:{tr['f1']:.4f} | "
              f"Val L:{va['loss']:.4f} F1:{va['f1']:.4f} | "
              f"Score:{ckpt_score:.4f}")
        print(f"         S6 F1:{va['per_class_f1'][5]:.4f}  "
              f"S6 P:{va['per_class_precision'][5]:.4f}  "
              f"S6 R:{s6_recall:.4f}")

        history.append({
            "epoch":                   epoch,
            "train_loss":              tr["loss"],
            "train_f1":                tr["f1"],
            "val_loss":                va["loss"],
            "val_f1":                  va["f1"],
            "val_ckpt_score":          ckpt_score,
            "val_s6_recall":           s6_recall,
            "val_macro_precision":     va["macro_precision"],
            "val_macro_recall":        va["macro_recall"],
            "val_per_class_f1":        va["per_class_f1"],
            "val_per_class_precision": va["per_class_precision"],
            "val_per_class_recall":    va["per_class_recall"],
            "lr":                      scheduler.get_last_lr()[0],
        })

        if ckpt_score > best_ckpt_score:
            best_ckpt_score = ckpt_score
            best_epoch      = epoch
            patience_counter = 0
            torch.save({
                "model":           student.state_dict(),
                "projector":       projector.state_dict(),
                "best_val_f1":     va["f1"],
                "best_ckpt_score": ckpt_score,
                "best_s6_recall":  s6_recall,
                "best_epoch":      epoch,
                "val_per_class_f1":         va["per_class_f1"],
                "val_per_class_precision":  va["per_class_precision"],
                "val_per_class_recall":     va["per_class_recall"],
                "val_confusion_matrix":     va["confusion_matrix"],
                "model_config": {
                    "input_dim":       FEATURE_DIM,
                    "num_classes":     NUM_CLASSES,
                    "branch_kernels":  STUDENT_BRANCH_KERNELS,
                    "branch_channels": STUDENT_BRANCH_CHANNELS,
                    "dilations":       STUDENT_DILATIONS,
                    "fusion_dim":      STUDENT_FUSION_DIM,
                    "gru_hidden":      STUDENT_GRU_HIDDEN,
                    "dropout":         DROPOUT_STUDENT,
                    "focal_gamma":     FOCAL_GAMMA,
                },
                "kd_config": {
                    "T": KD_T, "alpha": KD_ALPHA, "beta": KD_BETA,
                    "gamma": KD_GAMMA, "ckpt_s6_weight": CKPT_S6_WEIGHT,
                },
            }, CKPT_DIR_V6 / "student_best.pt")
            print(f"  ✓ Checkpoint  (score={ckpt_score:.4f}, "
                  f"f1={va['f1']:.4f}, s6r={s6_recall:.4f}, ep={epoch})")
        else:
            patience_counter += 1
            floor_active = STEP6_RECALL_FLOOR > 0.0 and s6_recall < STEP6_RECALL_FLOOR
            if patience_counter >= EARLY_STOP_PATIENCE:
                if floor_active:
                    if not s6_floor_active:
                        print(f"  ⚠  Patience exhausted, S6R={s6_recall:.4f} "
                              f"< floor {STEP6_RECALL_FLOOR} — continuing")
                        s6_floor_active = True
                else:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break
            if not floor_active:
                s6_floor_active = False

    # ── Final test ────────────────────────────────────────────────────────
    print("\n"+"="*62+"\nFINAL TEST EVALUATION\n"+"="*62)
    ckpt_s = torch.load(CKPT_DIR_V6 / "student_best.pt", map_location=device)
    student.load_state_dict(ckpt_s["model"])
    projector.load_state_dict(ckpt_s["projector"])
    print(f"  Loaded ep={ckpt_s['best_epoch']}  "
          f"score={ckpt_s['best_ckpt_score']:.4f}  "
          f"f1={ckpt_s['best_val_f1']:.4f}  "
          f"s6r={ckpt_s['best_s6_recall']:.4f}")

    test = run_eval_epoch_kd(student, teacher, projector,
                             test_dl, hard_criterion, device, detailed=True)

    print(f"\n  Test macro-F1  : {test['f1']:.4f}")
    print(f"  Test S6 recall : {test['per_class_recall'][5]:.4f}")
    print(f"  Test S6 F1     : {test['per_class_f1'][5]:.4f}")

    print(f"\n  {'Step':<22} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("  "+"─"*52)
    for i, name in enumerate(STEP_NAMES):
        flag = (" ← target" if i==5 else " ← minority" if i in (0,2) else "")
        print(f"  Step {i+1} ({name:14s})  "
              f"{test['per_class_precision'][i]:8.4f} "
              f"{test['per_class_recall'][i]:8.4f} "
              f"{test['per_class_f1'][i]:8.4f}{flag}")

    cm = test["confusion_matrix"]
    print(f"\n  Confusion matrix:")
    print("       "+"  ".join(f"P{i}" for i in range(1,8)))
    for i, row in enumerate(cm):
        print(f"    T{i+1}   "+"  ".join(f"{v:3d}" for v in row))

    history.append({
        "epoch": "final_test",
        "test_acc":  test["acc"],  "test_f1": test["f1"],
        "test_macro_precision":     test["macro_precision"],
        "test_macro_recall":        test["macro_recall"],
        "test_per_class_f1":        test["per_class_f1"],
        "test_per_class_precision": test["per_class_precision"],
        "test_per_class_recall":    test["per_class_recall"],
        "test_confusion_matrix":    test["confusion_matrix"],
    })

    with open(CKPT_DIR_V6 / "student_history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(CKPT_DIR_V6 / "student_confusion_matrix.npy",
            np.array(test["confusion_matrix"]))

    print(f"\n  Teacher : {t_params:,} params")
    print(f"  Student : {s_params:,} params  ({s_params/t_params*100:.1f}% of teacher)")
    print(f"\n  Best epoch      : {best_epoch}")
    print(f"  Best ckpt score : {best_ckpt_score:.4f}")
    print(f"  Test macro-F1   : {test['f1']:.4f}")
    print(f"  Test S6 recall  : {test['per_class_recall'][5]:.4f}")


if __name__ == "__main__":
    main()