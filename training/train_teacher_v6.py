"""
train_teacher_v6.py — Teacher training for v6 pipeline
=======================================================
Completely self-contained — no imports from train_teacher_v5.py.
All required classes and functions are defined here directly so this
file works in a fresh notebook with only config.py and
model_tcn_attention.py present.

Changes from v5:
  1. Paths → handwash_tcn_dataset_v6
  2. Step 6-aware checkpointing: ckpt_score = macro_f1 + 0.4 × s6_recall
  3. CLASS_WEIGHTS[5] = 1.25  (was 1.000)
  4. FOCAL_GAMMA = 1.5        (was 2.0)
  5. STEP6_BLUR_SIGMA = 1.2   (was 0.8)
  6. train_loss + val_loss saved to history
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
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    BATCH_SIZE, NUM_WORKERS, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, GRAD_CLIP,
    TEACHER_BRANCH_KERNELS, TEACHER_BRANCH_CHANNELS,
    TEACHER_DILATIONS, TEACHER_FUSION_DIM, TEACHER_GRU_HIDDEN,
    DROPOUT_TEACHER, FEATURE_DIM, PROTECT_START,
    MIXUP_ALPHA, AUG_ORIENT_PROTECT,
)
from model_tcn_attention import TCNAttentionHAR

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

EPOCHS_TEACHER     = 150
LR_TEACHER         = 1e-3
LABEL_SMOOTHING    = 0.1
NUM_CLASSES        = 7
FOCAL_GAMMA        = 1.5

CLASS_WEIGHTS_V6 = [
    1.414, 1.000, 1.414, 1.000, 1.000,
    1.250,   # Step 6 — was 1.000
    1.000,
]

STEP6_BLUR_SIGMA   = 1.2
STEP6_BLUR_PROB    = 0.7
STEP6_LABEL        = 5
STEP6_RECALL_FLOOR = 0.88
CKPT_S6_WEIGHT     = 0.4

STEP_NAMES = [
    "Rub Palms", "Rub Backs", "Interlace", "Back Fingers",
    "Thumbs", "Fingertips", "Wrists",
]

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, smoothing=0.1, num_classes=7):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        B, C     = logits.shape
        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()
        with torch.no_grad():
            sv = self.smoothing / C
            hv = 1.0 - self.smoothing + sv
            ys = torch.full_like(prob, sv)
            ys.scatter_(1, targets.unsqueeze(1), hv)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        fw  = (1.0 - p_t).pow(self.gamma)
        at  = self.alpha[targets]
        ce  = -(ys * log_prob).sum(dim=1)
        return (at * fw * ce).mean()


def build_criterion(device):
    alpha = torch.tensor(CLASS_WEIGHTS_V6, dtype=torch.float32, device=device)
    return FocalLoss(alpha, gamma=FOCAL_GAMMA,
                     smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedSeqDataset(Dataset):
    FLAG_INDICES     = [126, 127, 254, 255, 382, 383]
    STEP6_FEAT_START = 393
    STEP6_FEAT_END   = 403

    def __init__(self, index, split, seq_dir, augment=True):
        self.items   = [x for x in index if x["split"] == split]
        self.seq_dir = Path(seq_dir)
        self.augment = augment and (split == "train")
        if not self.items:
            raise ValueError(f"No samples for split='{split}'")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x  = np.load(self.seq_dir / it["file"]).astype(np.float32)
        y  = int(it["label"])
        if self.augment:
            x = self._augment(x, y)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def _augment(self, x, label):
        T, F = x.shape
        if np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            x = np.roll(x, shift, axis=0)
            if shift > 0:   x[:shift]  = x[shift:shift+1]
            elif shift < 0: x[shift:]  = x[shift-1:shift]
        if np.random.random() < 0.5:
            n = np.random.normal(0, 0.01, x.shape).astype(np.float32)
            n[:, self.FLAG_INDICES] = 0.0
            x = x + n
        if np.random.random() < 0.2:
            m = np.random.binomial(1, 0.95, F).astype(np.float32)
            for fi in self.FLAG_INDICES: m[fi] = 1.0
            if AUG_ORIENT_PROTECT: m[PROTECT_START:] = 1.0
            x = x * m
        if np.random.random() < 0.2:
            ml = np.random.randint(5, max(6, T // 4))
            ms = np.random.randint(0, T - ml)
            x[ms:ms+ml, :] = 0.0
        if np.random.random() < 0.2:
            sc = np.random.uniform(0.9, 1.1)
            x[:, 0:126]   *= sc
            x[:, 128:254] *= sc
            x[:, 256:382] *= sc
        if label == STEP6_LABEL and np.random.random() < STEP6_BLUR_PROB:
            x[:, self.STEP6_FEAT_START:self.STEP6_FEAT_END] = (
                gaussian_filter1d(
                    x[:, self.STEP6_FEAT_START:self.STEP6_FEAT_END],
                    sigma=STEP6_BLUR_SIGMA, axis=0, mode="nearest",
                ).astype(np.float32)
            )
        return x

# ─────────────────────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, alpha=0.4):
    lam   = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[index], y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device,
              optimizer=None, use_mixup=False, aux_weight=0.1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, ys, ps = [], [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train and use_mixup and np.random.random() < 0.5:
                x_mix, ya, yb, lam = mixup_batch(x, y, MIXUP_ALPHA)
                logits, fl, _ = model(x_mix, return_frame_logits=True)
                loss = (mixup_criterion(criterion, logits, ya, yb, lam)
                        + aux_weight * mixup_criterion(criterion, fl.mean(1), ya, yb, lam))
            else:
                logits, fl, _ = model(x, return_frame_logits=True)
                loss = criterion(logits, y) + aux_weight * criterion(fl.mean(1), y)
            if is_train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            losses.append(loss.item())
            ps.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)),
        "acc":  float(accuracy_score(ys, ps)),
        "f1":   float(f1_score(ys, ps, average="macro", zero_division=0)),
    }


def run_epoch_detailed(model, loader, criterion, device):
    model.eval()
    losses, ys, ps = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, fl, _ = model(x, return_frame_logits=True)
            loss = criterion(logits, y) + 0.1 * criterion(fl.mean(1), y)
            losses.append(loss.item())
            ps.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    return {
        "loss":                float(np.mean(losses)),
        "acc":                 float(accuracy_score(ys, ps)),
        "f1":                  float(f1_score(ys, ps, average="macro", zero_division=0)),
        "macro_precision":     float(precision_score(ys, ps, average="macro", zero_division=0)),
        "macro_recall":        float(recall_score(ys, ps, average="macro", zero_division=0)),
        "per_class_f1":        f1_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist(),
        "per_class_precision": precision_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist(),
        "per_class_recall":    recall_score(ys, ps, average=None, labels=range(NUM_CLASSES), zero_division=0).tolist(),
        "confusion_matrix":    confusion_matrix(ys, ps, labels=range(NUM_CLASSES)).tolist(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    CKPT_DIR_V6.mkdir(parents=True, exist_ok=True)

    print("="*62)
    print("TEACHER TRAINING  v6  (self-contained)")
    print("="*62)
    print(f"  FEATURE_DIM    : {FEATURE_DIM}")
    print(f"  FOCAL_GAMMA    : {FOCAL_GAMMA}")
    print(f"  S6 weight      : {CLASS_WEIGHTS_V6[5]}")
    print(f"  S6 blur sigma  : {STEP6_BLUR_SIGMA}")
    print(f"  CKPT score     : macro_f1 + {CKPT_S6_WEIGHT} × s6_recall")
    print(f"  Dataset        : {BASE_DIR_V6}")

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

    model = TCNAttentionHAR(
        input_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
        branch_kernels=TEACHER_BRANCH_KERNELS,
        branch_channels=TEACHER_BRANCH_CHANNELS,
        dilations=TEACHER_DILATIONS,
        fusion_dim=TEACHER_FUSION_DIM,
        gru_hidden=TEACHER_GRU_HIDDEN,
        dropout=DROPOUT_TEACHER,
        use_feature_attn=True, use_layer_norm=False,
    ).to(device)
    print(f"\n  Parameters : {model.count_parameters():,}")

    criterion = build_criterion(device)
    optimizer = AdamW(model.parameters(), lr=LR_TEACHER, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_TEACHER, eta_min=1e-6)

    print("\n"+"="*62+"\nTRAINING\n"+"="*62+"\n")

    best_ckpt_score  = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    s6_floor_active  = False

    for epoch in range(1, EPOCHS_TEACHER + 1):
        tr = run_epoch(model, train_dl, criterion, device,
                       optimizer=optimizer, use_mixup=True, aux_weight=0.1)
        va = run_epoch_detailed(model, val_dl, criterion, device)
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
                "model":           model.state_dict(),
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
                    "branch_kernels":  TEACHER_BRANCH_KERNELS,
                    "branch_channels": TEACHER_BRANCH_CHANNELS,
                    "dilations":       TEACHER_DILATIONS,
                    "fusion_dim":      TEACHER_FUSION_DIM,
                    "gru_hidden":      TEACHER_GRU_HIDDEN,
                    "dropout":         DROPOUT_TEACHER,
                    "focal_gamma":     FOCAL_GAMMA,
                },
            }, CKPT_DIR_V6 / "teacher_best.pt")
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
    ckpt = torch.load(CKPT_DIR_V6 / "teacher_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded epoch {ckpt['best_epoch']},  "
          f"score={ckpt['best_ckpt_score']:.4f},  f1={ckpt['best_val_f1']:.4f}")

    test = run_epoch_detailed(model, test_dl, criterion, device)
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

    with open(CKPT_DIR_V6 / "teacher_history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(CKPT_DIR_V6 / "teacher_confusion_matrix.npy",
            np.array(test["confusion_matrix"]))

    print(f"\n  Best epoch      : {best_epoch}")
    print(f"  Best ckpt score : {best_ckpt_score:.4f}")
    print(f"  Test macro-F1   : {test['f1']:.4f}")
    print(f"  Test S6 recall  : {test['per_class_recall'][5]:.4f}")


if __name__ == "__main__":
    main()