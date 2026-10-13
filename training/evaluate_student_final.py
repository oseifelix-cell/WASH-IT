"""
evaluate_student_final.py — Full evaluation suite for the locked-in student model
======================================================================================
Generates every standard evaluation artifact for student_best.pt
(best_epoch=30, val_f1=0.9848, s6r=1.0, test_f1=0.9731, test_s6r=0.85):

  1. Training/validation loss curves (train_loss, val_loss vs epoch)
  2. Training/validation F1 curves (train_f1, val_f1 vs epoch)
  3. Checkpoint score curve with the selected best-epoch marked
  4. S6 recall/precision/F1 trajectory across epochs (the target metric)
  5. Confusion matrix heatmap (test set, raw counts + normalised %)
  6. Per-class precision/recall/F1 bar chart (test set)
  7. Test vs Validation comparison bar chart (sanity check for overfitting)
  8. Learning rate schedule plot
  9. Summary text panel with all key numbers in one place

All plots saved to /kaggle/working/handwash_tcn_dataset_v6/evaluation/
and displayed inline in the notebook.

Usage
-----
exec(open("/kaggle/working/evaluate_student_final.py").read())
run_full_evaluation()
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from IPython.display import display, Image as IPImage
    from io import BytesIO
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False

sys.path.insert(0, "/kaggle/working")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path("/kaggle/working/handwash_tcn_dataset_v6")
CKPT_PATH   = BASE_DIR / "checkpoints" / "student_best.pt"
HISTORY_PATH= BASE_DIR / "checkpoints" / "student_history.json"
OUT_DIR     = BASE_DIR / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_NAMES = [
    "Step 1\nRub Palms",    "Step 2\nRub Backs",
    "Step 3\nInterlace",    "Step 4\nBack Fingers",
    "Step 5\nThumbs",       "Step 6\nFingertips",
    "Step 7\nWrists",
]
STEP_SHORT  = ["S1","S2","S3","S4","S5","S6","S7"]
STEP_COLORS = ["#3B82F6","#10B981","#F59E0B","#EF4444",
               "#8B5CF6","#EC4899","#14B8A6"]
NUM_CLASSES = 7


def _show(fig, dpi=120):
    fig.tight_layout()
    if _IN_NOTEBOOK:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        display(IPImage(data=buf.read()))
    else:
        plt.close(fig)


def _save_and_show(fig, name, dpi=130):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  Saved → {path}")
    _show(fig)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint_and_history():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"History not found: {HISTORY_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    with open(HISTORY_PATH) as f:
        history = json.load(f)

    # Separate per-epoch records from the final "final_test" summary record
    epoch_records = [h for h in history if h.get("epoch") != "final_test"]
    test_record   = next((h for h in history if h.get("epoch") == "final_test"), None)

    return ckpt, epoch_records, test_record


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(epoch_records, best_epoch):
    epochs = [h["epoch"] for h in epoch_records]
    train_loss = [h["train_loss"] for h in epoch_records]
    val_loss   = [h["val_loss"]   for h in epoch_records]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#3B82F6", lw=1.8)
    ax.plot(epochs, val_loss,   label="Val Loss",   color="#EF4444", lw=1.8)
    ax.axvline(best_epoch, color="#10B981", ls="--", lw=2,
              label=f"Best checkpoint (epoch {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss Curves", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save_and_show(fig, "01_loss_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. F1 CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1_curves(epoch_records, best_epoch, best_val_f1):
    epochs = [h["epoch"] for h in epoch_records]
    train_f1 = [h["train_f1"] for h in epoch_records]
    val_f1   = [h["val_f1"]   for h in epoch_records]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, train_f1, label="Train F1", color="#3B82F6", lw=1.8)
    ax.plot(epochs, val_f1,   label="Val F1",   color="#EF4444", lw=1.8)
    ax.axvline(best_epoch, color="#10B981", ls="--", lw=2,
              label=f"Best checkpoint (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val_f1], color="#10B981", s=90, zorder=5)
    ax.annotate(f"{best_val_f1:.4f}", (best_epoch, best_val_f1),
               textcoords="offset points", xytext=(10,-12), fontsize=10,
               fontweight="bold", color="#10B981")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Training & Validation F1 Curves", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    _save_and_show(fig, "02_f1_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHECKPOINT SCORE CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_ckpt_score_curve(epoch_records, best_epoch, best_score):
    epochs = [h["epoch"] for h in epoch_records]
    scores = [h["val_ckpt_score"] for h in epoch_records]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, scores, color="#8B5CF6", lw=1.8)
    ax.axvline(best_epoch, color="#10B981", ls="--", lw=2)
    ax.scatter([best_epoch], [best_score], color="#10B981", s=110, zorder=5,
              label=f"Selected checkpoint: epoch {best_epoch}, score={best_score:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Checkpoint Score\n(val_f1 + 0.4 × s6_recall)")
    ax.set_title("Checkpoint Selection Score Over Training", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    _save_and_show(fig, "03_checkpoint_score_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. S6 RECALL/PRECISION/F1 TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_s6_trajectory(epoch_records, best_epoch):
    epochs = [h["epoch"] for h in epoch_records]
    s6_recall = [h["val_per_class_recall"][5] for h in epoch_records]
    s6_prec   = [h["val_per_class_precision"][5] for h in epoch_records]
    s6_f1     = [h["val_per_class_f1"][5] for h in epoch_records]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, s6_recall, label="S6 Recall",    color="#EC4899", lw=2)
    ax.plot(epochs, s6_prec,   label="S6 Precision", color="#F59E0B", lw=1.6, alpha=0.8)
    ax.plot(epochs, s6_f1,     label="S6 F1",        color="#8B5CF6", lw=1.6, alpha=0.8)
    ax.axvline(best_epoch, color="#10B981", ls="--", lw=2,
              label=f"Best checkpoint (epoch {best_epoch})")
    ax.axhline(0.88, color="grey", ls=":", lw=1.3, label="S6 recall floor (0.88)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Step 6 (Fingertips) Metrics Over Training — The Target Class",
                fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    _save_and_show(fig, "04_s6_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFUSION MATRIX  (raw + normalised)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(test_record):
    cm = np.array(test_record["test_confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                   color=color, fontsize=11, fontweight="bold")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(STEP_SHORT); ax.set_yticklabels(STEP_SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Raw Counts", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Normalised %
    ax = axes[1]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i,j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]*100:.0f}%", ha="center", va="center",
                   color=color, fontsize=10, fontweight="bold")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(STEP_SHORT); ax.set_yticklabels(STEP_SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Row-Normalised %\n(= Recall per class)",
                fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Test Set Confusion Matrix (n=328)", fontsize=14, fontweight="bold", y=1.02)
    _save_and_show(fig, "05_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PER-CLASS PRECISION/RECALL/F1 BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(test_record):
    prec = test_record["test_per_class_precision"]
    rec  = test_record["test_per_class_recall"]
    f1   = test_record["test_per_class_f1"]

    x = np.arange(NUM_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, prec, width, label="Precision", color="#3B82F6")
    b2 = ax.bar(x,         rec,  width, label="Recall",    color="#EF4444")
    b3 = ax.bar(x + width, f1,   width, label="F1",        color="#10B981")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.015, f"{h:.2f}",
                   ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(STEP_SHORT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Test Set — Per-Class Precision, Recall, F1", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Highlight S6 as the target class
    ax.axvspan(4.6, 5.4, color="#EC4899", alpha=0.08, zorder=0)
    ax.text(5, 1.10, "target class", ha="center", fontsize=8,
           color="#EC4899", fontweight="bold")

    _save_and_show(fig, "06_per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VAL vs TEST COMPARISON (overfitting sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_vs_test(ckpt, test_record):
    val_recall = ckpt["val_per_class_recall"]
    test_recall = test_record["test_per_class_recall"]

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width/2, val_recall,  width, label="Validation Recall", color="#8B5CF6")
    ax.bar(x + width/2, test_recall, width, label="Test Recall",       color="#F59E0B")
    ax.set_xticks(x); ax.set_xticklabels(STEP_SHORT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_title("Validation vs Test Recall — Generalisation Check",
                fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    gap = np.array(val_recall) - np.array(test_recall)
    max_gap_idx = int(np.argmax(np.abs(gap)))
    ax.annotate(f"Largest val→test gap:\n{STEP_SHORT[max_gap_idx]} "
               f"({gap[max_gap_idx]:+.2f})",
               xy=(max_gap_idx, max(val_recall[max_gap_idx], test_recall[max_gap_idx])),
               xytext=(max_gap_idx+0.8, 1.05),
               fontsize=9, ha="left",
               arrowprops=dict(arrowstyle="->", color="grey"))

    _save_and_show(fig, "07_val_vs_test.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def plot_lr_schedule(epoch_records):
    if "lr" not in epoch_records[0]:
        print("  (No LR data in history — skipping)")
        return
    epochs = [h["epoch"] for h in epoch_records]
    lrs    = [h["lr"] for h in epoch_records]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, lrs, color="#14B8A6", lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Annealing LR Schedule", fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3)
    _save_and_show(fig, "08_lr_schedule.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY PANEL
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_panel(ckpt, test_record):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")

    lines = [
        ("MODEL SUMMARY", "header"),
        (f"Checkpoint: student_best.pt  (epoch {ckpt['best_epoch']})", "normal"),
        (f"KD config: T={ckpt['kd_config']['T']}  α={ckpt['kd_config']['alpha']}  "
         f"β={ckpt['kd_config']['beta']}  γ={ckpt['kd_config']['gamma']}", "normal"),
        ("", "normal"),
        ("VALIDATION METRICS (best checkpoint)", "header"),
        (f"  Macro F1        : {ckpt['best_val_f1']:.4f}", "normal"),
        (f"  S6 Recall       : {ckpt['best_s6_recall']:.4f}", "normal"),
        (f"  Checkpoint score: {ckpt['best_ckpt_score']:.4f}", "normal"),
        ("", "normal"),
        ("TEST SET METRICS (held-out, n=328)", "header"),
        (f"  Macro F1        : {test_record['test_f1']:.4f}", "normal"),
        (f"  Accuracy        : {test_record['test_acc']:.4f}", "normal"),
        (f"  Macro Precision : {test_record['test_macro_precision']:.4f}", "normal"),
        (f"  Macro Recall    : {test_record['test_macro_recall']:.4f}", "normal"),
        (f"  S6 (target) Recall: {test_record['test_per_class_recall'][5]:.4f}", "highlight"),
        (f"  S6 (target) F1    : {test_record['test_per_class_f1'][5]:.4f}", "highlight"),
        ("", "normal"),
        ("PER-CLASS TEST RECALL", "header"),
    ]
    for i, name in enumerate(STEP_SHORT):
        r = test_record["test_per_class_recall"][i]
        flag = "  ← target" if i == 5 else ""
        lines.append((f"  {name}: {r:.4f}{flag}", "normal"))

    y = 0.97
    for text, style in lines:
        if style == "header":
            ax.text(0.02, y, text, fontsize=13, fontweight="bold",
                   color="#1e293b", transform=ax.transAxes)
            y -= 0.052
        elif style == "highlight":
            ax.text(0.02, y, text, fontsize=11, fontweight="bold",
                   color="#EC4899", transform=ax.transAxes)
            y -= 0.042
        else:
            ax.text(0.02, y, text, fontsize=11,
                   color="#334155", transform=ax.transAxes)
            y -= 0.042

    ax.set_title("Final Model Summary", fontweight="bold", fontsize=15, pad=15)
    _save_and_show(fig, "09_summary_panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation():
    print("="*62)
    print("FULL EVALUATION SUITE — student_best.pt")
    print("="*62)

    ckpt, epoch_records, test_record = load_checkpoint_and_history()

    print(f"\n  Checkpoint: epoch {ckpt['best_epoch']}, "
          f"val_f1={ckpt['best_val_f1']:.4f}, "
          f"s6_recall={ckpt['best_s6_recall']:.4f}")
    print(f"  Test:       f1={test_record['test_f1']:.4f}, "
          f"s6_recall={test_record['test_per_class_recall'][5]:.4f}")
    print(f"  Epoch records: {len(epoch_records)}")

    best_epoch = ckpt["best_epoch"]
    best_score = ckpt["best_ckpt_score"]
    best_val_f1= ckpt["best_val_f1"]

    print(f"\n[1/9] Loss curves...")
    plot_loss_curves(epoch_records, best_epoch)

    print(f"\n[2/9] F1 curves...")
    plot_f1_curves(epoch_records, best_epoch, best_val_f1)

    print(f"\n[3/9] Checkpoint score curve...")
    plot_ckpt_score_curve(epoch_records, best_epoch, best_score)

    print(f"\n[4/9] S6 trajectory...")
    plot_s6_trajectory(epoch_records, best_epoch)

    print(f"\n[5/9] Confusion matrix...")
    plot_confusion_matrix(test_record)

    print(f"\n[6/9] Per-class metrics...")
    plot_per_class_metrics(test_record)

    print(f"\n[7/9] Val vs Test comparison...")
    plot_val_vs_test(ckpt, test_record)

    print(f"\n[8/9] LR schedule...")
    plot_lr_schedule(epoch_records)

    print(f"\n[9/9] Summary panel...")
    plot_summary_panel(ckpt, test_record)

    print(f"\n{'='*62}")
    print(f"ALL PLOTS SAVED → {OUT_DIR}")
    print(f"{'='*62}")


if __name__ == "__main__":
    run_full_evaluation()