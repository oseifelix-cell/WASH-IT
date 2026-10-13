"""
package_for_local_testing.py — Run this ON KAGGLE to bundle everything
==========================================================================
Creates a single zip on Kaggle containing the 4 model files PLUS the
local testing scripts, ready for you to download as ONE file instead
of hunting through Kaggle's file browser for each piece separately.

Run this in a Kaggle code cell, then download the resulting zip from
Kaggle's Output panel (it will appear as
/kaggle/working/local_realtime_testing_bundle.zip).
"""

import os
import shutil
import zipfile
from pathlib import Path

# ── Source paths on Kaggle ───────────────────────────────────────────────────
MODEL_FILES = {
    "/kaggle/working/handwash_tcn_dataset_v6/checkpoints/student_best.pt": "student_best.pt",
    "/kaggle/working/handwash_tcn_dataset_v6/metadata/norm_stats.npz":     "norm_stats.npz",
    "/kaggle/working/models/hand_landmarker.task":                         "hand_landmarker.task",
    "/kaggle/working/model_tcn_attention.py":                              "model_tcn_attention.py",
}

OUTPUT_DIR  = Path("/kaggle/working/local_realtime_bundle")
OUTPUT_ZIP  = Path("/kaggle/working/local_realtime_testing_bundle.zip")
STATIC_DIR  = OUTPUT_DIR / "static"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

print("="*62)
print("PACKAGING MODEL FILES FOR LOCAL REAL-TIME TESTING")
print("="*62)

missing = []
for src, dst_name in MODEL_FILES.items():
    src_path = Path(src)
    if not src_path.exists():
        print(f"  ✗ MISSING: {src}")
        missing.append(src)
        continue
    dst_path = OUTPUT_DIR / dst_name
    shutil.copy2(src_path, dst_path)
    size_mb = dst_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Copied: {dst_name}  ({size_mb:.2f} MB)")

if missing:
    print(f"\n  ⚠ {len(missing)} file(s) missing — bundle will be incomplete")
else:
    print(f"\n  ✓ All 4 model files copied successfully")

# ── Write a verification README directly into the bundle ────────────────────
readme_content = """# Local Real-Time Testing Bundle

Contains the 4 model files needed to run real-time webcam inference
with the v6 student model (checkpoint: student_best.pt, epoch 30,
val_f1=0.9848, test_f1=0.9731, test_s6_recall=0.85).

Files in this bundle:
  - student_best.pt          (trained model checkpoint)
  - norm_stats.npz           (feature normalisation stats)
  - hand_landmarker.task     (MediaPipe hand detection model)
  - model_tcn_attention.py   (model architecture definition)

NEXT STEPS:
1. Extract this zip on your laptop
2. Download server.py, preflight_check.py, visibility_guard.py,
   webcam_inference_v6.py, and the static/ folder from the earlier
   deployment_bundle.zip (already provided separately)
3. Place ALL files together in one folder:

   pi_webapp/
   |-- server.py
   |-- preflight_check.py
   |-- visibility_guard.py
   |-- webcam_inference_v6.py
   |-- model_tcn_attention.py     <- from this bundle
   |-- student_best.pt             <- from this bundle
   |-- norm_stats.npz              <- from this bundle
   |-- hand_landmarker.task        <- from this bundle
   `-- static/
       |-- index.html
       `-- app.js

4. Run: python preflight_check.py
5. If all checks pass: python server.py
6. Open http://localhost:8000 in your browser
"""

with open(OUTPUT_DIR / "README.txt", "w") as f:
    f.write(readme_content)
print(f"\n  ✓ README.txt written")

# ── Zip it all up ─────────────────────────────────────────────────────────────
if OUTPUT_ZIP.exists():
    OUTPUT_ZIP.unlink()

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in OUTPUT_DIR.rglob("*"):
        if file.is_file():
            arcname = file.relative_to(OUTPUT_DIR)
            zf.write(file, arcname)

zip_size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
print(f"\n{'='*62}")
print(f"✓ BUNDLE CREATED: {OUTPUT_ZIP}")
print(f"  Size: {zip_size_mb:.2f} MB")
print(f"{'='*62}")
print(f"\nNow go to Kaggle's Output/Data panel (right sidebar) and")
print(f"download: local_realtime_testing_bundle.zip")