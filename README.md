# WASH-IT

Offline handwashing monitor that recognizes the 7 WHO handwashing steps in real time from hand landmarks.
TCN + BiGRU + attention teacher, knowledge-distilled student, Raspberry Pi 4B kiosk.

## Results (held-out subject, 328 windows)
| | Teacher | Student |
|---|---|---|
| Parameters | 3,126,721 | 1,969,761 |
| Test accuracy | 95.43% (313/328) | 97.26% (319/328) |
| Test macro F1 | 0.9557 | 0.9731 |
| Step 6 recall | 0.750 | 0.850 |

## Repository layout
- `training/` Kaggle scripts: class remapping, sequence building, teacher and student training, evaluation, tests
- `notebooks/` the Kaggle notebook with outputs removed
- `app/` the kiosk application, kept flat because the scripts import each other: Raspberry Pi backend (`pi_server.py`), kiosk launch and systemd setup scripts, browser interface (`washit_v7_live.html` plus `assets/`), local test scripts, PIR and benchmark tests
- `docs/` result figures
- `hardware/` CAD files, wiring, Bill of Materials (add)

## Reproducing the training (Kaggle)
1. Attach the Sample Hand Wash Dataset (Kaggle: realtimear/hand-wash-dataset).
2. Run the notebook cells in order. Paths assume `/kaggle/input` and `/kaggle/working`.
3. Scripts write checkpoints to `/kaggle/working/handwash_tcn_dataset_v6/`.

## Not included
Dataset videos, test recordings of people, and trained weights (see Releases).

## Licence
Code: MIT. The dataset and any weights have their own terms.

## Running on the Raspberry Pi
1. Copy `app/` to `~/washit/` on the Pi and add `student_best.pt` and `hand_landmarker.task` next to the scripts (see Releases and the MediaPipe model page).
2. Run `pi_autostart_setup.sh` to install the backend and kiosk services.
3. The backend starts the PIR trigger only when given a pin: `python3 pi_server.py --flip --pir-pin <BCM pin>`. Edit the `ExecStart` line in `washit-server.service` to include your pin.
4. Window settings for the Pi are 5 frames/s and 22 frames (stride 4); training used 15 frames/s and 64 frames.
