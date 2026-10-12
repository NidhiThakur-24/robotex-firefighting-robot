# Robotex Firefighting Robot — Computer Vision

A YOLO-based computer-vision project developed for the Robotex firefighting robotics competition. The work explored flame detection from camera images and how detection results could support a firefighting robot's search-and-approach workflow.

## Project overview

- **Task:** Detect target objects in camera images for a firefighting robotics scenario.
- **Model family:** Ultralytics YOLO (the notebook includes training and experimentation workflows).
- **Tools:** Python, Google Colab, Ultralytics, OpenCV, NumPy, pandas.
- **Hardware context:** Raspberry Pi 5 and USB webcam were part of the intended/used robotics setup; check individual scripts and notebook cells for the exact hardware workflow.

## Repository contents

- `notebooks/robotex.ipynb` — original experimentation notebook, with outputs cleared and embedded API keys redacted.
- `src/realtime_detection.py` — webcam inference and on-screen detection overlay.
- `src/evaluate_model.py` — webcam-based evaluation logger that writes detections to CSV and a summary.
- `src/camera_test.py`, `src/camera_index.py`, `src/identify_camera.py` — camera testing utilities.
- `models/competition_final.pt` and `models/final_v2.pt` — supplied model checkpoints.

## Setup

Use Python 3.10 or another Python version supported by the selected Ultralytics release.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Run real-time webcam detection from the repository root:

```bash
python src/realtime_detection.py
```

If your webcam is not camera index `0`, change `CAMERA` in the script. Press `q` to exit.

**Model path note:** the scripts use relative paths under `models/`. Run them from the repository root. If you change model filenames, update the path in the relevant script.

## Dataset and API-key safety

The training notebook originally contained Roboflow API-key strings. Those strings have been replaced with placeholders in this public copy. If the keys were real, revoke/regenerate them in Roboflow before using them again. Never commit API keys, private datasets, credentials, or personal file paths.

The dataset itself is not included. Dataset access and redistribution rights should be checked before sharing it publicly.

## Evaluation notes

Previously documented project metrics were mAP50 = 0.574 and mAP50-95 = 0.323. These are historical reported values and have not been independently recomputed from the two included checkpoints in this packaging step; verify which checkpoint and evaluation run produced them before presenting them as final model results.

## Limitations

- The included webcam scripts are local inference utilities; they are not a complete robot-control stack.
- Detection quality depends on the dataset, lighting, camera position, confidence threshold, and test conditions.
- The repository does not claim competition victory or fully autonomous firefighting.

## Acknowledgements

Developed as a student robotics/computer-vision project for Robotex. Confirm team permissions before publishing team-owned code or model weights.
