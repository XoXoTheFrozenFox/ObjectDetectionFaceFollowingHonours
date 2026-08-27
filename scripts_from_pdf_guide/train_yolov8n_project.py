from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(
    r"C:\PATH_TO\ObjectDetectionFaceFollowingHonours"
)
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
RUN_NAME = "wider_face_yolov8n"
RUN_DIR = RESULTS_DIR / RUN_NAME
BEST_MODEL = RUN_DIR / "weights" / "best.pt"
LAST_MODEL = RUN_DIR / "weights" / "last.pt"
EXPORTED_BEST = RESULTS_DIR / "wider_face_yolov8n_best.pt"
EXPORTED_LAST = RESULTS_DIR / "wider_face_yolov8n_last.pt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")
model.train(
    data=str(DATA_YAML),
    epochs=200,
    imgsz=640,
    batch=16,
    patience=50,
    device=0,
    workers=0,
    project=str(RESULTS_DIR),
    name=RUN_NAME,
    pretrained=True,
    plots=True,
)

model = YOLO(str(BEST_MODEL))
model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    project=str(RESULTS_DIR),
    name=f"{RUN_NAME}_test",
    plots=True,
)

shutil.copy2(BEST_MODEL, EXPORTED_BEST)
shutil.copy2(LAST_MODEL, EXPORTED_LAST)
print(f"Best model copied to: {EXPORTED_BEST}")
print(f"Last model copied to: {EXPORTED_LAST}")
