from pathlib import Path

from ultralytics import YOLO


DATA_YAML = Path(r"C:\path\to\dataset\data.yaml")
RESULTS_DIR = Path(r"C:\path\to\results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Nano models are a sensible starting point for Raspberry Pi deployment.
model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=200,
    imgsz=640,
    batch=16,
    patience=50,
    device=0,          # Use "cpu" if no CUDA-capable GPU is available.
    workers=0,         # Reliable default for Windows teaching machines.
    project=str(RESULTS_DIR),
    name="yolov8n_custom",
    pretrained=True,
    plots=True,
    hsv_h=0.015,
    hsv_s=0.70,
    hsv_v=0.40,
    translate=0.10,
    scale=0.50,
    fliplr=0.50,
)

model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    device=0,
    workers=0,
)
