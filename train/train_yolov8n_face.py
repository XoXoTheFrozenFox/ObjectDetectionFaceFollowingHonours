from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(
    r"C:\Users\berna\Downloads\ObjectDetectionFaceFollowingHonours"
)

DATASET_DIR = PROJECT_ROOT / "dataset"

# The data.yaml file is directly inside the dataset folder.
DATA_YAML = DATASET_DIR / "data.yaml"

RESULTS_DIR = PROJECT_ROOT / "results"

RUN_NAME = "wider_face_yolov8n"
RUN_DIR = RESULTS_DIR / RUN_NAME

BEST_MODEL = RUN_DIR / "weights" / "best.pt"
LAST_MODEL = RUN_DIR / "weights" / "last.pt"

EXPORTED_BEST_MODEL = RESULTS_DIR / "wider_face_yolov8n_best.pt"
EXPORTED_LAST_MODEL = RESULTS_DIR / "wider_face_yolov8n_last.pt"

TEST_RUN_NAME = f"{RUN_NAME}_test"
TEST_RUN_DIR = RESULTS_DIR / TEST_RUN_NAME

EPOCHS = 200
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 50

# Use the first NVIDIA GPU.
DEVICE = 0

WORKERS = 0

RANDOM_SEED = 42

RESET_EXISTING_RUN = True


def remove_existing_outputs() -> None:
    if not RESET_EXISTING_RUN:
        return

    for folder in (RUN_DIR, TEST_RUN_DIR):
        if folder.exists():
            print(f"Removing previous folder: {folder}")
            shutil.rmtree(folder)

    for weight_file in (EXPORTED_BEST_MODEL, EXPORTED_LAST_MODEL):
        if weight_file.exists():
            print(f"Removing previous model copy: {weight_file}")
            weight_file.unlink()


def validate_paths() -> None:
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Project folder was not found:\n{PROJECT_ROOT}"
        )

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder was not found:\n{DATASET_DIR}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            "The data.yaml file was not found at the required absolute path:\n"
            f"{DATA_YAML}"
        )


def main() -> None:
    print("YOLOv8n WIDER FACE training")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset YAML: {DATA_YAML}")
    print(f"Results:      {RESULTS_DIR}")
    print()

    validate_paths()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_outputs()

    model = YOLO("yolov8n.pt")

    print("\nStarting training...")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RESULTS_DIR),
        name=RUN_NAME,
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        plots=True,
        save=True,
        seed=RANDOM_SEED,
        deterministic=True,
        close_mosaic=10,
        exist_ok=True,
        verbose=True,
    )

    if not BEST_MODEL.is_file():
        raise FileNotFoundError(
            "Training finished, but the best model was not found at:\n"
            f"{BEST_MODEL}"
        )

    shutil.copy2(BEST_MODEL, EXPORTED_BEST_MODEL)

    if LAST_MODEL.is_file():
        shutil.copy2(LAST_MODEL, EXPORTED_LAST_MODEL)

    print("\nRunning final evaluation on the test split...")

    best_model = YOLO(str(BEST_MODEL))

    metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RESULTS_DIR),
        name=TEST_RUN_NAME,
        plots=True,
        save_json=False,
        exist_ok=True,
        verbose=True,
    )

    print("\nTraining and test evaluation completed.")
    print(f"Best model:       {BEST_MODEL}")
    print(f"Best model copy:  {EXPORTED_BEST_MODEL}")

    if LAST_MODEL.is_file():
        print(f"Last model:       {LAST_MODEL}")
        print(f"Last model copy:  {EXPORTED_LAST_MODEL}")

    print(f"Training results: {RUN_DIR}")
    print(f"Test results:     {TEST_RUN_DIR}")

    try:
        print(f"Test precision:   {metrics.box.mp:.6f}")
        print(f"Test recall:      {metrics.box.mr:.6f}")
        print(f"Test mAP50-95:    {metrics.box.map:.6f}")
        print(f"Test mAP50:       {metrics.box.map50:.6f}")
        print(f"Test mAP75:       {metrics.box.map75:.6f}")
    except AttributeError:
        print(
            "Ultralytics saved the test metrics in the test-results folder."
        )


if __name__ == "__main__":
    main()