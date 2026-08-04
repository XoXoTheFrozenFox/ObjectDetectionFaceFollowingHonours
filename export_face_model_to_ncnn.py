from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(
    r"C:\Users\berna\Downloads\ObjectDetectionFaceFollowingHonours"
)

RESULTS_DIR = PROJECT_ROOT / "results"
SOURCE_MODEL = RESULTS_DIR / "wider_face_yolov8n_best.pt"
EXPORTED_MODEL_DIR = RESULTS_DIR / "wider_face_yolov8n_best_ncnn_model"

IMAGE_SIZE = 320
BATCH_SIZE = 1
RESET_EXISTING_EXPORT = True


def main() -> None:
    print("Exporting the trained YOLOv8n face detector to NCNN")
    print(f"Source model: {SOURCE_MODEL}")
    print(f"Export folder: {EXPORTED_MODEL_DIR}")
    print(f"Image size: {IMAGE_SIZE}")
    print()

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            "The trained model was not found at the required absolute path:\n"
            f"{SOURCE_MODEL}\n\n"
            "Finish YOLOv8n training first and confirm that this file exists."
        )

    if RESET_EXISTING_EXPORT and EXPORTED_MODEL_DIR.exists():
        print(f"Removing previous NCNN export: {EXPORTED_MODEL_DIR}")
        shutil.rmtree(EXPORTED_MODEL_DIR)

    model = YOLO(str(SOURCE_MODEL))

    exported_path = Path(
        model.export(
            format="ncnn",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
        )
    )

    if not exported_path.is_dir():
        raise FileNotFoundError(
            "Ultralytics reported that export finished, but the NCNN model "
            f"folder was not found:\n{exported_path}"
        )

    param_files = list(exported_path.glob("*.param"))
    bin_files = list(exported_path.glob("*.bin"))

    if not param_files or not bin_files:
        raise FileNotFoundError(
            "The exported folder does not contain both an NCNN .param file "
            f"and an NCNN .bin file:\n{exported_path}"
        )

    print("\nNCNN export completed successfully.")
    print(f"Copy this entire folder to the Raspberry Pi:\n{exported_path}")
    print("\nDo not copy only one file. The complete model folder is required.")


if __name__ == "__main__":
    main()
