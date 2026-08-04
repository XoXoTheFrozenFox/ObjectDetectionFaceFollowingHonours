from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

PROJECT_ROOT = Path(
    r"C:\Users\berna\Downloads\ObjectDetectionFaceFollowingHonours"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "wider_face_yolov8n"
    / "weights"
    / "best.pt"
)

CAMERA_INDEX = 0

IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.50

IOU_THRESHOLD = 0.45

DEVICE = 0

WINDOW_TITLE = "YOLO Face Detection Camera"


class FaceDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1100x760")
        self.root.minsize(850, 620)

        self.model: Optional[YOLO] = None
        self.camera: Optional[cv2.VideoCapture] = None

        self.running = False
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        self.frames_processed = 0
        self.current_face_count = 0

        self.confidence_variable = tk.DoubleVar(
            value=CONFIDENCE_THRESHOLD
        )

        self.status_variable = tk.StringVar(
            value="Loading model..."
        )

        self.face_count_variable = tk.StringVar(
            value="Faces detected: 0"
        )

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        self.root.after(100, self.load_model)

    def create_widgets(self) -> None:
        main_frame = tk.Frame(
            self.root,
            padx=12,
            pady=12,
        )
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        title_label = tk.Label(
            main_frame,
            text="YOLO Face Detection",
            font=("Arial", 20, "bold"),
        )
        title_label.pack(pady=(0, 10))

        self.video_label = tk.Label(
            main_frame,
            text="Camera preview will appear here",
            bg="black",
            fg="white",
            font=("Arial", 14),
        )
        self.video_label.pack(
            fill=tk.BOTH,
            expand=True,
        )

        controls_frame = tk.Frame(
            main_frame,
            pady=10,
        )
        controls_frame.pack(
            fill=tk.X,
        )

        self.start_button = tk.Button(
            controls_frame,
            text="Start camera",
            width=16,
            command=self.start_camera,
            state=tk.DISABLED,
        )
        self.start_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        self.stop_button = tk.Button(
            controls_frame,
            text="Stop camera",
            width=16,
            command=self.stop_camera,
            state=tk.DISABLED,
        )
        self.stop_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        self.exit_button = tk.Button(
            controls_frame,
            text="Exit",
            width=12,
            command=self.close_application,
        )
        self.exit_button.pack(
            side=tk.RIGHT,
            padx=5,
        )

        confidence_frame = tk.Frame(main_frame)
        confidence_frame.pack(
            fill=tk.X,
            pady=(0, 8),
        )

        confidence_label = tk.Label(
            confidence_frame,
            text="Confidence threshold:",
            font=("Arial", 10),
        )
        confidence_label.pack(side=tk.LEFT)

        confidence_slider = tk.Scale(
            confidence_frame,
            from_=0.10,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_variable,
            length=300,
        )
        confidence_slider.pack(
            side=tk.LEFT,
            padx=10,
        )

        self.face_count_label = tk.Label(
            confidence_frame,
            textvariable=self.face_count_variable,
            font=("Arial", 11, "bold"),
        )
        self.face_count_label.pack(
            side=tk.RIGHT,
            padx=10,
        )

        status_label = tk.Label(
            main_frame,
            textvariable=self.status_variable,
            anchor="w",
            relief=tk.SUNKEN,
            padx=8,
            pady=5,
        )
        status_label.pack(
            fill=tk.X,
        )

    def load_model(self) -> None:
        if not MODEL_PATH.is_file():
            self.status_variable.set("Model file not found.")

            messagebox.showerror(
                "Model not found",
                "The trained model could not be found at:\n\n"
                f"{MODEL_PATH}\n\n"
                "Check that training completed successfully.",
            )
            return

        try:
            self.model = YOLO(str(MODEL_PATH))

        except Exception as error:
            self.status_variable.set("Failed to load model.")

            messagebox.showerror(
                "Model loading error",
                f"Could not load the YOLO model:\n\n{error}",
            )
            return

        self.status_variable.set(
            f"Model loaded: {MODEL_PATH.name}"
        )

        self.start_button.config(state=tk.NORMAL)

    def start_camera(self) -> None:
        if self.running:
            return

        if self.model is None:
            messagebox.showerror(
                "Model unavailable",
                "The YOLO model has not been loaded.",
            )
            return

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera error",
                "Could not open the camera.\n\n"
                "Try changing CAMERA_INDEX from 0 to 1.",
            )
            return

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280,
        )
        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720,
        )

        self.running = True

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.status_variable.set(
            "Camera running. Detecting faces..."
        )

        self.update_frame()

    def stop_camera(self) -> None:
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.start_button.config(
            state=tk.NORMAL if self.model is not None else tk.DISABLED
        )
        self.stop_button.config(state=tk.DISABLED)

        self.face_count_variable.set(
            "Faces detected: 0"
        )

        self.status_variable.set(
            "Camera stopped."
        )

        self.video_label.config(
            image="",
            text="Camera preview stopped",
        )

        self.current_photo = None

    def update_frame(self) -> None:
        if not self.running:
            return

        if self.camera is None or self.model is None:
            self.stop_camera()
            return

        success, frame = self.camera.read()

        if not success:
            self.status_variable.set(
                "Failed to read a camera frame."
            )
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        confidence = float(
            self.confidence_variable.get()
        )

        try:
            results = self.model.predict(
                source=frame,
                conf=confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )

        except Exception as error:
            self.stop_camera()

            messagebox.showerror(
                "Detection error",
                f"An error occurred during detection:\n\n{error}",
            )
            return

        annotated_frame = frame.copy()
        face_count = 0

        if results:
            result = results[0]

            if result.boxes is not None:
                face_count = len(result.boxes)

                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    score = float(
                        box.conf[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                    class_id = int(
                        box.cls[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                    class_name = self.get_class_name(
                        class_id
                    )

                    label = (
                        f"{class_name} {score:.2f}"
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    self.draw_label(
                        annotated_frame,
                        label,
                        x1,
                        y1,
                    )

        self.current_face_count = face_count

        self.face_count_variable.set(
            f"Faces detected: {face_count}"
        )

        self.frames_processed += 1

        display_frame = self.resize_for_display(
            annotated_frame
        )

        rgb_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb_frame)

        self.current_photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.config(
            image=self.current_photo,
            text="",
        )

        # Schedule the next frame.
        self.root.after(10, self.update_frame)

    def get_class_name(self, class_id: int) -> str:
        if self.model is None:
            return "face"

        names = self.model.names

        if isinstance(names, dict):
            return str(
                names.get(class_id, "face")
            )

        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return str(names[class_id])

        return "face"

    @staticmethod
    def draw_label(
        frame,
        text: str,
        x: int,
        y: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        text_width, text_height = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )[0]

        label_top = max(
            0,
            y - text_height - 12,
        )

        cv2.rectangle(
            frame,
            (x, label_top),
            (x + text_width + 10, y),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + 5, y - 6),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def resize_for_display(self, frame):
        available_width = max(
            self.video_label.winfo_width(),
            640,
        )

        available_height = max(
            self.video_label.winfo_height(),
            480,
        )

        frame_height, frame_width = frame.shape[:2]

        scale = min(
            available_width / frame_width,
            available_height / frame_height,
        )

        new_width = max(
            1,
            int(frame_width * scale),
        )

        new_height = max(
            1,
            int(frame_height * scale),
        )

        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def close_application(self) -> None:
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        cv2.destroyAllWindows()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()