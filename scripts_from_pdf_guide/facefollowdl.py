from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pantilthat
from picamera2 import Picamera2
from ultralytics import YOLO


# =============================================================================
# Absolute Raspberry Pi paths
# =============================================================================
PROJECT_DIR = Path(
    "/PATH_TO/FollowFaceOBD/FollowFaceDL"
)

# This is a folder, not a single file.
MODEL_PATH = (
    PROJECT_DIR
    / "wider_face_yolov8n_best_ncnn_model"
)


# =============================================================================
# Camera and model settings
# =============================================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# This must match the size used by export_face_model_to_ncnn.py.
MODEL_IMAGE_SIZE = 320

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20

# Rotate the camera image by 180 degrees so the video, detections and
# tracking coordinates are all upright.
ROTATE_180 = True

# Set False when running without a monitor or graphical desktop.
SHOW_WINDOW = True
WINDOW_NAME = "YOLOv8n Face Following"


# =============================================================================
# Pan-Tilt HAT settings
# =============================================================================
START_PAN = 0.0
START_TILT = 0.0

# Conservative limits protect the mechanism and camera ribbon cable.
PAN_MIN = -80.0
PAN_MAX = 80.0
TILT_MIN = -55.0
TILT_MAX = 55.0

# Direction settings for this assembled pan-tilt unit.
#
# The displayed image is rotated by 180 degrees. The horizontal correction
# uses a negative direction on this mount, while the physical tilt servo uses
# the opposite sign. These values make both axes move towards the detected
# face rather than away from it.
PAN_DIRECTION = -1.0
TILT_DIRECTION = 1.0

# The camera does not move while the face centre is inside this area.
# These are fractions of half the image width and height.
DEAD_ZONE_X = 0.12
DEAD_ZONE_Y = 0.14

# Desired vertical position of the centre point, measured from the top.
# 0.50 is the exact middle. 0.40 raises it by 10% of the frame height.
TARGET_CENTRE_Y_RATIO = 0.40

# Movement strength per processed frame.
PAN_GAIN = 8.0
TILT_GAIN = 7.0

# Prevent a large detection error from causing a sudden servo jump.
MAX_PAN_STEP = 4.0
MAX_TILT_STEP = 3.5

# Minimum time between commands sent to the HAT.
SERVO_COMMAND_INTERVAL = 0.04

# Keep the servos powered while centred so the camera holds its position.
# The program disables both servos when it exits.
HOLD_POSITION_WHILE_RUNNING = True


# =============================================================================
# Face-lock settings
# =============================================================================
# While locked, detections are matched to the previously tracked face.
# The servos hold their last position while the target is temporarily missing.
LOST_FRAMES_BEFORE_UNLOCK = 25

# Maximum normalised movement accepted for the locked face in one frame.
# It becomes more tolerant while recovering after a missed detection.
NORMAL_MATCH_DISTANCE = 0.38
RECOVERY_MATCH_DISTANCE = 0.62

# Weight used to discourage switching to a face of a very different size.
AREA_CHANGE_WEIGHT = 0.18


@dataclass(frozen=True)
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


class FaceLock:
    """
    Locks onto one detected face.

    This is detection-based association, not face recognition. It is designed
    to keep following the same face when one main person is in view.
    """

    def __init__(self) -> None:
        self.locked = False
        self.missed_frames = 0
        self.previous_centre: tuple[float, float] | None = None
        self.previous_area: float | None = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def reset(self) -> None:
        self.locked = False
        self.missed_frames = 0
        self.previous_centre = None
        self.previous_area = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def _lock_to(self, face: FaceDetection) -> FaceDetection:
        centre_x, centre_y = face.centre

        if self.previous_centre is not None:
            movement_x = centre_x - self.previous_centre[0]
            movement_y = centre_y - self.previous_centre[1]

            self.velocity_x = (
                0.70 * self.velocity_x
                + 0.30 * movement_x
            )
            self.velocity_y = (
                0.70 * self.velocity_y
                + 0.30 * movement_y
            )

        self.previous_centre = (centre_x, centre_y)
        self.previous_area = float(face.area)
        self.missed_frames = 0
        self.locked = True
        return face

    def select(
        self,
        faces: list[FaceDetection],
        frame_width: int,
        frame_height: int,
    ) -> FaceDetection | None:
        if not faces:
            if self.locked:
                self.missed_frames += 1

                if self.missed_frames > LOST_FRAMES_BEFORE_UNLOCK:
                    self.reset()

            return None

        # First lock: choose the largest clear face, normally the person
        # closest to the camera.
        if not self.locked or self.previous_centre is None:
            first_face = max(
                faces,
                key=lambda face: face.area * face.confidence,
            )
            return self._lock_to(first_face)

        predicted_x = self.previous_centre[0] + self.velocity_x
        predicted_y = self.previous_centre[1] + self.velocity_y
        frame_diagonal = math.hypot(frame_width, frame_height)

        best_face: FaceDetection | None = None
        best_cost = float("inf")
        best_distance = float("inf")

        for face in faces:
            centre_x, centre_y = face.centre

            centre_distance = math.hypot(
                centre_x - predicted_x,
                centre_y - predicted_y,
            ) / frame_diagonal

            area_cost = 0.0

            if self.previous_area is not None and self.previous_area > 0:
                area_ratio = max(
                    face.area / self.previous_area,
                    self.previous_area / face.area,
                )
                area_cost = math.log(max(1.0, area_ratio))

            total_cost = (
                centre_distance
                + AREA_CHANGE_WEIGHT * area_cost
            )

            if total_cost < best_cost:
                best_face = face
                best_cost = total_cost
                best_distance = centre_distance

        allowed_distance = (
            RECOVERY_MATCH_DISTANCE
            if self.missed_frames > 0
            else NORMAL_MATCH_DISTANCE
        )

        if best_face is not None and best_distance <= allowed_distance:
            return self._lock_to(best_face)

        # A different face may have appeared far away. Do not immediately
        # switch targets; hold the current servo position first.
        self.missed_frames += 1

        if self.missed_frames > LOST_FRAMES_BEFORE_UNLOCK:
            self.reset()

        return None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def remove_dead_zone(error: float, dead_zone: float) -> float:
    """Remove the centre dead zone and rescale the remaining error."""
    magnitude = abs(error)

    if magnitude <= dead_zone:
        return 0.0

    scaled = (magnitude - dead_zone) / (1.0 - dead_zone)
    return math.copysign(scaled, error)


def read_detections(result) -> list[FaceDetection]:
    faces: list[FaceDetection] = []

    if result.boxes is None or len(result.boxes) == 0:
        return faces

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    for box, confidence, class_id in zip(
        boxes_xyxy,
        confidences,
        class_ids,
    ):
        # The custom WIDER FACE model has one class: 0 = face.
        if int(class_id) != 0:
            continue

        x1, y1, x2, y2 = box

        faces.append(
            FaceDetection(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                confidence=float(confidence),
            )
        )

    return faces


def draw_interface(
    frame: np.ndarray,
    faces: list[FaceDetection],
    target: FaceDetection | None,
    face_lock: FaceLock,
    pan_angle: float,
    tilt_angle: float,
    fps: float,
    is_centred: bool,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    centre_x = frame_width // 2
    centre_y = int(frame_height * TARGET_CENTRE_Y_RATIO)

    dead_zone_half_width = int(
        DEAD_ZONE_X * frame_width / 2.0
    )
    dead_zone_half_height = int(
        DEAD_ZONE_Y * frame_height / 2.0
    )

    cv2.rectangle(
        frame,
        (
            centre_x - dead_zone_half_width,
            centre_y - dead_zone_half_height,
        ),
        (
            centre_x + dead_zone_half_width,
            centre_y + dead_zone_half_height,
        ),
        (255, 255, 0),
        1,
    )

    cv2.line(
        frame,
        (centre_x - 12, centre_y),
        (centre_x + 12, centre_y),
        (255, 255, 0),
        1,
    )
    cv2.line(
        frame,
        (centre_x, centre_y - 12),
        (centre_x, centre_y + 12),
        (255, 255, 0),
        1,
    )

    for face in faces:
        is_target = face is target
        colour = (0, 255, 0) if is_target else (180, 180, 180)
        thickness = 3 if is_target else 1

        cv2.rectangle(
            frame,
            (face.x1, face.y1),
            (face.x2, face.y2),
            colour,
            thickness,
        )

        label = (
            f"LOCKED {face.confidence:.2f}"
            if is_target
            else f"face {face.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (face.x1, max(20, face.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2 if is_target else 1,
            cv2.LINE_AA,
        )

    if target is not None:
        target_x, target_y = target.centre

        cv2.line(
            frame,
            (int(target_x), int(target_y)),
            (centre_x, centre_y),
            (0, 255, 0),
            1,
        )
        cv2.circle(
            frame,
            (int(target_x), int(target_y)),
            5,
            (0, 255, 0),
            -1,
        )

    if target is not None and is_centred:
        status = "LOCKED - CENTRED"
        status_colour = (0, 255, 0)
    elif target is not None:
        status = "LOCKED - FOLLOWING"
        status_colour = (0, 255, 255)
    elif face_lock.locked:
        status = (
            "TARGET LOST - HOLDING "
            f"({face_lock.missed_frames}/{LOST_FRAMES_BEFORE_UNLOCK})"
        )
        status_colour = (0, 165, 255)
    else:
        status = "SEARCHING FOR A FACE"
        status_colour = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_colour,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Pan {pan_angle:+.1f}  "
            f"Tilt {tilt_angle:+.1f}  "
            f"FPS {fps:.1f}"
        ),
        (15, frame_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Q/Esc quit | C centre camera | U unlock face",
        (15, frame_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            "The NCNN model folder was not found at the required absolute "
            f"path:\n{MODEL_PATH}\n\n"
            "Copy the complete exported NCNN folder from Windows."
        )

    print("Loading face-detection model...")
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Starting Raspberry Pi camera...")

    camera = Picamera2()
    camera_config = camera.create_video_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": "RGB888",
        },
        buffer_count=2,
    )
    camera.configure(camera_config)
    camera.start()
    time.sleep(1.0)

    pan_angle = START_PAN
    tilt_angle = START_TILT
    last_servo_command = 0.0

    face_lock = FaceLock()

    # A timeout of zero keeps the drive signal active while the program runs.
    pantilthat.idle_timeout(
        0 if HOLD_POSITION_WHILE_RUNNING else 2
    )

    pantilthat.pan(int(round(pan_angle)))
    pantilthat.tilt(int(round(tilt_angle)))
    time.sleep(0.8)

    fps = 0.0
    previous_frame_time = time.perf_counter()

    print("\nFace following is running.")
    print("The camera holds its position until a face is detected.")
    print("Press Q or Esc in the video window to stop.\n")

    try:
        while True:
            frame = camera.capture_array()

            if ROTATE_180:
                frame = cv2.rotate(
                    frame,
                    cv2.ROTATE_180,
                )

            results = model.predict(
                source=frame,
                imgsz=MODEL_IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                device="cpu",
                verbose=False,
            )

            result = results[0]
            faces = read_detections(result)

            frame_height, frame_width = frame.shape[:2]

            target = face_lock.select(
                faces=faces,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            is_centred = False

            if target is not None:
                target_x, target_y = target.centre

                desired_centre_x = frame_width / 2.0
                desired_centre_y = (
                    frame_height * TARGET_CENTRE_Y_RATIO
                )

                error_x = (
                    target_x - desired_centre_x
                ) / (frame_width / 2.0)

                error_y = (
                    target_y - desired_centre_y
                ) / (frame_height / 2.0)

                controlled_error_x = remove_dead_zone(
                    error_x,
                    DEAD_ZONE_X,
                )
                controlled_error_y = remove_dead_zone(
                    error_y,
                    DEAD_ZONE_Y,
                )

                is_centred = (
                    controlled_error_x == 0.0
                    and controlled_error_y == 0.0
                )

                current_time = time.perf_counter()

                if (
                    not is_centred
                    and current_time - last_servo_command
                    >= SERVO_COMMAND_INTERVAL
                ):
                    pan_step = clamp(
                        PAN_DIRECTION
                        * PAN_GAIN
                        * controlled_error_x,
                        -MAX_PAN_STEP,
                        MAX_PAN_STEP,
                    )

                    tilt_step = clamp(
                        TILT_DIRECTION
                        * TILT_GAIN
                        * controlled_error_y,
                        -MAX_TILT_STEP,
                        MAX_TILT_STEP,
                    )

                    new_pan_angle = clamp(
                        pan_angle + pan_step,
                        PAN_MIN,
                        PAN_MAX,
                    )

                    new_tilt_angle = clamp(
                        tilt_angle + tilt_step,
                        TILT_MIN,
                        TILT_MAX,
                    )

                    if (
                        abs(new_pan_angle - pan_angle) >= 0.20
                        or abs(new_tilt_angle - tilt_angle) >= 0.20
                    ):
                        pan_angle = new_pan_angle
                        tilt_angle = new_tilt_angle

                        pantilthat.pan(
                            int(round(pan_angle))
                        )
                        pantilthat.tilt(
                            int(round(tilt_angle))
                        )

                        last_servo_command = current_time

            current_frame_time = time.perf_counter()
            frame_duration = max(
                1e-6,
                current_frame_time - previous_frame_time,
            )
            instantaneous_fps = 1.0 / frame_duration
            previous_frame_time = current_frame_time

            if fps == 0.0:
                fps = instantaneous_fps
            else:
                fps = 0.90 * fps + 0.10 * instantaneous_fps

            if SHOW_WINDOW:
                draw_interface(
                    frame=frame,
                    faces=faces,
                    target=target,
                    face_lock=face_lock,
                    pan_angle=pan_angle,
                    tilt_angle=tilt_angle,
                    fps=fps,
                    is_centred=is_centred,
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("u"):
                    face_lock.reset()
                    print("Face lock cleared.")

                if key == ord("c"):
                    face_lock.reset()
                    pan_angle = START_PAN
                    tilt_angle = START_TILT

                    pantilthat.pan(
                        int(round(pan_angle))
                    )
                    pantilthat.tilt(
                        int(round(tilt_angle))
                    )

                    print("Camera returned to the centre position.")

    except KeyboardInterrupt:
        print("\nStopped with Ctrl+C.")

    finally:
        print("Stopping camera and disabling servos...")

        camera.stop()

        pantilthat.servo_enable(1, False)
        pantilthat.servo_enable(2, False)

        cv2.destroyAllWindows()

        print("Shutdown complete.")


if __name__ == "__main__":
    main()
