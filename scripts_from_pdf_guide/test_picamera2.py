import time

from picamera2 import Picamera2


camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(config)
camera.start()
time.sleep(2)
camera.capture_file("camera_test.jpg")
camera.stop()
print("Saved camera_test.jpg")
