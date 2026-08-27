import time

import pantilthat


# Tilt is servo channel 2 on the Pimoroni Pan-Tilt HAT.
SAFE_TEST_ANGLES = (-20, 0, 20, 0)

try:
    for angle in SAFE_TEST_ANGLES:
        print(f"Tilt angle: {angle}")
        pantilthat.tilt(angle)
        time.sleep(1.0)
finally:
    pantilthat.tilt(0)
    time.sleep(0.5)
    pantilthat.servo_enable(2, False)
