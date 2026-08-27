import time

import pantilthat


# Pan is servo channel 1 on the Pimoroni Pan-Tilt HAT.
SAFE_TEST_ANGLES = (-25, 0, 25, 0)

try:
    for angle in SAFE_TEST_ANGLES:
        print(f"Pan angle: {angle}")
        pantilthat.pan(angle)
        time.sleep(1.0)
finally:
    pantilthat.pan(0)
    time.sleep(0.5)
    pantilthat.servo_enable(1, False)
