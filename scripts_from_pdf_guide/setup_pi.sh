#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  python3-venv python3-pip python3-picamera2 python3-opencv \
  python3-smbus i2c-tools git

python3 -m venv --system-site-packages ~/vision-env
source ~/vision-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt

printf '\nSetup complete. Reboot if the camera or I2C interface was just enabled.\n'
