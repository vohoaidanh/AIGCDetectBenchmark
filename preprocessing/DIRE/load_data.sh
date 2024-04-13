#!/bin/bash

pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio
python3 download.py "1kKtelsG569d4uR7rk4zvJohULtwcRhnP" "RealFakeDB_tiny.zip"