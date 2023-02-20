#!/bin/bash
# Скрипт запускается до перехода на узел

python3 -m venv .venv

source .venv/bin/activate
pip install numpy
pip install pythreshold
pip install scikit-image
pip install scikit-learn
pip install opencv-python
pip install h5py
pip install tqdm
pip install torch
pip install pytorch-lightning
