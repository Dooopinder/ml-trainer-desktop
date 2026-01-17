ML Trainer Desktop App

A native desktop (non-browser) machine learning application.

Features:
- Classification
- Regression
- Clustering
- Anomaly Detection
- Dimensionality Reduction

Run (development):

conda create -n ml-desktop python=3.10 -y
conda activate ml-desktop
pip install -r requirements.txt
python main.py

Build EXE (Windows):

pip install pyinstaller
pyinstaller --noconsole --onefile --name "ML-Trainer" main.py
