ML Trainer Desktop App

A native desktop (non-browser) machine learning application.

Features:
- Classification
- Regression
- Clustering
- Anomaly Detection
- Dimensionality Reduction




🛠️ Running from Source (Development Mode)

If you want to run the app from source code instead of the EXE:

conda create -n ml-desktop python=3.10 -y
conda activate ml-desktop
pip install -r requirements.txt
python main.py

🏗️ Build Windows Executable (EXE)

To build a standalone Windows app:

pip install pyinstaller
pyinstaller --noconsole --onefile --name "ML-Trainer" main.py


The executable will be created in:

dist/ML-Trainer.exe

⚠️ Important Notes

Python 3.10 is required

First launch may take a few seconds (PyInstaller extraction)

Do NOT commit dist/ or build/ folders to git

Upload EXE to GitHub Releases, not source tree

If antivirus flags the EXE → it’s a PyInstaller false positive

📦 Recommended .gitignore
# Python
__pycache__/
*.pyc

# Conda
.env
.venv

# Build artifacts
build/
dist/
*.exe
*.spec

# OS
.DS_Store
Thumbs.db

🚀 Versioning

v1.0.0 → First stable desktop release

Source = GitHub repo

Binary = GitHub Releases
