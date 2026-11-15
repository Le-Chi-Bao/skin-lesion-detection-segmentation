# Run in the repository root, Windows PowerShell.
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe outputs\phase2a1\check_cuda.py
.\.venv\Scripts\python.exe -m pip install segmentation-models-pytorch==0.5.0 -c requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe outputs\phase2a1\download_assets.py
.\.venv\Scripts\python.exe outputs\phase2a1\check_checkpoint.py
