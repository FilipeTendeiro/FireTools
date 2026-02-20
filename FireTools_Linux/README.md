# FireTools — Linux

Web-based wildfire detection and segmentation platform.  
For full project information see the [main README](../README.md).

---

## Requirements

- Ubuntu 20.04+ / Debian 11+ / macOS
- Python 3.8 or higher
- FFmpeg
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with 4 GB+ VRAM recommended (CUDA 12.1); CPU fallback supported

---

## Setup

For the complete step-by-step installation guide see **[LINUX_SETUP.md](LINUX_SETUP.md)**.

### Quick Start

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install python3 python3-pip python3-venv ffmpeg
# macOS: brew install python3 ffmpeg

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Run the application
cd scripts
python3 app.py
```

Open **http://localhost:8000** in your browser.

---

## Models

Pre-trained models are included in `models/` (downloaded via Git LFS):

- `best.pt` (6 MB) — YOLOv8n fire/smoke detection
- `best_deeplabv3.pth` (103 MB) — DeepLabv3+ segmentation

Verify they are present before starting:

```bash
ls -lh models/
```

---

## Common Issues

|         Problem         |                                                     Solution                                               |
|-------------------------|------------------------------------------------------------------------------------------------------------|
| GPU not detected        | Run `python3 -c "import torch; print(torch.cuda.is_available())"` and see [LINUX_SETUP.md](LINUX_SETUP.md) |
| FFmpeg not found        |                                 `sudo apt-get install --reinstall ffmpeg`                                  |
| Permission denied       |                               `sudo chown -R $USER:$USER ~/FireTools_Linux`                                |
| Port 8000 in use        |             `sudo lsof -i :8000` then kill the process, or change port in `scripts/app.py`                 | 

For detailed troubleshooting and optional GPU/CUDA setup see **[LINUX_SETUP.md](LINUX_SETUP.md)**.
