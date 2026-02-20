# FireTools — Windows

Web-based wildfire detection and segmentation platform.  
For full project information see the [main README](../README.md).

---

## Requirements

- Windows 10 or 11
- Python 3.11
- FFmpeg
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with 4 GB+ VRAM recommended (CUDA 12.1); CPU fallback supported

---

## Setup

For the complete step-by-step installation guide see **[SETUP_WINDOWS.md](SETUP_WINDOWS.md)**.

### Quick Start

```powershell
# 1. Install Python 3.11 and FFmpeg (close and reopen terminal after)
winget install --id Python.Python.3.11 -e
winget install --id Gyan.FFmpeg -e

# 2. Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.1 support
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
python -m pip install -r requirements.txt

# 5. Run the application
cd scripts
py -3.11 app.py
```

Open **http://localhost:8000** in your browser.

---

## Models

Pre-trained models are included in `models/` (downloaded via Git LFS):

- `best.pt` (6 MB) — YOLOv8n fire/smoke detection
- `best_deeplabv3.pth` (103 MB) — DeepLabv3+ segmentation

Verify they are present before starting:

```powershell
dir models\
```

---

## Common Issues

|          Problem          |                        Solution                         |
|---------------------------|---------------------------------------------------------|
| `pip not recognized`      | Use `py -3.11 -m pip` instead of `pip`                  |
| `python` opens Store      | Use `py -3.11` instead of `python`                      |
| `ffmpeg not recognized`   | Close and reopen terminal, or reboot                    |
| GPU not detected          | See GPU section in [SETUP_WINDOWS.md](SETUP_WINDOWS.md) |
| Port 8000 in use          | Change port in `scripts/app.py`                         |

For detailed troubleshooting see **[SETUP_WINDOWS.md](SETUP_WINDOWS.md)**.
