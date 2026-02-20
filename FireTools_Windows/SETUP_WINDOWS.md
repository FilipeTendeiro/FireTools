# Windows Setup Guide

Complete step-by-step setup guide for FireTools on Windows.

## Prerequisites Checklist

Before installing Python packages, ensure you have:

- [ ] Windows 10 or 11
- [ ] NVIDIA GPU with CUDA support (optional but recommended)
- [ ] Internet connection
- [ ] Administrator privileges (for installing system software)

## Step 1: Install Python 3.11

Python 3.11 is required and includes pip (Python package manager).

### Using winget (Recommended)

Open PowerShell and run:

```powershell
winget install --id Python.Python.3.11 -e
```

### Alternative: Manual Installation

1. Download from [python.org](https://www.python.org/downloads/release/python-3119/)
2. Run installer
3. **Important:** Check ✓ "Add python.exe to PATH"
4. Check ✓ "Install pip"
5. Click "Install Now"

### Verify Installation

**Close and reopen your terminal**, then verify:

```powershell
py -3.11 --version
# Should show: Python 3.11.9 (or similar)

py -3.11 -m pip --version
# Should show: pip 24.x.x (or similar)
```

**Note:** Use `py -3.11 -m pip` instead of just `pip` if you have multiple Python versions.

## Step 2: Install FFmpeg

FFmpeg is required for video processing and conversion.

### Using winget (Recommended)

```powershell
winget install --id Gyan.FFmpeg -e
```

### Verify Installation

**Close and reopen your terminal** (FFmpeg is added to PATH), then verify:

```powershell
ffmpeg -version
# Should show: ffmpeg version ...
```

**If ffmpeg is not recognized:** You may need to manually add it to PATH **or reboot your computer.**

## Step 3: Clone the Repository

```powershell
cd D:\
git clone <repository-url> FireTools_Windows
cd FireTools_Windows
```

**All following steps (4–8) are run from this project folder** (FireTools_Windows). If you open a new terminal, run `cd` to your clone path first (e.g. `cd G:\GitHub\FireTools_Windows` or `cd D:\FireTools_Windows`).

## Step 4: Create Virtual Environment (Recommended)

From the **FireTools_Windows** folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Note: If you get a script execution error, run:** 

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

## Step 5: Install PyTorch with CUDA Support

**CRITICAL:** Install PyTorch BEFORE other dependencies to get GPU support.

**If you created a virtual environment (Step 4):** With `.venv` activated, use `python -m pip` or `pip` so packages install into the venv. Using `py -3.11 -m pip` can install to global Python instead.

### For CUDA 12.1 (Recommended for RTX 30xx/40xx GPUs)

```powershell
# With venv activated (recommended):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Without venv (global install):
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CUDA 11.8 (For older GPUs)

```powershell
# With venv activated (recommended):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Without venv (global install):
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Detection

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**Expected output:**
```
CUDA available: True
GPU: Your GPU model
```

**If CUDA is False:** Your GPU may not support CUDA, or you need to install NVIDIA drivers. The app will still work on CPU (slower).

## Step 6: Install Python Dependencies

From the **FireTools_Windows** folder (where `requirements.txt` is), with venv activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Without venv (global install): use `py -3.11 -m pip` instead of `python -m pip`.

This will install FastAPI, OpenCV, and other Python packages.

## Step 7: Verify Model Files

The application requires two pre-trained models (included with repository):

- `models/best.pt` (6MB) - YOLOv8n detection model
- `models/best_deeplabv3.pth` (103MB) - DeepLabv3+ segmentation model

**Verify the models are present:**

```powershell
dir models\
# Should show best.pt and best_deeplabv3.pth
```

## Step 8: Start the Application

```powershell
cd scripts
py -3.11 app.py
```

The application will be available at **http://localhost:8000**

## Troubleshooting

### "pip is not recognized"

**Cause:** Terminal hasn't refreshed PATH after Python installation.

**Solution:**
1. Close and reopen your terminal
2. Or use: `py -3.11 -m pip` instead of `pip`

### "python is not recognized" or opens Microsoft Store

**Cause:** Windows App Execution Aliases redirect `python` to the Store.

**Solution:**
- Use `py -3.11` instead of `python`
- Or disable aliases: Settings → Apps → Advanced app settings → App execution aliases → Turn OFF python.exe and python3.exe

### "ffmpeg is not recognized"

**Cause:** Terminal hasn't refreshed PATH after FFmpeg installation.

**Solution:**
1. Close and reopen your terminal
2. If still not working, reboot your computer
3. Verify FFmpeg was installed: Check if `C:\Program Files\ffmpeg` exists

### GPU Not Detected (torch.cuda.is_available() = False)

**Possible causes:**
1. Installed CPU-only PyTorch
2. NVIDIA drivers not installed
3. GPU doesn't support CUDA

**Solution:**
1. Uninstall PyTorch: `py -3.11 -m pip uninstall torch torchvision torchaudio -y`
2. Reinstall with CUDA: `py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Update NVIDIA drivers from [nvidia.com](https://www.nvidia.com/download/index.aspx)

### Port 8000 Already in Use

**Solution:** Change the port in `scripts/app.py`:

```python
uvicorn.run("app:app", host="0.0.0.0", port=8001)  # Changed to 8001
```

### Video Conversion Errors

**Cause:** FFmpeg not in PATH or corrupted video file.

**Solution:**
1. Verify FFmpeg: `ffmpeg -version`
2. Check disk space
3. Try a different video file
4. Check logs in `logs/` directory

## Quick Reference Commands

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install package
py -3.11 -m pip install <package-name>

# Update all packages
py -3.11 -m pip install --upgrade -r requirements.txt

# Check Python version
py -3.11 --version

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Start application
cd scripts
py -3.11 app.py
```

## System Requirements

### Minimum
- Python 3.11
- 8GB RAM
- 5GB disk space
- Any CPU

### Recommended
- Python 3.11
- 16GB RAM
- 10GB disk space
- NVIDIA GPU with 4GB+ VRAM
- CUDA 12.1 support

## Next Steps

After installation:
1. Open http://localhost:8000
2. Upload a video or image for detection
3. Try the segmentation feature
4. Check the gallery for results

For more information, see [README.md](README.md).
