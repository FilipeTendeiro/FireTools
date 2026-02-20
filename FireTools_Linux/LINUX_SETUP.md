# Linux Setup Guide

Quick setup guide for FireTools on Linux (Ubuntu/Debian).

## Prerequisites

- Ubuntu 20.04+ or Debian 11+
- Python 3.8 or higher
- FFmpeg
- (Optional) NVIDIA GPU with CUDA support

## Installation

### 1. Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install Python 3 and pip
sudo apt-get install python3 python3-pip python3-venv

# Install FFmpeg
sudo apt-get install ffmpeg

# Verify installations
python3 --version
pip3 --version
ffmpeg -version
```

### 2. Clone Repository

```bash
git clone <repository-url> FireTools_Linux
cd FireTools_Linux
```

### 3. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch with CUDA (for GPU support)

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify GPU detection:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output: `CUDA available: True` (if GPU is available)

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Models

Check that model files are present:

```bash
ls -lh models/
# Should show:
# best.pt (6MB)
# best_deeplabv3.pth (103MB)
```

### 7. Start the Application

```bash
cd scripts
python3 app.py
```

The application will be available at **http://localhost:8000**

## NVIDIA GPU Setup (Optional)

If you have an NVIDIA GPU and want to use it for inference:

### 1. Install NVIDIA Drivers

```bash
# Check if drivers are installed
nvidia-smi

# If not installed, add NVIDIA PPA and install
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-535  # or latest version
sudo reboot
```

### 2. Install CUDA Toolkit (Optional)

```bash
# Download from NVIDIA website or use:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 3. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

## Troubleshooting

### FFmpeg Not Found

```bash
# Reinstall FFmpeg
sudo apt-get install --reinstall ffmpeg

# Check PATH
which ffmpeg
```

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Permission Denied

```bash
# Make sure you have proper permissions
sudo chown -R $USER:$USER ~/FireTools_Linux

# Or run with sudo (not recommended)
sudo python3 app.py
```

### Port 8000 Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or change port in scripts/app.py
```

## Running as a Service (Optional)

To run FireTools as a systemd service:

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/firetools.service
```

### 2. Add Service Configuration

```ini
[Unit]
Description=FireTools Linux
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/FireTools_Linux/scripts
Environment="PATH=/path/to/FireTools_Linux/.venv/bin"
ExecStart=/path/to/FireTools_Linux/.venv/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable firetools
sudo systemctl start firetools
sudo systemctl status firetools
```

## Quick Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Start application
cd scripts && python3 app.py

# Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# View logs (if running as service)
sudo journalctl -u firetools -f
```

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 5GB disk space
- Any CPU

### Recommended
- Python 3.10+
- 16GB RAM
- 10GB disk space
- NVIDIA GPU with 4GB+ VRAM
- CUDA 12.1 support

---

For general usage and API documentation, see [README.md](README.md).
