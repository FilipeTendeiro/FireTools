"""
Configuration file for FireTools Interface
All paths are relative to the project root directory
"""
from pathlib import Path

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).resolve().parent

# Base directories
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATIC_DIR = SCRIPTS_DIR / "static"

# Model paths
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
DEEPLABV3_MODEL_PATH = MODELS_DIR / "best_deeplabv3.pth"

# Script paths
SEGMENTATION_SCRIPT = SCRIPTS_DIR / "unified_fire_smoke_segmentation_video_optimized.py"
INFERENCE_SCRIPT = SCRIPTS_DIR / "run_inference.py"

# Output subdirectories
FRAMES_DIR = OUTPUTS_DIR / "frames"
SEGMENTATION_DIR = OUTPUTS_DIR / "segmentation"
RESULT_DIR = OUTPUTS_DIR / "result"

# Cache files
GALLERY_JSON = SCRIPTS_DIR / "gallery_cache.json"
SEGMENTATION_GALLERY_JSON = SCRIPTS_DIR / "segmentation_cache.json"

# Server configuration
SERVER_CONFIG = {
    "uploads_base": str(UPLOADS_DIR),
    "outputs_base": str(OUTPUTS_DIR)
}

# Segmentation configuration
SEGMENTATION_CONFIG = {
    "yolo_model": str(YOLO_MODEL_PATH),
    "deeplabv3_model": str(DEEPLABV3_MODEL_PATH),
    "script_path": str(SEGMENTATION_SCRIPT)
}

# Create directories if they don't exist
for directory in [UPLOADS_DIR, OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, FRAMES_DIR, SEGMENTATION_DIR, RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
