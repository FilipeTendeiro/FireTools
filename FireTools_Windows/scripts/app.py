from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request, Response
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os
import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
import uuid
import asyncio
from datetime import datetime
from pydantic import BaseModel
import time
import platform

# Fix Windows subprocess support
# Set event loop policy for Windows before any async operations
if platform.system() == 'Windows':
    # Use ProactorEventLoop on Windows for subprocess support
    if sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print(f"[INIT] Windows ProactorEventLoop policy set (Python {sys.version_info.major}.{sys.version_info.minor})")
    else:
        # For Python < 3.8, set the event loop explicitly
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        print(f"[INIT] Windows ProactorEventLoop set explicitly (Python {sys.version_info.major}.{sys.version_info.minor})")

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROJECT_ROOT, MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, FRAMES_DIR, 
    SEGMENTATION_DIR, STATIC_DIR, RESULT_DIR, GALLERY_JSON, 
    SEGMENTATION_GALLERY_JSON, SERVER_CONFIG, SEGMENTATION_CONFIG
)

# Cache-busting middleware for development
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add no-cache headers to static files and HTML pages
        if (request.url.path.startswith("/static/") or 
            request.url.path.endswith(".html") or
            request.url.path.endswith(".js") or
            request.url.path.endswith(".css")):
            
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            response.headers["Last-Modified"] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
            response.headers["ETag"] = f'"{int(time.time())}"'
            
        return response

app = FastAPI(title="FireTools API", 
              description="API for FireTools Detection Platform")

# Add cache-busting middleware first
app.add_middleware(NoCacheMiddleware)

# CORS configuration (important for frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local paths (using config module)
BASE_DIR = Path(__file__).resolve().parent

@app.on_event("startup")
async def startup_event():
    """Verify Windows event loop policy is set correctly"""
    if platform.system() == 'Windows':
        loop = asyncio.get_event_loop()
        policy = asyncio.get_event_loop_policy()
        print(f"[STARTUP] Event loop type: {type(loop).__name__}")
        print(f"[STARTUP] Event loop policy: {type(policy).__name__}")
        
        # If somehow the wrong loop is active, try to fix it
        if not isinstance(loop, asyncio.ProactorEventLoop):
            print("[STARTUP] WARNING: ProactorEventLoop not active, attempting to set...")
            if sys.version_info >= (3, 8):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                print("[STARTUP] ProactorEventLoop policy reset")

# Create subdirectories if they don't exist
(UPLOADS_DIR / "images").mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR / "videos").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)

# Task status tracking
tasks = {}
segmentation_tasks = {}

# Helper functions
def get_media_type(filename: str) -> str:
    """Determine if file is image or video based on extension"""
    ext = filename.split(".")[-1].lower()
    
    # Image extensions
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
    
    # Video extensions (including various formats)
    video_extensions = [
        "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", 
        "m4v", "3gp", "ts", "mts", "m2ts", "vob", "asf", 
        "rm", "rmvb", "f4v", "swf", "mpg", "mpeg", "m2v"
    ]
    
    if ext in image_extensions:
        return "images"
    elif ext in video_extensions:
        return "videos"
    else:
        # Default to videos for unknown extensions that might be video
        return "videos"

def get_subprocess_env():
    """
    Get environment for subprocess with refreshed PATH (Windows-compatible).
    Ensures FFmpeg and other system tools are available in subprocesses.
    """
    env = os.environ.copy()
    
    # Preserve or set PYTHONPATH
    if 'PYTHONPATH' not in env:
        import site
        python_paths = [site.getsitepackages()[0] if site.getsitepackages() else '']
        env['PYTHONPATH'] = os.pathsep.join(python_paths)
    
    # On Windows, refresh PATH from registry to ensure FFmpeg is available
    if platform.system() == 'Windows':
        import winreg
        try:
            # Get machine PATH
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment') as key:
                machine_path = winreg.QueryValueEx(key, 'Path')[0]
            # Get user PATH
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment') as key:
                user_path = winreg.QueryValueEx(key, 'Path')[0]
            # Combine and update environment
            env['PATH'] = machine_path + ';' + user_path + ';' + env.get('PATH', '')
        except Exception as e:
            print(f"[WARNING] Could not refresh PATH from registry: {e}")
    
    return env

def needs_video_conversion(filename: str) -> bool:
    """Check if video file needs conversion to MP4 format"""
    ext = filename.split(".")[-1].lower()
    
    # These formats need conversion to MP4 for optimal processing
    conversion_needed = [
        "ts", "mts", "m2ts", "vob", "asf", "rm", "rmvb", 
        "f4v", "swf", "mpg", "mpeg", "m2v", "avi", "wmv", 
        "flv", "webm", "mov", "mkv", "3gp"
    ]
    
    return ext in conversion_needed

async def convert_video_to_mp4(input_path: str, output_path: str) -> bool:
    """Convert video file to MP4 format with H.264 codec"""
    try:
        # Use ffmpeg to convert to MP4 with H.264 (GPU accelerated)
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'h264_nvenc',     # NVIDIA GPU H.264 encoder
            '-preset', 'fast',        # Fast encoding preset
            '-crf', '23',             # Good quality setting
            '-c:a', 'aac',            # AAC audio codec
            '-movflags', '+faststart', # Optimize for web streaming
            '-y',                     # Overwrite output file if exists
            output_path
        ]
        
        print(f"Converting {input_path} to MP4 format...")
        
        # Get environment with refreshed PATH (for FFmpeg)
        env = get_subprocess_env()
        
        # Run conversion asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"✓ Successfully converted to: {output_path}")
            return True
        else:
            print(f"✗ Conversion failed: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"✗ Error during conversion: {str(e)}")
        return False

async def range_requests_response(
    request: Request, file_path: Path, chunk_size: int = 1024 * 1024
):
    """
    Stream video file with HTTP range request support for seeking.
    This enables proper video seeking in HTML5 video players.
    """
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    headers = {
        "accept-ranges": "bytes",
        "content-type": "video/mp4",
    }

    # If no range header, send entire file
    if not range_header:
        headers["content-length"] = str(file_size)
        
        def iterfile():
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
        
        return StreamingResponse(
            iterfile(),
            headers=headers,
            status_code=200
        )

    # Parse range header (e.g., "bytes=0-1023")
    try:
        range_str = range_header.replace("bytes=", "")
        range_start, range_end = range_str.split("-")
        range_start = int(range_start) if range_start else 0
        range_end = int(range_end) if range_end else file_size - 1
        
        # Ensure range is valid
        range_start = max(0, range_start)
        range_end = min(file_size - 1, range_end)
        content_length = range_end - range_start + 1

        headers["content-length"] = str(content_length)
        headers["content-range"] = f"bytes {range_start}-{range_end}/{file_size}"

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk_to_read = min(chunk_size, remaining)
                    chunk = f.read(chunk_to_read)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iterfile(),
            headers=headers,
            status_code=206  # Partial Content
        )
    except Exception as e:
        # If range parsing fails, return full file
        print(f"Error parsing range header: {e}")
        headers["content-length"] = str(file_size)
        
        def iterfile():
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
        
        return StreamingResponse(
            iterfile(),
            headers=headers,
            status_code=200
        )

def path_to_url(absolute_path: str) -> str:
    """Convert absolute file path to relative URL for serving via static mounts"""
    abs_path = Path(absolute_path).resolve()
    
    # Check if path is under OUTPUTS_DIR
    try:
        rel_path = abs_path.relative_to(OUTPUTS_DIR)
        return f"/outputs/{rel_path.as_posix()}"
    except ValueError:
        pass
    
    # Check if path is under UPLOADS_DIR
    try:
        rel_path = abs_path.relative_to(UPLOADS_DIR)
        return f"/uploads/{rel_path.as_posix()}"
    except ValueError:
        pass
    
    # Check if path is under FRAMES_DIR
    try:
        rel_path = abs_path.relative_to(FRAMES_DIR)
        return f"/frames/{rel_path.as_posix()}"
    except ValueError:
        pass
    
    # Check if path is under SEGMENTATION_DIR
    try:
        rel_path = abs_path.relative_to(SEGMENTATION_DIR)
        return f"/segmentation-files/{rel_path.as_posix()}"
    except ValueError:
        pass
    
    # Fallback: return original path (shouldn't happen)
    print(f"[WARNING] Could not convert path to URL: {absolute_path}")
    return absolute_path

def url_to_path(url_path: str) -> str:
    """Convert relative URL to absolute file path"""
    # Check if it's a Linux absolute path BEFORE stripping the slash
    if url_path.startswith('/home/') or url_path.startswith('/mnt/'):
        # Extract the relative part and reconstruct
        if 'frames/' in url_path:
            filename = url_path.split('frames/')[-1]
            return str(FRAMES_DIR / filename)
        elif 'outputs/' in url_path:
            rel_part = url_path.split('outputs/')[-1]
            return str(OUTPUTS_DIR / rel_part)
        elif 'uploads/' in url_path:
            rel_part = url_path.split('uploads/')[-1]
            return str(UPLOADS_DIR / rel_part)
    
    # Check if it's a Windows absolute path
    if url_path.startswith(('D:', 'C:', 'E:', 'D:\\', 'C:\\', 'E:\\')):
        return url_path
    
    # Remove leading slash if present (for relative URLs)
    url_path = url_path.lstrip('/')
    
    # Map URL prefixes to directory paths
    if url_path.startswith('outputs/'):
        rel_path = url_path[len('outputs/'):]
        return str(OUTPUTS_DIR / rel_path)
    elif url_path.startswith('uploads/'):
        rel_path = url_path[len('uploads/'):]
        return str(UPLOADS_DIR / rel_path)
    elif url_path.startswith('frames/'):
        rel_path = url_path[len('frames/'):]
        return str(FRAMES_DIR / rel_path)
    elif url_path.startswith('segmentation-files/'):
        rel_path = url_path[len('segmentation-files/'):]
        return str(SEGMENTATION_DIR / rel_path)
    
    # Check if it looks like a frame filename (contains "_frame_")
    elif '_frame_' in url_path and url_path.endswith('.jpg'):
        return str(FRAMES_DIR / url_path)
    
    # Check if it's home path after lstrip (shouldn't happen but just in case)
    elif url_path.startswith('home/'):
        if 'frames/' in url_path:
            filename = url_path.split('frames/')[-1]
            return str(FRAMES_DIR / filename)
        elif 'outputs/' in url_path:
            rel_part = url_path.split('outputs/')[-1]
            return str(OUTPUTS_DIR / rel_part)
        elif 'uploads/' in url_path:
            rel_part = url_path.split('uploads/')[-1]
            return str(UPLOADS_DIR / rel_part)
    
    # If it doesn't match any pattern, assume it's a relative path from project root
    return str(PROJECT_ROOT / url_path)

def load_gallery():
    """Load gallery cache from JSON file"""
    if GALLERY_JSON.exists():
        with open(GALLERY_JSON, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_gallery(items):
    """Save gallery cache to JSON file"""
    with open(GALLERY_JSON, "w") as f:
        json.dump(items, f)

def cleanup_converted_file(gallery_item):
    """Clean up function - kept for backward compatibility but simplified"""
    # No longer needed since we don't keep separate converted files
    # Original .ts files are deleted immediately after conversion
    pass

def load_segmentation_gallery():
    """Load segmentation gallery cache from JSON file"""
    if SEGMENTATION_GALLERY_JSON.exists():
        with open(SEGMENTATION_GALLERY_JSON, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_segmentation_gallery(items):
    """Save segmentation gallery cache to JSON file"""
    with open(SEGMENTATION_GALLERY_JSON, "w") as f:
        json.dump(items, f)

async def process_file(task_id: str, file_path: str, original_filename: str):
    """Process a file in the background"""
    try:
        # Update task status
        tasks[task_id]["status"] = "uploading"
        print(f"[DEBUG] Task {task_id}: Status set to uploading")
        
        # Get file type
        file_type = get_media_type(original_filename)
        
        # Track the path used for processing (converted MP4 or original upload)
        processing_file_path = tasks[task_id]["file_path"]
        
        # Handle video conversion if needed
        if tasks[task_id].get("needs_conversion", False):
            tasks[task_id]["status"] = "converting"
            print(f"[DEBUG] Task {task_id}: Status set to converting")
            
            original_file_path = tasks[task_id]["original_file_path"]
            
            print(f"Converting {original_filename} to MP4 for processing...")
            conversion_success = await convert_video_to_mp4(original_file_path, processing_file_path)
            
            if not conversion_success:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = "Video conversion to MP4 failed"
                print(f"[DEBUG] Task {task_id}: Status set to failed - Conversion failed")
                return
            
            # Delete the original file after successful conversion
            try:
                if os.path.exists(original_file_path) and original_file_path != processing_file_path:
                    os.remove(original_file_path)
                    print(f"✓ Deleted original file: {original_filename}")
            except Exception as e:
                print(f"⚠ Warning: Failed to delete original file {original_filename}: {e}")
            
            print(f"✓ Conversion completed for {original_filename}")
        
        # Update task status
        tasks[task_id]["status"] = "processing"
        print(f"[DEBUG] Task {task_id}: Status set to processing")
        
        # Run inference directly using the upload path
        # Get environment with refreshed PATH (for FFmpeg and other tools)
        env = get_subprocess_env()
        
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).parent / "run_inference.py"), file_path,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Debug output for all cases
        print(f"[DEBUG] Task {task_id}: Command executed: {sys.executable} {Path(__file__).parent / 'run_inference.py'} {file_path}")
        print(f"[DEBUG] Task {task_id}: Working directory: {PROJECT_ROOT}")
        print(f"[DEBUG] Task {task_id}: Return code: {process.returncode}")
        print(f"[DEBUG] Task {task_id}: STDOUT length: {len(stdout.decode())}")
        print(f"[DEBUG] Task {task_id}: STDERR length: {len(stderr.decode())}")
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            stdout_msg = stdout.decode()
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Detection failed: {error_msg}"
            print(f"[DEBUG] Task {task_id}: Status set to failed - Detection failed")
            print(f"[DEBUG] Task {task_id}: STDERR: {error_msg}")
            print(f"[DEBUG] Task {task_id}: STDOUT: {stdout_msg}")
            print(f"[DEBUG] Task {task_id}: Return code: {process.returncode}")
            return
        
        # Find result file from inference output
        tasks[task_id]["status"] = "finishing"
        print(f"[DEBUG] Task {task_id}: Status set to finishing")
        
        # Use the base filename (without extension) to find results
        # For converted files, we use the .mp4 name (without _converted suffix)
        result_filename = os.path.basename(processing_file_path)
        result_stem = os.path.splitext(result_filename)[0]
            
        result_dir = RESULT_DIR
        result_files = [f for f in os.listdir(result_dir) if result_stem in f]
        
        print(f"[DEBUG] Looking for results with stem: {result_stem}")
        print(f"[DEBUG] Found result files: {result_files}")
        
        if not result_files:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "Output file not found"
            print(f"[DEBUG] Task {task_id}: Status set to failed - Output file not found")
            return
        
        # Create individual folder for this item (like segmentation structure)
        item_folder = OUTPUTS_DIR / file_type / result_stem
        item_folder.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Created item folder: {item_folder}")
        
        # Copy all result files to the individual folder
        main_result_path = None
        for result_filename in result_files:
            result_path = f"{result_dir}/{result_filename}"
            output_path = str(item_folder / result_filename)
            
            print(f"[DEBUG] Copying from {result_path} to {output_path}")
            
            try:
                shutil.copy2(result_path, output_path)
            except Exception as copy_error:
                print(f"[ERROR] Copy failed: {copy_error}")
                # Try copy again
                shutil.copy2(result_path, output_path)
            
            print(f"[DEBUG] Output file exists after copy: {os.path.exists(output_path)}")
            print(f"[DEBUG] File permissions: {oct(os.stat(output_path).st_mode)[-3:]}")
            
            # Set read permissions for everyone
            try:
                os.chmod(output_path, 0o644)
                print(f"[DEBUG] Updated permissions: {oct(os.stat(output_path).st_mode)[-3:]}")
            except Exception as perm_error:
                print(f"[ERROR] Failed to set permissions: {perm_error}")
                
            # Determine which file should be the main result (video/image, not metadata)
            if not result_filename.endswith('_metadata.json'):
                main_result_path = output_path
                print(f"[DEBUG] Set main result path: {main_result_path}")
        
        # If we didn't find a non-metadata file, use the first file as fallback
        if main_result_path is None:
            main_result_path = str(item_folder / result_files[0])
            print(f"[DEBUG] Using fallback result path: {main_result_path}")
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        print(f"[DEBUG] Task {task_id}: Status set to completed")
        tasks[task_id]["result_path"] = main_result_path
        
        # Add to gallery
        # Use the processing file path (which is the MP4 after conversion)
        display_filename = tasks[task_id].get("original_filename", original_filename)
        original_display_path = processing_file_path  # This is the MP4 file in uploads
        
        gallery_entry = {
            "local_result_path": path_to_url(main_result_path),
            "local_original_path": path_to_url(original_display_path),
            "output_folder": path_to_url(str(item_folder)),
            "media_type": file_type,
            "label": display_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        gallery = load_gallery()
        gallery.append(gallery_entry)
        save_gallery(gallery)
        print(f"[DEBUG] Task {task_id}: Added to gallery successfully")
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"[DEBUG] Task {task_id}: Status set to failed due to exception")

async def process_segmentation(task_id: str, input_path: str, original_filename: str):
    """Process video or image segmentation in the background"""
    try:
        # Convert URL path to absolute file path if needed
        if input_path.startswith('/'):
            input_path = url_to_path(input_path)
            print(f"[DEBUG] Converted URL to path: {input_path}")
        
        # Get input type from task data
        input_type = segmentation_tasks[task_id]["input_type"]
        
        # Update task status
        segmentation_tasks[task_id]["status"] = "processing"
        print(f"[DEBUG] Segmentation task {task_id}: Status set to processing ({input_type})")
        
        # Create output directory for this segmentation
        stem_name = os.path.splitext(original_filename)[0]
        output_dir = SEGMENTATION_DIR / stem_name
        output_dir.mkdir(exist_ok=True)
        
        if input_type == "video":
            # Run video segmentation script
            segmentation_cmd = (
                f"{sys.executable} {SEGMENTATION_CONFIG['script_path']} "
                f"--video {input_path} "
                f"--yolo_model {SEGMENTATION_CONFIG['yolo_model']} "
                f"--deeplabv3_model {SEGMENTATION_CONFIG['deeplabv3_model']} "
                f"--output_dir {output_dir} "
                f"--confidence 0.3"
            )
        else:  # input_type == "image"
            # For images, we'll use the same script but with image input
            # The script will need to handle single images (we'll modify it if needed)
            segmentation_cmd = (
                f"{sys.executable} {SEGMENTATION_CONFIG['script_path']} "
                f"--image {input_path} "
                f"--yolo_model {SEGMENTATION_CONFIG['yolo_model']} "
                f"--deeplabv3_model {SEGMENTATION_CONFIG['deeplabv3_model']} "
                f"--output_dir {output_dir} "
                f"--confidence 0.3"
            )
        
        print(f"[DEBUG] Running segmentation command: {segmentation_cmd}")
        
        # Get environment with refreshed PATH (for FFmpeg and other tools)
        env = get_subprocess_env()
        
        # Parse the command to use subprocess_exec instead of shell
        if input_type == "video":
            cmd_args = [
                sys.executable, str(SEGMENTATION_CONFIG['script_path']),
                "--video", input_path,
                "--yolo_model", str(SEGMENTATION_CONFIG['yolo_model']),
                "--deeplabv3_model", str(SEGMENTATION_CONFIG['deeplabv3_model']),
                "--output_dir", str(output_dir),
                "--confidence", "0.3"
            ]
        else:  # image
            cmd_args = [
                sys.executable, str(SEGMENTATION_CONFIG['script_path']),
                "--image", input_path,
                "--yolo_model", str(SEGMENTATION_CONFIG['yolo_model']),
                "--deeplabv3_model", str(SEGMENTATION_CONFIG['deeplabv3_model']),
                "--output_dir", str(output_dir),
                "--confidence", "0.3"
            ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            segmentation_tasks[task_id]["status"] = "failed"
            segmentation_tasks[task_id]["error"] = f"Segmentation failed: {error_msg}"
            segmentation_tasks[task_id]["completed_at"] = time.time()  # Track completion time for cleanup
            print(f"[DEBUG] Segmentation task {task_id}: Status set to failed - {error_msg}")
            return
        
        # Find the generated output files based on input type
        if input_type == "video":
            output_files = list(output_dir.glob("*.mp4"))
        else:  # image
            output_files = list(output_dir.glob("*.jpg"))
        
        if not output_files:
            segmentation_tasks[task_id]["status"] = "failed"
            segmentation_tasks[task_id]["error"] = "No output files generated"
            segmentation_tasks[task_id]["completed_at"] = time.time()  # Track completion time for cleanup
            print(f"[DEBUG] Segmentation task {task_id}: No output files found")
            return
        
        # Find the specific output files
        mask_file = None
        overlay_file = None
        detection_file = None
        
        for file in output_files:
            if "_masks" in file.name:
                mask_file = file
            elif "_overlay" in file.name:
                overlay_file = file
            elif "_detections" in file.name:
                detection_file = file
        
        # Update task status
        segmentation_tasks[task_id]["status"] = "completed"
        segmentation_tasks[task_id]["completed_at"] = time.time()  # Track completion time for cleanup
        print(f"[DEBUG] Segmentation task {task_id}: Status set to completed")

        # Copy the detection metadata JSON into the segmentation output directory so
        # the timeline remains available even after the detection gallery item is deleted.
        import shutil as _shutil
        metadata_filename = f"{stem_name}_metadata.json"
        metadata_candidates = [
            OUTPUTS_DIR / "videos" / metadata_filename,
            RESULT_DIR / metadata_filename,
            OUTPUTS_DIR / "images" / metadata_filename,
        ]
        for metadata_src in metadata_candidates:
            if metadata_src.exists():
                metadata_dst = output_dir / metadata_filename
                try:
                    _shutil.copy2(str(metadata_src), str(metadata_dst))
                    print(f"[INFO] Copied detection metadata to segmentation folder: {metadata_dst}")
                except Exception as copy_err:
                    print(f"[WARNING] Could not copy metadata to segmentation folder: {copy_err}")
                break
        else:
            print(f"[WARNING] No detection metadata found to copy for {stem_name}")

        # Store result paths
        segmentation_tasks[task_id]["result_paths"] = {
            "mask_file": str(mask_file) if mask_file else None,
            "overlay_file": str(overlay_file) if overlay_file else None,
            "detection_file": str(detection_file) if detection_file else None,
            "output_dir": str(output_dir)
        }
        
        # Add to segmentation gallery
        gallery_entry = {
            "local_original_path": path_to_url(input_path),
            "local_mask_path": path_to_url(str(mask_file)) if mask_file else None,
            "local_overlay_path": path_to_url(str(overlay_file)) if overlay_file else None,
            "local_detection_path": path_to_url(str(detection_file)) if detection_file else None,
            "output_dir": path_to_url(str(output_dir)),
            "label": original_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        segmentation_gallery = load_segmentation_gallery()
        segmentation_gallery.append(gallery_entry)
        save_segmentation_gallery(segmentation_gallery)
        print(f"[DEBUG] Segmentation task {task_id}: Added to segmentation gallery successfully")
        
    except Exception as e:
        print(f"[ERROR] Segmentation processing failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        segmentation_tasks[task_id]["status"] = "failed"
        segmentation_tasks[task_id]["error"] = str(e)
        segmentation_tasks[task_id]["completed_at"] = time.time()  # Track completion time for cleanup
        print(f"[DEBUG] Segmentation task {task_id}: Status set to failed due to exception")

# Mount static subdirectories FIRST (more specific paths)
app.mount("/static/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="static_js")
app.mount("/static/css", StaticFiles(directory=str(STATIC_DIR / "css")), name="static_css")

# Also serve uploads and outputs directories for frontend access
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="static_uploads")
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="static_outputs")

# Mount general static files LAST (less specific path)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Direct access to outputs directory (for compatibility with existing URLs)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Direct access to uploads directory
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Mount frames directory for captured frame access
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

# Mount segmentation directory for segmentation results access
app.mount("/segmentation-files", StaticFiles(directory=str(SEGMENTATION_DIR)), name="segmentation_files")

# Mount the result directory to make metadata files accessible
if RESULT_DIR.exists():
    app.mount("/outputs/result", StaticFiles(directory=str(RESULT_DIR)), name="outputs_result")

# Define models for request bodies
class GalleryItemDelete(BaseModel):
    path: str

# Video Streaming Endpoints (with range request support for seeking)
@app.get("/stream/outputs/videos/{filename:path}")
async def stream_output_video(request: Request, filename: str):
    """Stream output video files with range request support"""
    file_path = OUTPUTS_DIR / "videos" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return await range_requests_response(request, file_path)

@app.get("/stream/uploads/videos/{filename:path}")
async def stream_upload_video(request: Request, filename: str):
    """Stream uploaded video files with range request support"""
    file_path = UPLOADS_DIR / "videos" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return await range_requests_response(request, file_path)

@app.get("/stream/segmentation-files/{filename:path}")
async def stream_segmentation_video(request: Request, filename: str):
    """Stream segmentation video files with range request support"""
    file_path = SEGMENTATION_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return await range_requests_response(request, file_path)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint that redirects to index.html"""
    return RedirectResponse(url="/index.html")

@app.get("/ui")
async def ui():
    """Serve the frontend UI"""
    return FileResponse(str(STATIC_DIR / "index.html"))

# Add a debug endpoint to check paths
@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check paths"""
    return {
        "uploads_base": str(UPLOADS_DIR),
        "outputs_base": str(OUTPUTS_DIR),
        "images_dir": str(OUTPUTS_DIR / "images"),
        "videos_dir": str(OUTPUTS_DIR / "videos"),
        "images_dir_exists": os.path.exists(str(OUTPUTS_DIR / "images")),
        "videos_dir_exists": os.path.exists(str(OUTPUTS_DIR / "videos")),
        "images_files": os.listdir(str(OUTPUTS_DIR / "images")) if os.path.exists(str(OUTPUTS_DIR / "images")) else [],
        "videos_files": os.listdir(str(OUTPUTS_DIR / "videos")) if os.path.exists(str(OUTPUTS_DIR / "videos")) else [],
    }

@app.get("/debug/uploads")
async def debug_uploads():
    """Debug endpoint to check uploads directory"""
    uploads_images_dir = str(UPLOADS_DIR / "images")
    uploads_videos_dir = str(UPLOADS_DIR / "videos")
    
    # List files in uploads directories
    images_files = os.listdir(uploads_images_dir) if os.path.exists(uploads_images_dir) else []
    videos_files = os.listdir(uploads_videos_dir) if os.path.exists(uploads_videos_dir) else []
    
    # Check file permissions
    images_access = {}
    for img in images_files:
        img_path = os.path.join(uploads_images_dir, img)
        try:
            stat = os.stat(img_path)
            images_access[img] = {
                "exists": True,
                "size": stat.st_size,
                "permissions": oct(stat.st_mode)[-3:],
                "is_readable": os.access(img_path, os.R_OK),
                "absolute_path": os.path.abspath(img_path)
            }
        except Exception as e:
            images_access[img] = {"error": str(e)}
    
    # Check mount points
    mount_points = {
        "/uploads": str(UPLOADS_DIR),
        "/uploads/images": str(UPLOADS_DIR / "images"),
        "/uploads/videos": str(UPLOADS_DIR / "videos"),
        "/static/uploads": str(UPLOADS_DIR),
        "/static/uploads/images": str(UPLOADS_DIR / "images"),
        "/static/uploads/videos": str(UPLOADS_DIR / "videos"),
    }
    
    return {
        "uploads_base": str(UPLOADS_DIR),
        "uploads_images_dir": uploads_images_dir,
        "uploads_videos_dir": uploads_videos_dir,
        "images_files": images_files,
        "videos_files": videos_files,
        "images_access": images_access,
        "mount_points": mount_points,
        "uploads_dir_exists": os.path.exists(str(UPLOADS_DIR)),
        "uploads_images_dir_exists": os.path.exists(uploads_images_dir),
        "uploads_videos_dir_exists": os.path.exists(uploads_videos_dir),
    }

@app.post("/upload/")
async def upload_files(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...)
):
    """
    Upload files for fire detection processing
    
    This endpoint accepts multiple image or video files and starts the processing
    pipeline for fire detection.
    """
    results = []
    
    for file in files:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Determine file type and path
        file_type = get_media_type(file.filename)
        file_dir = UPLOADS_DIR / file_type
        original_file_path = str(file_dir / file.filename)
        
        # Save original file locally (always preserve original)
        with open(original_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Normalize video timestamps if it's a video file
        if file_type == "videos":
            print(f"Normalizing timestamps for uploaded video: {file.filename}")
            try:
                # Create temp file for normalized version
                temp_normalized = original_file_path + ".normalized.mp4"
                
                # Normalize timestamps to start at 0 and convert to CFR
                normalize_cmd = [
                    'ffmpeg', '-y', '-i', original_file_path,
                    '-c:v', 'libx264',  # Re-encode to ensure CFR (constant frame rate)
                    '-preset', 'fast',  # Fast encoding
                    '-crf', '23',      # Good quality
                    '-vsync', 'cfr',   # Force constant frame rate
                    '-c:a', 'copy',    # Copy audio stream
                    '-avoid_negative_ts', 'make_zero',  # Normalize timestamps
                    '-start_at_zero',  # Force start at 0
                    temp_normalized
                ]
                
                result = subprocess.run(normalize_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(temp_normalized):
                    # Replace original with normalized version
                    os.remove(original_file_path)
                    os.rename(temp_normalized, original_file_path)
                    print(f"✓ Video timestamps normalized: {file.filename}")
                else:
                    print(f"⚠ Timestamp normalization failed, using original file")
                    if os.path.exists(temp_normalized):
                        os.remove(temp_normalized)
            except Exception as e:
                print(f"⚠ Error normalizing timestamps: {e}, using original file")
        
        # For videos that need conversion, create MP4 version for processing
        processing_file_path = original_file_path
        converted_filename = file.filename
        
        if file_type == "videos" and needs_video_conversion(file.filename):
            # Generate MP4 filename - use same base name, just change extension
            base_name = os.path.splitext(file.filename)[0]
            converted_filename = f"{base_name}.mp4"
            processing_file_path = str(file_dir / converted_filename)
            
            print(f"Video {file.filename} needs conversion to MP4 for processing")
        
        # Create task entry
        tasks[task_id] = {
            "id": task_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "converted_filename": converted_filename,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "file_path": processing_file_path,
            "original_file_path": original_file_path,
            "file_type": file_type,
            "needs_conversion": file_type == "videos" and needs_video_conversion(file.filename)
        }
        
        # Add task to background tasks
        background_tasks.add_task(
            process_file, 
            task_id, 
            processing_file_path, 
            file.filename
        )
        
        # Add to results
        results.append({
            "task_id": task_id,
            "filename": file.filename,
            "status": "queued"
        })
    
    return JSONResponse(
        status_code=202,
        content={
            "message": f"{len(results)} file(s) queued for processing",
            "tasks": results
        }
    )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Add debug log
    print(f"[DEBUG] Returning task status for {task_id}: {tasks[task_id]['status']}")
    
    # Create a response with no-cache headers
    response = JSONResponse(content=tasks[task_id])
    
    # Add cache control headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@app.get("/gallery/")
async def get_gallery(media_type: Optional[str] = None, limit: Optional[int] = None):
    """
    Get the gallery of processed files
    
    Optionally filter by media_type ('images' or 'videos')
    Optionally limit the number of results returned (sorted by most recent first)
    """
    gallery = load_gallery()
    
    # Convert old absolute paths to relative URLs if needed
    for item in gallery:
        if 'local_result_path' in item and item['local_result_path'].startswith(('D:', 'C:', '/')):
            item['local_result_path'] = path_to_url(item['local_result_path'])
        if 'local_original_path' in item and item['local_original_path'].startswith(('D:', 'C:', '/')):
            item['local_original_path'] = path_to_url(item['local_original_path'])
    
    # Sort by timestamp (most recent first)
    gallery.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Filter by media type if specified
    if media_type:
        gallery = [item for item in gallery if item["media_type"] == media_type]
    
    # Limit results if specified
    if limit and limit > 0:
        gallery = gallery[:limit]
    
    return gallery

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """Download the processed result file"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not yet completed")
    
    if "result_path" not in task:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return FileResponse(
        path=task["result_path"],
        filename=f"annotated_{task['filename']}"
    )

@app.delete("/gallery/item")
async def delete_gallery_item(item: GalleryItemDelete):
    """
    Delete a specific gallery item by path and all related files
    """
    gallery = load_gallery()
    
    # Find the item with the matching path
    item_index = -1
    for i, gallery_item in enumerate(gallery):
        if gallery_item["local_result_path"] == item.path:
            item_index = i
            break
    
    if item_index == -1:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Remove the item from the gallery
    removed_item = gallery.pop(item_index)
    save_gallery(gallery)
    
    # Clean up any converted MP4 files
    cleanup_converted_file(removed_item)
    
    # Extract filename information for finding related files
    result_filename = os.path.basename(removed_item["local_result_path"])
    original_filename = os.path.basename(removed_item["local_original_path"])
    
    # Get the base filename (without extension) to find metadata files
    base_filename = os.path.splitext(result_filename)[0]
    if base_filename.endswith('_metadata'):
        base_filename = base_filename.replace('_metadata', '')
    
    # Define all possible file paths to delete
    files_to_delete = []
    
    # 1. Main result file in outputs/videos (or outputs/images)
    files_to_delete.append(removed_item["local_result_path"])
    print(f"[INFO] Will delete main result: {removed_item['local_result_path']}")
    
    # 2. Metadata file in outputs/videos (or outputs/images)
    metadata_filename = f"{base_filename}_metadata.json"
    metadata_path = os.path.join(os.path.dirname(removed_item["local_result_path"]), metadata_filename)
    files_to_delete.append(metadata_path)
    print(f"[INFO] Will delete metadata in outputs: {metadata_path}")
    
    # 3. Delete the entire item folder in outputs/videos/ or outputs/images/
    item_folder = OUTPUTS_DIR / removed_item["media_type"] / base_filename
    if os.path.exists(item_folder):
        print(f"[INFO] Will delete entire item folder: {item_folder}")
        # Add all files in the folder to deletion list
        try:
            for file_path in os.listdir(item_folder):
                full_path = os.path.join(item_folder, file_path)
                if full_path not in files_to_delete:
                    files_to_delete.append(full_path)
                    print(f"[INFO] Found file in item folder: {file_path}")
            # Also add the folder itself
            files_to_delete.append(str(item_folder))
        except Exception as e:
            print(f"[ERROR] Error listing item folder {item_folder}: {e}")
    
    # 4. ALL files in outputs/result directory that match the base filename
    result_dir = RESULT_DIR
    if os.path.exists(result_dir):
        for file_path in os.listdir(result_dir):
            if file_path.startswith(base_filename):
                full_path = os.path.join(result_dir, file_path)
                files_to_delete.append(full_path)
                print(f"[INFO] Found related file in result dir: {file_path}")
    
    # 5. ALL files in uploads/videos/ or uploads/images/ that match the base filename
    upload_dir = UPLOADS_DIR / removed_item["media_type"]
    if os.path.exists(upload_dir):
        for file_path in os.listdir(upload_dir):
            if file_path.startswith(base_filename) or file_path.startswith(os.path.splitext(original_filename)[0]):
                full_path = os.path.join(upload_dir, file_path)
                files_to_delete.append(full_path)
                print(f"[INFO] Found related file in uploads/{removed_item['media_type']}: {file_path}")
    
    # 6. Original file (explicit path from gallery)
    files_to_delete.append(removed_item["local_original_path"])
    print(f"[INFO] Will delete original file: {removed_item['local_original_path']}")
    
    # Log all files to be deleted
    print(f"[INFO] Files to delete for {base_filename}:")
    for file_path in files_to_delete:
        print(f"  - {file_path}")
    
    # Delete all files (but don't fail if they don't exist)
    deleted_files = []
    failed_deletes = []
    
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    deleted_files.append(file_path)
                    print(f"[INFO] ✓ Deleted folder: {file_path}")
                else:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"[INFO] ✓ Deleted file: {file_path}")
            else:
                print(f"[INFO] ✗ File/folder not found (skipping): {file_path}")
        except Exception as e:
            failed_deletes.append(f"{file_path}: {str(e)}")
            print(f"[WARNING] ✗ Failed to delete {file_path}: {e}")
    
    # Prepare response
    response_message = f"Item deleted successfully. Deleted {len(deleted_files)} files."
    
    if failed_deletes:
        response_message += f" Failed to delete {len(failed_deletes)} files."
        print(f"[WARNING] Failed deletes: {failed_deletes}")
    
    return {
        "message": response_message,
        "deleted_files": deleted_files,
        "failed_deletes": failed_deletes
    }

@app.delete("/gallery/{item_type}")
async def clear_gallery(item_type: str):
    """
    Clear gallery items of specified type and delete all related files from disk
    
    item_type must be one of: 'images', 'videos', 'all'
    """
    if item_type not in ["images", "videos", "all"]:
        raise HTTPException(status_code=400, detail="Invalid item type")
    
    print(f"[INFO] Clearing {item_type} gallery - deleting ALL files from directories")
    
    # Track deletion results
    total_deleted_files = []
    total_failed_deletes = []
    
    # Determine which directories to clear based on item_type
    directories_to_clear = []
    
    if item_type == "videos" or item_type == "all":
        directories_to_clear.extend([
            OUTPUTS_DIR / "videos",
            RESULT_DIR,  # Also clear result directory
            UPLOADS_DIR / "videos"
        ])
    
    if item_type == "images" or item_type == "all":
        directories_to_clear.extend([
            OUTPUTS_DIR / "images",
            UPLOADS_DIR / "images"
        ])
    
    # Delete ALL files and folders from the specified directories
    for directory in directories_to_clear:
        if not directory.exists():
            print(f"[INFO] Directory doesn't exist: {directory}")
            continue
        
        print(f"[INFO] Clearing directory: {directory}")
        
        for item_path in directory.glob("*"):
            try:
                if item_path.is_dir():
                    # Delete folder and all its contents (for individual item folders)
                    shutil.rmtree(item_path)
                    total_deleted_files.append(str(item_path))
                    print(f"[INFO] Deleted folder: {item_path}")
                elif item_path.is_file():
                    # Delete individual file
                    item_path.unlink()
                    total_deleted_files.append(str(item_path))
                    print(f"[INFO] Deleted file: {item_path}")
            except Exception as e:
                total_failed_deletes.append(f"{item_path}: {str(e)}")
                print(f"[WARNING] Failed to delete {item_path}: {e}")
    
    # Clear the gallery cache
    gallery = load_gallery()
    
    if item_type == "all":
        save_gallery([])
        items_cleared = len(gallery)
    else:
        items_before = len(gallery)
        filtered_gallery = [item for item in gallery if item["media_type"] != item_type]
        save_gallery(filtered_gallery)
        items_cleared = items_before - len(filtered_gallery)
    
    print(f"[INFO] Clearing {item_type} gallery - deleted {len(total_deleted_files)} files from disk")
    
    # Prepare response message
    response_message = f"Gallery {item_type} cleared successfully. "
    response_message += f"Cleared {items_cleared} items from gallery. "
    response_message += f"Deleted {len(total_deleted_files)} files from disk."
    
    if total_failed_deletes:
        response_message += f" Failed to delete {len(total_failed_deletes)} files."
        print(f"[WARNING] Failed deletes during clear: {total_failed_deletes}")
    
    print(f"[INFO] Clear gallery {item_type} completed: {response_message}")
    
    return {
        "message": response_message,
        "cleared_items": items_cleared,
        "deleted_files": total_deleted_files,
        "failed_deletes": total_failed_deletes
    }

@app.post("/segmentation/")
async def start_segmentation(
    background_tasks: BackgroundTasks,
    video_path: Optional[str] = Form(None),
    image_path: Optional[str] = Form(None),
    original_filename: str = Form(...)
):
    """
    Start segmentation processing for a video or image file
    
    This endpoint accepts a video file path or image file path and starts the segmentation
    pipeline for fire and smoke detection.
    """
    # Validate that either video_path or image_path is provided
    if not video_path and not image_path:
        raise HTTPException(status_code=400, detail="Either video_path or image_path must be provided")
    
    if video_path and image_path:
        raise HTTPException(status_code=400, detail="Only one of video_path or image_path should be provided")
    
    # Determine the input path and type
    input_path = video_path if video_path else image_path
    input_type = "video" if video_path else "image"
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    segmentation_tasks[task_id] = {
        "id": task_id,
        "filename": original_filename,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "input_path": input_path,
        "input_type": input_type
    }
    
    # Add task to background tasks
    background_tasks.add_task(
        process_segmentation,
        task_id,
        input_path,
        original_filename
    )
    
    print(f"[DEBUG] Started segmentation task {task_id} for {original_filename}")
    
    return JSONResponse(
        status_code=202,
        content={
            "message": "Segmentation task queued",
            "task_id": task_id,
            "filename": original_filename,
            "status": "queued"
        }
    )

def cleanup_old_segmentation_tasks():
    """Clean up segmentation tasks older than 10 minutes"""
    current_time = time.time()
    tasks_to_remove = []
    
    for task_id, task_data in segmentation_tasks.items():
        if "completed_at" in task_data:
            # Remove tasks completed more than 10 minutes ago
            if current_time - task_data["completed_at"] > 600:  # 10 minutes
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del segmentation_tasks[task_id]
        print(f"[DEBUG] Cleaned up old segmentation task: {task_id}")

@app.get("/segmentation/task/{task_id}")
@app.get("/segmentation/status/{task_id}")  # Alias for frontend compatibility
async def get_segmentation_task_status(task_id: str):
    """Get the status of a segmentation task"""
    # Clean up old tasks before checking
    cleanup_old_segmentation_tasks()
    
    if task_id not in segmentation_tasks:
        raise HTTPException(status_code=404, detail="Segmentation task not found")
    
    print(f"[DEBUG] Returning segmentation task status for {task_id}: {segmentation_tasks[task_id]['status']}")
    
    # Create a response with no-cache headers
    response = JSONResponse(content=segmentation_tasks[task_id])
    
    # Add cache control headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@app.get("/segmentation/results")
async def get_segmentation_results():
    """
    Get segmentation results by scanning the segmentation output directory
    Returns both images and videos found in the segmentation directory
    """
    results = []
    
    if not SEGMENTATION_DIR.exists():
        return results
    
    try:
        # Scan each subdirectory in the segmentation output folder
        for item_dir in SEGMENTATION_DIR.iterdir():
            if item_dir.is_dir():
                item_name = item_dir.name
                
                # Check what type of files are in this directory
                has_video = any(f.suffix.lower() == '.mp4' for f in item_dir.glob('*'))
                has_image = any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in item_dir.glob('*'))
                
                # Determine type based on file contents
                if has_video:
                    item_type = 'video'
                elif has_image:
                    item_type = 'image'
                else:
                    continue  # Skip directories without recognized media files
                
                results.append({
                    'name': item_name,
                    'type': item_type,
                    'path': str(item_dir)
                })
        
        # Sort by directory name
        results.sort(key=lambda x: x['name'])
        
    except Exception as e:
        print(f"Error scanning segmentation directory: {e}")
    
    return results

@app.delete("/segmentation/results/{item_name}")
async def delete_segmentation_result(item_name: str):
    """
    Delete a specific segmentation result by name
    """
    item_dir = SEGMENTATION_DIR / item_name
    
    if not item_dir.exists():
        raise HTTPException(status_code=404, detail=f"Segmentation result '{item_name}' not found")
    
    try:
        # Delete all files in the directory
        deleted_files = []
        for file_path in item_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_files.append(str(file_path))
        
        # Remove the directory
        item_dir.rmdir()
        
        return {
            "message": f"Segmentation result '{item_name}' deleted successfully",
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete segmentation result: {str(e)}")

@app.delete("/segmentation/results")
async def clear_all_segmentation_results():
    """
    Clear all segmentation results
    """
    if not SEGMENTATION_DIR.exists():
        return {"message": "No segmentation results to clear"}
    
    try:
        deleted_count = 0
        for item_dir in SEGMENTATION_DIR.iterdir():
            if item_dir.is_dir():
                # Delete all files in the directory
                for file_path in item_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                
                # Remove the directory
                item_dir.rmdir()
                deleted_count += 1
        
        return {"message": f"Cleared {deleted_count} segmentation results"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear segmentation results: {str(e)}")

@app.delete("/segmentation/results/{item_type}/clear")
async def clear_segmentation_by_type(item_type: str):
    """
    Clear segmentation results by type
    
    item_type must be one of: 'images', 'videos', 'all'
    """
    if item_type not in ["images", "videos", "all"]:
        raise HTTPException(status_code=400, detail="Invalid item type. Must be 'images', 'videos', or 'all'")
    
    if not SEGMENTATION_DIR.exists():
        return {"message": f"No segmentation {item_type} to clear"}
    
    import time
    import gc
    
    deleted_count = 0
    total_deleted_files = []
    total_failed_deletes = []
    
    try:
        for item_dir in SEGMENTATION_DIR.iterdir():
            if item_dir.is_dir():
                # Check what type of files are in this directory
                has_video = any(f.suffix.lower() == '.mp4' for f in item_dir.glob('*'))
                has_image = any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in item_dir.glob('*'))
                
                # Determine if we should delete this directory based on type filter
                should_delete = False
                if item_type == "all":
                    should_delete = True
                elif item_type == "videos" and has_video:
                    should_delete = True
                elif item_type == "images" and has_image and not has_video:
                    should_delete = True
                
                if should_delete:
                    # Delete all files in the directory with retry logic
                    all_files_deleted = True
                    for file_path in item_dir.glob("*"):
                        if file_path.is_file():
                            deleted = False
                            # Try multiple times with increasing delays
                            for attempt in range(3):
                                try:
                                    if attempt > 0:
                                        # Force garbage collection to release file handles
                                        gc.collect()
                                        # Wait before retry
                                        time.sleep(0.5 * attempt)
                                        print(f"[INFO] Retry {attempt + 1}/3 to delete {file_path}")
                                    
                                    file_path.unlink()
                                    total_deleted_files.append(str(file_path))
                                    deleted = True
                                    break
                                except PermissionError as e:
                                    if attempt == 2:  # Last attempt
                                        total_failed_deletes.append(f"{file_path}: File is in use")
                                        print(f"[WARNING] Failed to delete {file_path}: File is in use (may be open in browser)")
                                        all_files_deleted = False
                                except Exception as e:
                                    if attempt == 2:  # Last attempt
                                        total_failed_deletes.append(f"{file_path}: {str(e)}")
                                        print(f"[WARNING] Failed to delete {file_path}: {e}")
                                        all_files_deleted = False
                    
                    # Try to remove the directory only if all files were deleted
                    if all_files_deleted:
                        try:
                            item_dir.rmdir()
                            deleted_count += 1
                            print(f"[INFO] Deleted segmentation result: {item_dir.name} (type: {item_type})")
                        except Exception as e:
                            print(f"[WARNING] Failed to delete directory {item_dir}: {e}")
                    else:
                        print(f"[WARNING] Skipping directory removal for {item_dir.name}: some files could not be deleted")
        
        message = f"Cleared {deleted_count} segmentation {item_type}. Deleted {len(total_deleted_files)} files."
        if total_failed_deletes:
            message += f" Failed to delete {len(total_failed_deletes)} files (may be in use)."
        
        print(f"[INFO] {message}")
        return {
            "message": message,
            "deleted_count": deleted_count,
            "deleted_files": total_deleted_files,
            "failed_deletes": total_failed_deletes
        }
    except Exception as e:
        print(f"[ERROR] Failed to clear segmentation {item_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear segmentation {item_type}: {str(e)}")

@app.get("/segmentation/check/{filename}")
async def check_segmentation_exists(filename: str):
    """
    Check if segmentation already exists for a given filename
    Returns true if segmentation exists, false otherwise
    """
    try:
        # Remove extension to get stem name
        stem_name = os.path.splitext(filename)[0]
        segmentation_dir = SEGMENTATION_DIR / stem_name
        
        print(f"[DEBUG] Checking segmentation for: {filename} (stem: {stem_name})")
        print(f"[DEBUG] Looking in: {segmentation_dir}")
        print(f"[DEBUG] Directory exists: {segmentation_dir.exists()}")
        
        # Check if segmentation directory exists and has files
        if segmentation_dir.exists() and segmentation_dir.is_dir():
            # Check if there are any result files (not just metadata)
            result_files = list(segmentation_dir.glob("*_overlay.jpg")) + \
                          list(segmentation_dir.glob("*_overlay.mp4")) + \
                          list(segmentation_dir.glob("*_masks.jpg")) + \
                          list(segmentation_dir.glob("*_masks.mp4")) + \
                          list(segmentation_dir.glob("*_detections.jpg")) + \
                          list(segmentation_dir.glob("*_detections.mp4"))
            
            print(f"[DEBUG] Found {len(result_files)} result files")
            return {"exists": len(result_files) > 0}
        
        return {"exists": False}
    except Exception as e:
        print(f"[ERROR] Error checking segmentation existence: {e}")
        import traceback
        print(traceback.format_exc())
        return {"exists": False}

@app.get("/segmentation/gallery/")
async def get_segmentation_gallery(limit: Optional[int] = None):
    """
    Get the segmentation gallery of processed files
    
    Optionally limit the number of results returned (sorted by most recent first)
    """
    gallery = load_segmentation_gallery()
    
    # Convert old absolute paths to relative URLs if needed
    for item in gallery:
        if 'local_original_path' in item and item['local_original_path'].startswith(('D:', 'C:', '/')):
            item['local_original_path'] = path_to_url(item['local_original_path'])
        if 'local_mask_path' in item and item['local_mask_path'] and item['local_mask_path'].startswith(('D:', 'C:', '/')):
            item['local_mask_path'] = path_to_url(item['local_mask_path'])
        if 'local_overlay_path' in item and item['local_overlay_path'] and item['local_overlay_path'].startswith(('D:', 'C:', '/')):
            item['local_overlay_path'] = path_to_url(item['local_overlay_path'])
        if 'local_detection_path' in item and item['local_detection_path'] and item['local_detection_path'].startswith(('D:', 'C:', '/')):
            item['local_detection_path'] = path_to_url(item['local_detection_path'])
        if 'output_dir' in item and item['output_dir'].startswith(('D:', 'C:', '/')):
            item['output_dir'] = path_to_url(item['output_dir'])
    
    # Sort by timestamp (most recent first)
    gallery.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Limit results if specified
    if limit and limit > 0:
        gallery = gallery[:limit]
    
    return gallery

@app.delete("/segmentation/gallery/item")
async def delete_segmentation_gallery_item(item: GalleryItemDelete):
    """
    Delete a specific segmentation gallery item by path and all related files
    """
    gallery = load_segmentation_gallery()
    
    # Find the item with the matching path (we'll use the output directory as the identifier)
    item_index = -1
    for i, gallery_item in enumerate(gallery):
        if gallery_item["output_dir"] == item.path:
            item_index = i
            break
    
    if item_index == -1:
        raise HTTPException(status_code=404, detail="Segmentation item not found")
    
    # Remove the item from the gallery
    removed_item = gallery.pop(item_index)
    save_segmentation_gallery(gallery)
    
    # Delete all files in the output directory
    output_dir = Path(removed_item["output_dir"])
    deleted_files = []
    failed_deletes = []
    
    if output_dir.exists():
        for file_path in output_dir.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    print(f"[INFO] Deleted segmentation file: {file_path}")
            except Exception as e:
                failed_deletes.append(f"{file_path}: {str(e)}")
                print(f"[WARNING] Failed to delete {file_path}: {e}")
        
        # Try to remove the directory
        try:
            output_dir.rmdir()
            print(f"[INFO] Deleted segmentation directory: {output_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to delete directory {output_dir}: {e}")
    
    # Prepare response
    response_message = f"Segmentation item deleted successfully. Deleted {len(deleted_files)} files."
    
    if failed_deletes:
        response_message += f" Failed to delete {len(failed_deletes)} files."
        print(f"[WARNING] Failed deletes: {failed_deletes}")
    
    return {
        "message": response_message,
        "deleted_files": deleted_files,
        "failed_deletes": failed_deletes
    }

@app.delete("/segmentation/gallery/clear")
async def clear_segmentation_gallery():
    """
    Clear all segmentation gallery items and delete all related files
    """
    gallery = load_segmentation_gallery()
    
    if not gallery:
        return {"message": "No segmentation items found in gallery"}
    
    print(f"[INFO] Clearing segmentation gallery - found {len(gallery)} items to delete")
    
    import time
    import gc
    
    # Track deletion results
    total_deleted_files = []
    total_failed_deletes = []
    
    # Delete files for each item
    for item in gallery:
        output_dir = Path(item["output_dir"])
        
        if output_dir.exists():
            all_files_deleted = True
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    deleted = False
                    # Try multiple times with increasing delays
                    for attempt in range(3):
                        try:
                            if attempt > 0:
                                # Force garbage collection to release file handles
                                gc.collect()
                                # Wait before retry
                                time.sleep(0.5 * attempt)
                                print(f"[INFO] Retry {attempt + 1}/3 to delete {file_path}")
                            
                            file_path.unlink()
                            total_deleted_files.append(str(file_path))
                            deleted = True
                            print(f"[INFO] Deleted segmentation file: {file_path}")
                            break
                        except PermissionError as e:
                            if attempt == 2:  # Last attempt
                                total_failed_deletes.append(f"{file_path}: File is in use")
                                print(f"[WARNING] Failed to delete {file_path}: File is in use (may be open in browser)")
                                all_files_deleted = False
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                total_failed_deletes.append(f"{file_path}: {str(e)}")
                                print(f"[WARNING] Failed to delete {file_path}: {e}")
                                all_files_deleted = False
            
            # Try to remove the directory only if all files were deleted
            if all_files_deleted:
                try:
                    output_dir.rmdir()
                    print(f"[INFO] Deleted segmentation directory: {output_dir}")
                except Exception as e:
                    print(f"[WARNING] Failed to delete directory {output_dir}: {e}")
            else:
                print(f"[WARNING] Skipping directory removal for {output_dir}: some files could not be deleted")
    
    # Clear the gallery cache
    save_segmentation_gallery([])
    
    # Prepare response message
    response_message = f"Segmentation gallery cleared successfully. "
    response_message += f"Processed {len(gallery)} items. "
    response_message += f"Deleted {len(total_deleted_files)} files."
    
    if total_failed_deletes:
        response_message += f" Failed to delete {len(total_failed_deletes)} files (may be in use)."
        print(f"[WARNING] Failed deletes during clear: {total_failed_deletes}")
    
    print(f"[INFO] Clear segmentation gallery completed: {response_message}")
    
    return {
        "message": response_message,
        "cleared_items": len(gallery),
        "deleted_files": total_deleted_files,
        "failed_deletes": total_failed_deletes
    }

@app.get("/metadata/{filename}")
async def get_metadata(filename: str):
    """Serve a specific metadata file"""
    
    # Ensure filename ends with _metadata.json
    if not filename.endswith('_metadata.json'):
        filename = filename.split('.')[0] + '_metadata.json'
    
    # Check in the outputs/videos directory
    metadata_path = OUTPUTS_DIR / "videos" / filename
    if os.path.exists(metadata_path):
        return FileResponse(metadata_path)
    
    # Check in the outputs/result directory
    result_metadata_path = OUTPUTS_DIR / "result" / filename
    if os.path.exists(result_metadata_path):
        return FileResponse(result_metadata_path)

    # Check in the segmentation output directory (copied there when segmentation ran)
    # The segmentation folder is named after the stem of the original file.
    stem = filename.replace('_metadata.json', '')
    seg_metadata_path = SEGMENTATION_DIR / stem / filename
    if os.path.exists(seg_metadata_path):
        return FileResponse(seg_metadata_path)
    
    # If not found, return 404
    raise HTTPException(status_code=404, detail=f"Metadata file {filename} not found")

@app.get("/test-frames")
async def test_frames_debug():
    """Simple test endpoint to check if new routes are being loaded"""
    return {"message": "Frames endpoints are loaded!", "frames_dir": str(FRAMES_DIR), "exists": FRAMES_DIR.exists()}

@app.get("/test-timeline")
async def test_timeline_debug():
    """Test endpoint to check timeline functionality"""
    # Check if our sample metadata file exists
    sample_metadata_path = RESULT_DIR / "sample_timeline_metadata.json"
    exists = sample_metadata_path.exists()
    
    metadata_content = None
    if exists:
        try:
            with open(sample_metadata_path, 'r') as f:
                metadata_content = json.load(f)
        except Exception as e:
            metadata_content = {"error": str(e)}
    
    return {
        "message": "Timeline test endpoint",
        "sample_metadata_exists": exists,
        "sample_metadata_path": str(sample_metadata_path),
        "metadata_content": metadata_content,
        "timeline_js_loaded": True
    }

@app.post("/capture-frame/")
async def capture_frame(
    frame: UploadFile = File(...),
    timestamp: str = Form(...),
    video_name: str = Form(...)
):
    """
    Capture and save a frame from video for georeferencing purposes
    
    This endpoint receives a frame image captured from the original video
    and saves it with metadata for later georeferencing processing.
    """
    try:
        print(f"[DEBUG] Capture frame called - frame: {frame.filename if frame else 'None'}, timestamp: {timestamp}, video_name: {video_name}")
        
        # Validate inputs
        if not frame:
            raise HTTPException(status_code=400, detail="No frame file provided")
            
        if not frame.filename or not frame.filename.endswith(('.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid frame format. Only JPEG images are supported.")
        
        if not timestamp or not video_name:
            raise HTTPException(status_code=400, detail="Timestamp and video name are required.")
        
        # Create a safe filename
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in "._-")
        frame_filename = f"{safe_video_name}_frame_{timestamp}s.jpg"
        frame_path = FRAMES_DIR / frame_filename
        
        print(f"[DEBUG] Saving frame to: {frame_path}")
        
        # Save the frame
        try:
            with open(frame_path, "wb") as buffer:
                content = await frame.read()
                buffer.write(content)
                print(f"[DEBUG] Frame saved successfully, size: {len(content)} bytes")
        except Exception as file_error:
            print(f"[ERROR] Failed to save frame file: {str(file_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to save frame: {str(file_error)}")
        
        # Set file permissions
        try:
            os.chmod(frame_path, 0o644)
        except Exception as perm_error:
            print(f"[WARNING] Failed to set file permissions: {str(perm_error)}")
        
        # Create metadata for the frame
        frame_metadata = {
            "video_name": video_name,
            "timestamp": float(timestamp),
            "filename": frame_filename,
            "file_path": str(frame_path),
            "capture_time": datetime.now().isoformat(),
            "file_size": os.path.getsize(frame_path),
            "purpose": "georeferencing"
        }
        
        # Save metadata file
        metadata_filename = f"{safe_video_name}_frame_{timestamp}s_metadata.json"
        metadata_path = FRAMES_DIR / metadata_filename
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(frame_metadata, f, indent=2)
                print(f"[DEBUG] Metadata saved to: {metadata_path}")
        except Exception as meta_error:
            print(f"[WARNING] Failed to save metadata: {str(meta_error)}")
        
        print(f"[INFO] Frame captured: {frame_filename} at {timestamp}s from {video_name}")
        
        return {
            "message": "Frame captured successfully",
            "filename": frame_filename,
            "timestamp": float(timestamp),
            "video_name": video_name,
            "file_path": str(frame_path),
            "metadata_path": str(metadata_path)
        }
        
    except ValueError as e:
        print(f"[ERROR] ValueError in frame capture: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {str(e)}")
    except HTTPException as he:
        print(f"[ERROR] HTTPException in frame capture: {he.detail}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in frame capture: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Frame capture failed: {type(e).__name__}: {str(e)}")

@app.get("/api/frames/")
async def list_captured_frames():
    """
    List all captured frames with their metadata
    """
    try:
        print(f"[DEBUG] Looking for frames in: {FRAMES_DIR}")
        frames = []
        
        # Find all frame files in the frames directory
        frame_files = list(FRAMES_DIR.glob("*_frame_*.jpg"))
        print(f"[DEBUG] Found {len(frame_files)} frame files")
        
        for frame_file in frame_files:
            print(f"[DEBUG] Processing frame: {frame_file.name}")
            
            # Construct metadata filename directly
            metadata_filename = frame_file.stem + '_metadata.json'
            metadata_file = FRAMES_DIR / metadata_filename
            
            print(f"[DEBUG] Looking for metadata: {metadata_file}")
            
            frame_info = {
                "filename": frame_file.name,
                "file_path": path_to_url(str(frame_file)),
                "file_size": frame_file.stat().st_size,
                "created": datetime.fromtimestamp(frame_file.stat().st_mtime).isoformat()
            }
            
            # Add metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        # Don't let metadata overwrite the correct file_path
                        if 'file_path' in metadata:
                            del metadata['file_path']
                        frame_info.update(metadata)
                    print(f"[DEBUG] Loaded metadata for {frame_file.name}")
                except Exception as e:
                    print(f"[WARNING] Failed to read metadata for {frame_file.name}: {e}")
            else:
                print(f"[WARNING] Metadata file not found: {metadata_file}")
            
            frames.append(frame_info)
        
        # Sort by capture time (most recent first)
        frames.sort(key=lambda x: x.get("capture_time", ""), reverse=True)
        
        print(f"[DEBUG] Returning {len(frames)} frames")
        return frames
        
    except Exception as e:
        print(f"[ERROR] Failed to list frames: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to list frames: {str(e)}")

@app.delete("/api/frames/clear")
async def clear_all_frames():
    """
    Delete all captured frames and their metadata
    """
    try:
        deleted_files = []
        failed_deletes = []
        
        # Delete all frame files and metadata
        for file_path in FRAMES_DIR.glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    print(f"[INFO] Deleted frame file: {file_path}")
                except Exception as e:
                    failed_deletes.append(f"{file_path}: {str(e)}")
                    print(f"[WARNING] Failed to delete {file_path}: {e}")
        
        response_message = f"Cleared frames directory. Deleted {len(deleted_files)} files."
        if failed_deletes:
            response_message += f" Failed to delete {len(failed_deletes)} files."
            print(f"[WARNING] Failed deletes: {failed_deletes}")
        
        return {
            "message": response_message,
            "deleted_files": deleted_files,
            "failed_deletes": failed_deletes
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to clear frames: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear frames: {str(e)}")

@app.delete("/api/frames/{filename}")
async def delete_frame(filename: str):
    """
    Delete a specific captured frame and its metadata
    """
    try:
        # Validate filename
        if not filename or not filename.endswith('.jpg'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Find and delete the frame file
        frame_path = FRAMES_DIR / filename
        metadata_filename = filename.replace('.jpg', '_metadata.json')
        metadata_path = FRAMES_DIR / metadata_filename
        
        deleted_files = []
        failed_deletes = []
        
        # Delete frame file
        if frame_path.exists():
            try:
                frame_path.unlink()
                deleted_files.append(str(frame_path))
                print(f"[INFO] Deleted frame: {frame_path}")
            except Exception as e:
                failed_deletes.append(f"{frame_path}: {str(e)}")
                print(f"[WARNING] Failed to delete frame {frame_path}: {e}")
        else:
            print(f"[WARNING] Frame file not found: {frame_path}")
        
        # Delete metadata file
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                deleted_files.append(str(metadata_path))
                print(f"[INFO] Deleted metadata: {metadata_path}")
            except Exception as e:
                failed_deletes.append(f"{metadata_path}: {str(e)}")
                print(f"[WARNING] Failed to delete metadata {metadata_path}: {e}")
        
        if not frame_path.exists() and not metadata_path.exists() and len(deleted_files) == 0:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        response_message = f"Frame deleted successfully. Removed {len(deleted_files)} files."
        if failed_deletes:
            response_message += f" Failed to delete {len(failed_deletes)} files."
        
        return {
            "message": response_message,
            "deleted_files": deleted_files,
            "failed_deletes": failed_deletes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to delete frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete frame: {str(e)}")

# Route to serve HTML files directly - this should be last to avoid conflicts
@app.get("/{html_file:path}")
async def serve_html(html_file: str):
    """Serve HTML files from the static directory"""
    # Only handle HTML files, and avoid intercepting API routes
    if html_file.startswith("api/") or html_file.startswith("segmentation/") or html_file.startswith("gallery/") or html_file.startswith("task/") or html_file.startswith("upload/") or html_file.startswith("download/") or html_file.startswith("metadata/") or html_file.startswith("debug/") or html_file.startswith("capture-frame/") or html_file.startswith("test-frames") or html_file.startswith("openapi.json") or html_file.startswith("docs"):
        raise HTTPException(status_code=404, detail="File not found")
    
    if html_file.endswith(".html"):
        file_path = STATIC_DIR / html_file
        if file_path.exists():
            return FileResponse(file_path)
    
    # If the file doesn't exist or isn't an HTML file, 
    # let the request continue to other routes
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    
    # Force Windows to use ProactorEventLoop for subprocess support
    if platform.system() == 'Windows':
        if sys.version_info >= (3, 8):
            # Set the policy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            # Create a new event loop with the policy
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(f"[MAIN] Created ProactorEventLoop: {type(loop).__name__}")
            print("[MAIN] Windows subprocess support enabled")
    
    # Function to open browser after a short delay
    def open_browser():
        import time
        time.sleep(0.5)  # Wait 2 seconds for server to start
        webbrowser.open('http://localhost:8000/')
        print("[MAIN] Opened browser at http://localhost:8000/")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("[MAIN] Starting server at http://localhost:8000/")
    print("[MAIN] Browser will open automatically...")
    
    # Run uvicorn without reload to avoid event loop issues
    # Reload creates child processes that may not inherit the event loop properly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False) 