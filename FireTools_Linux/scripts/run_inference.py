import sys
import os
from ultralytics import YOLO
import torch
import glob
import shutil
import json
import numpy as np
from pathlib import Path
from ultralytics.utils.plotting import colors
import cv2
import subprocess
import time

print("[INFO] Script started.")

# Read input file path from arguments
input_path = sys.argv[1]
print(f"[INFO] Input path: {input_path}")

# Get project root (parent of scripts directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths (relative to project root)
model_path = PROJECT_ROOT / "models" / "best.pt"
output_base = PROJECT_ROOT / "outputs"
print(f"[INFO] Model path: {model_path}")
print(f"[INFO] Output base: {output_base}")

# Make sure output path exists
output_base.mkdir(parents=True, exist_ok=True)

def get_video_properties(video_path):
    """Extract video properties using OpenCV and ffprobe"""
    try:
        # Use OpenCV to get basic properties
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[WARNING] Could not open video with OpenCV: {video_path}")
            return None, None, None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Use ffprobe for more accurate timing information
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'stream=duration,r_frame_rate,nb_frames', 
                '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line and '/' in line:  # Frame rate line
                        parts = line.split(',')
                        if len(parts) >= 1:
                            # Parse frame rate (e.g., "30/1" -> 30.0)
                            frame_rate_str = parts[0]
                            if '/' in frame_rate_str:
                                num, den = frame_rate_str.split('/')
                                accurate_fps = float(num) / float(den)
                                fps = accurate_fps
                                break
                    elif line and line.replace('.', '').isdigit():  # Duration line
                        accurate_duration = float(line)
                        duration = accurate_duration
                        break
        except Exception as e:
            print(f"[WARNING] ffprobe failed: {e}, using OpenCV values")
        
        print(f"[INFO] Video properties: FPS={fps:.2f}, Duration={duration:.2f}s, Frames={frame_count}")
        return fps, duration, frame_count
        
    except Exception as e:
        print(f"[ERROR] Could not extract video properties: {e}")
        return 30.0, 0, 0  # Fallback values

# Load model with GPU acceleration
model = YOLO(str(model_path))
# Force model to use GPU 0 (which is free) if available
if torch.cuda.is_available():
    model.to('cuda:0')  # Specifically use GPU 0
    print(f"[INFO] Model loaded on GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Model loaded on CPU (GPU not available)")
print("[INFO] Model loaded.")

# Override YOLO's default color palette
# Class 0 (fire) = Red (0, 0, 255)
# Class 1 (smoke) = Blue (255, 0, 0)
colors.palette = [
    (255, 0, 0),    # Class 0: Fire = Red
    (0, 0, 255),    # Class 1: Smoke = Blue
]
# Set environment variable to force MP4 output
os.environ['OPENCV_FFMPEG_WRITER_OPTIONS'] = 'video_codec;h264_nvenc'

# Run inference (save to output_base/result)
print("[INFO] Running YOLO inference...")
results = model.predict(
    source=input_path,
    save=True,
    save_txt=False,
    save_conf=True,
    iou=0.3,
    project=str(output_base),
    name="result",   # folder name
    exist_ok=True,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',  # Specifically use GPU 0 (free)
    half=True if torch.cuda.is_available() else False,     # Use FP16 for faster GPU inference
    verbose=False,  # Reduce output verbosity
    save_crop=False,  # Don't save cropped detections (faster)
    stream=True,    # Use streaming for faster processing
    imgsz=640       # Use smaller image size for faster inference
)
print("[INFO] YOLO inference complete.")

# Determine file type (image or video)
input_ext = os.path.splitext(input_path)[1].lower()
print(f"[INFO] Input extension: {input_ext}")
file_type = "images" if input_ext.lower() in [".jpg", ".jpeg", ".png"] else "videos"
result_dir = output_base / "result"
result_dir.mkdir(parents=True, exist_ok=True)

# For images, we need to consume the generator to trigger YOLO's save functionality
if file_type == "images":
    print("[INFO] Processing image results...")
    for result in results:
        # Just iterate to trigger YOLO's save mechanism
        pass
    print("[INFO] Image processing complete.")

# Extract original video properties for metadata
original_fps, original_duration, original_frame_count = None, None, None
if file_type == "videos":
    original_fps, original_duration, original_frame_count = get_video_properties(input_path)

# Create detection timestamps metadata for videos
if file_type == "videos":
    print("[INFO] Creating detection timestamps metadata")
    
    # Process YOLO detection results to extract timestamps
    detection_timestamps = []
    frame_count = 0
    fps = original_fps if original_fps else 30  # Use actual FPS from original video
    
    # Time the full YOLO streaming + per-frame metadata pass
    detection_processing_start = time.time()
    
    # Extract actual detections from results
    for i, result in enumerate(results):
        boxes = result.boxes
        
        # Calculate time in seconds for this frame using actual FPS
        time_sec = round(frame_count / fps, 3)
        
        # Check if any fire/smoke was detected in this frame
        if boxes is not None and len(boxes) > 0:
            try:
                # Determine detection types based on class labels
                # Assuming class 0 is fire and class 1 is smoke (adjust based on your model)
                class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                
                if len(confidences) > 0:
                    # Group detections by type
                    fire_detections = [confidences[i] for i, cls_id in enumerate(class_ids) if i < len(class_ids) and cls_id == 0]
                    smoke_detections = [confidences[i] for i, cls_id in enumerate(class_ids) if i < len(class_ids) and cls_id == 1]
                    
                    has_fire = len(fire_detections) > 0
                    has_smoke = len(smoke_detections) > 0
                    
                    fire_confidence = float(np.mean(fire_detections)) if fire_detections else 0.0
                    smoke_confidence = float(np.mean(smoke_detections)) if smoke_detections else 0.0
                    overall_confidence = float(np.mean(confidences))
                    
                    detection_type = "fire"  # Default to fire if no class info available
                    if len(class_ids) > 0:  # Only classify if we have class information
                        if has_fire and has_smoke:
                            detection_type = "both"
                        elif has_fire:
                            detection_type = "fire"
                        elif has_smoke:
                            detection_type = "smoke"
                    
                    print(f"[INFO] Detection found at frame {frame_count} ({time_sec}s) - Type: {detection_type}")
                    detection_timestamps.append({
                        "frame": frame_count,
                        "time": time_sec,
                        "confidence": overall_confidence,
                        "detection_type": detection_type,
                        "fire_confidence": fire_confidence,
                        "smoke_confidence": smoke_confidence,
                        "detections_count": len(boxes)
                    })
            except Exception as e:
                print(f"[WARNING] Error processing detection data for frame {frame_count}: {e}")
                # Fallback: just record a basic detection
                detection_timestamps.append({
                    "frame": frame_count,
                    "time": time_sec,
                    "confidence": 0.5,
                    "detection_type": "fire",
                    "fire_confidence": 0.5,
                    "smoke_confidence": 0.0,
                    "detections_count": len(boxes)
                })
        
        frame_count += 1
    
    detection_processing_time = time.time() - detection_processing_start
    print(f"[INFO] Detection + metadata processing time (per-frame loop): {detection_processing_time:.2f}s")
    print(f"[BENCH] detection_processing_time={detection_processing_time:.4f}")
    
    # Create metadata file (JSON write)
    file_basename = os.path.basename(input_path)
    stem_name = os.path.splitext(file_basename)[0]
    
    metadata = {
        "filename": file_basename,
        "processed_timestamp": Path(input_path).stat().st_mtime,
        "total_frames": frame_count,
        "fps": fps,
        "original_duration": original_duration,
        "detections": detection_timestamps
    }
    
    metadata_file = result_dir / f"{stem_name}_metadata.json"
    metadata_start_time = time.time()
    with open(str(metadata_file), 'w') as f:
        json.dump(metadata, f, indent=2)
    metadata_time = time.time() - metadata_start_time
    print(f"[INFO] Metadata JSON write time: {metadata_time:.2f}s")
    print(f"[BENCH] detection_metadata_time={metadata_time:.4f}")

    print(f"[INFO] Saved detection metadata to {metadata_file}")

# Convert YOLO output to web-compatible format
conversion_time_total = 0.0
if file_type == "videos":
    print(f"[INFO] Checking for YOLO output files in: {result_dir}")
    
    # Check for both .avi and .mp4 files from YOLO
    video_files = list(result_dir.glob("*.avi")) + list(result_dir.glob("*.mp4"))
    video_files = [str(f) for f in video_files]
    
    if video_files:
        print(f"[INFO] Found YOLO video files: {video_files}")
        conversion_block_start = time.time()
        for video_file in video_files:
            file_conversion_start = time.time()
            # Always convert to web-compatible H.264 MP4
            base_name = os.path.splitext(video_file)[0]
            web_mp4_file = base_name + "_web.mp4"
            final_mp4_file = base_name + ".mp4" if video_file.endswith('.avi') else base_name + "_final.mp4"
            
            print(f"[INFO] Converting {video_file} to web-compatible H.264 format")
            
            # Use ffmpeg to convert to web-compatible H.264 (normalize timestamps)
            ffmpeg_cmd = [
                'ffmpeg', '-y', 
                '-hwaccel', 'cuda', '-hwaccel_device', '0',  # Use GPU 0 for acceleration
                '-i', video_file,
                '-c:v', 'h264_nvenc',     # NVIDIA GPU H.264 encoder
                '-preset', 'fast',        # Fast encoding preset
                '-crf', '23',             # Good quality setting
                '-vsync', 'cfr',          # Constant frame rate (prevent drift)
                '-r', str(fps),           # Preserve original frame rate
                '-avoid_negative_ts', 'make_zero',  # Normalize timestamps to start at 0
                '-start_at_zero',         # Force video to start at timestamp 0
                '-t', str(original_duration),  # Trim to original duration
                '-c:a', 'aac',            # AAC audio codec
                '-movflags', '+faststart', # Optimize for web streaming
                '-pix_fmt', 'yuv420p',    # Ensure web compatibility
                web_mp4_file
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and os.path.exists(web_mp4_file):
                    # Replace the original file with the web-compatible version
                    if video_file.endswith('.avi'):
                        # For .avi files, replace with .mp4
                        if os.path.exists(final_mp4_file):
                            os.remove(final_mp4_file)
                        os.rename(web_mp4_file, final_mp4_file)
                        os.remove(video_file)  # Remove the .avi file
                        print(f"[INFO] ✓ Web-compatible conversion successful: {os.path.basename(final_mp4_file)}")
                    else:
                        # For .mp4 files, replace the original
                        os.remove(video_file)
                        os.rename(web_mp4_file, video_file)
                        print(f"[INFO] ✓ Web-compatible conversion successful: {os.path.basename(video_file)}")
                else:
                    print(f"[WARNING] Web conversion failed: {result.stderr}")
                    # Fallback: try simple copy if GPU encoding fails
                    fallback_cmd = [
                        'ffmpeg', '-y', '-i', video_file,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-vsync', 'cfr', '-r', str(fps),  # Constant frame rate (prevent drift)
                        '-avoid_negative_ts', 'make_zero',  # Normalize timestamps to start at 0
                        '-start_at_zero',  # Force video to start at timestamp 0
                        '-c:a', 'aac', '-movflags', '+faststart',
                        '-pix_fmt', 'yuv420p', web_mp4_file
                    ]
                    fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=120)
                    if fallback_result.returncode == 0 and os.path.exists(web_mp4_file):
                        if video_file.endswith('.avi'):
                            if os.path.exists(final_mp4_file):
                                os.remove(final_mp4_file)
                            os.rename(web_mp4_file, final_mp4_file)
                            os.remove(video_file)
                        else:
                            os.remove(video_file)
                            os.rename(web_mp4_file, video_file)
                        print(f"[INFO] ✓ Fallback CPU conversion successful")
                    else:
                        print(f"[ERROR] Both GPU and CPU conversion failed")
            except Exception as e:
                print(f"[WARNING] Web conversion error: {e}")
            finally:
                file_conversion_time = time.time() - file_conversion_start
                conversion_time_total += file_conversion_time
                print(f"[INFO] Conversion time for {os.path.basename(video_file)}: {file_conversion_time:.2f}s")
                print(f"[BENCH] conversion_time_file={os.path.basename(video_file)}:{file_conversion_time:.4f}")
        conversion_block_duration = time.time() - conversion_block_start
        print(f"[INFO] Total conversion time (all files): {conversion_time_total:.2f}s "
              f"(wall-clock: {conversion_block_duration:.2f}s)")
    else:
        print(f"[INFO] No video files found from YOLO")
else:
    print(f"[INFO] Skipping conversion (input type: {file_type})")

print(f"[BENCH] conversion_time_total={conversion_time_total:.4f}")

# Copy the result files to individual folder (like segmentation structure)
print(f"[INFO] Copying output files to individual folder structure: {file_type}")
stem_name = Path(input_path).stem

# Create individual folder for this video/image
item_folder = output_base / file_type / stem_name
item_folder.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Created item folder: {item_folder}")

# Find the output file in the result directory
result_files = [f.name for f in result_dir.iterdir() if f.is_file() and stem_name in f.name]

copy_start_time = time.time()

if result_files:
    for result_file in result_files:
        source_path = result_dir / result_file
        dest_path = item_folder / result_file
        print(f"[INFO] Copying {source_path} to {dest_path}")
        shutil.copy2(str(source_path), str(dest_path))
        print(f"[INFO] Setting file permissions")
        os.chmod(str(dest_path), 0o644)  # Make readable by everyone
else:
    print(f"[WARNING] No output files found with stem name: {stem_name}")

copy_total_time = time.time() - copy_start_time
print(f"[INFO] Output copy time: {copy_total_time:.2f}s")
print(f"[BENCH] detection_copy_time={copy_total_time:.4f}")

print("[INFO] Script finished.")