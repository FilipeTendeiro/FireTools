#!/usr/bin/env python3
"""
Unified Fire & Smoke Segmentation Video Script - Optimized Version (NO RGB FIRE ANALYSIS)
Combines YOLO + DeepLabv3+ Enhanced Segmentation WITHOUT rule-based RGB fire detection

OPTIMIZED CONFIGURATION (from hyperparameter tuning on 459 validation images):
- YOLO Confidence: 0.1 (high recall for smoke detection)
- Smoke Threshold: 0.35 (base threshold for DeepLabv3+ smoke)
- Confidence Adjustment: 0.3 (factor for YOLO-guided smoke enhancement)
- Smoke Fusion: 'union' (best balance between precision and recall)
- NO RGB Fire Analysis: Disabled for better performance (2x faster, more accurate)

KEY IMPROVEMENTS:
- Fire Detection: DeepLabv3+ only (NO RGB/YCbCr rules) → More accurate, sharper results
- Smoke Detection: YOLO-guided DeepLabv3+ enhancement → Best smoke IoU (0.5153)
- Inference Speed: 2x faster than RGB version (0.090s vs 0.189s per frame)
- Segmentation Quality: Single-pass inference (NO TTA) → Sharper, more detailed masks

PERFORMANCE (comparison_results_7):
- Fire IoU: 0.6228 (only 0.7% below DeepLabv3+ baseline)
- Smoke IoU: 0.5153 (8.6% better than DeepLabv3+ baseline)
- Smoke Detection Rate: 85.1% (best among all methods)
- Fire Detection Rate: 88.7%
"""

import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import argparse
from pathlib import Path
import subprocess
import time
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology, measure, filters
import albumentations as A
import subprocess

# Constants
CLASS_COLORS = {
    'background': (0, 0, 0),      # Black
    'smoke': (255, 0, 0),         # Blue
    'fire': (0, 0, 255),          # Red
    'fire_yolo': (0, 128, 255),   # Orange (YOLO fire)
    'fire_combined': (0, 255, 255) # Yellow (Combined fire)
}
NUM_CLASSES = 3

def convert_ts_to_mp4(input_path, output_dir=None):
    """
    Convert .ts file to .mp4 format using ffmpeg
    
    Args:
        input_path: Path to input .ts file
        output_dir: Directory to save converted file (default: same as input)
    
    Returns:
        Path to converted .mp4 file
    """
    input_path = Path(input_path)
    
    # Check if input is .ts file
    if input_path.suffix.lower() != '.ts':
        return str(input_path)  # Return original path if not .ts
    
    # Determine output path
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{input_path.stem}.mp4"
    
    # Skip conversion if mp4 already exists
    if output_path.exists():
        print(f"Converted file already exists: {output_path}")
        return str(output_path)
    
    print(f"Converting {input_path.name} to MP4 format...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # FFmpeg command for conversion (normalize timestamps to start at 0)
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',          # Video codec
        '-c:a', 'aac',              # Audio codec
        '-preset', 'medium',         # Encoding speed/quality balance
        '-crf', '23',               # Constant rate factor (quality)
        '-vsync', 'cfr',            # Constant frame rate for consistency
        '-avoid_negative_ts', 'make_zero',  # Normalize timestamps to start at 0
        '-start_at_zero',           # Force video to start at timestamp 0
        '-y',                       # Overwrite output file
        str(output_path)
    ]
    
    try:
        # Run conversion
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Conversion completed successfully!")
        print(f"Converted file saved as: {output_path}")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        print(f"Falling back to original file: {input_path}")
        return str(input_path)
        
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg to convert .ts files.")
        print("For Ubuntu/Debian: sudo apt install ffmpeg")
        print(f"Falling back to original file: {input_path}")
        return str(input_path)

class UnifiedFireSmokeSegmentation:
    def __init__(self, yolo_model_path, deeplabv3_model_path, confidence_threshold=0.1, 
                 base_smoke_threshold=0.35, confidence_adjustment_factor=0.3):
        """
        Initialize unified fire and smoke segmentation system (NO RGB FIRE ANALYSIS)
        
        Args:
            yolo_model_path: Path to YOLOv8 model (.pt file)
            deeplabv3_model_path: Path to DeepLabv3+ model (.pth file)
            confidence_threshold: Minimum confidence for YOLO detection (OPTIMIZED: 0.1)
            base_smoke_threshold: Base threshold for smoke segmentation (OPTIMIZED: 0.35)
            confidence_adjustment_factor: Factor for adjusting smoke threshold (OPTIMIZED: 0.3)
        """
        self.confidence_threshold = confidence_threshold
        self.base_smoke_threshold = base_smoke_threshold
        self.confidence_adjustment_factor = confidence_adjustment_factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Smoke fusion strategy (OPTIMIZED: union for best balance)
        self.smoke_fusion_strategy = 'union'  # Options: 'union', 'enhanced_only', 'weighted_average'
        
        # Initialize YOLO model with GPU optimization
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        if torch.cuda.is_available():
            self.yolo_model.to('cuda:0')  # Use GPU 0 specifically
            print(f"YOLO model moved to GPU: {torch.cuda.get_device_name(0)}")
        self.fire_class_id = self._detect_fire_class_id()
        
        # Initialize DeepLabv3+ model
        print(f"Loading DeepLabv3+ model: {deeplabv3_model_path}")
        self.deeplabv3_model = self._load_deeplabv3_model(deeplabv3_model_path)
        
        # DeepLabv3+ preprocessing
        self.val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
        ])
        
        print(f"Models loaded successfully on device: {self.device}")
    
    def _detect_fire_class_id(self):
        """Auto-detect the class ID for fire from YOLO model class names"""
        class_names = self.yolo_model.names
        fire_keywords = ['fire', 'flame', 'burn']
        
        for class_id, class_name in class_names.items():
            if any(keyword in class_name.lower() for keyword in fire_keywords):
                print(f"Auto-detected fire class: ID={class_id}, Name='{class_name}'")
                return class_id
        
        print(f"Warning: Could not auto-detect fire class. Using class 0.")
        print(f"Available classes: {class_names}")
        return 0
    
    def _load_deeplabv3_model(self, model_path):
        """Load DeepLabv3+ model"""
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model
    
    def low_light_enhancement(self, image):
        """Enhance image for low light conditions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        hsv[:, :, 2] = cv2.medianBlur(hsv[:, :, 2], 5)
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced_img

    def ycbcr_rules(self, image):
        """Apply YCbCr color space rules for fire detection"""
        img = image.copy()
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cb, cr = cv2.split(ycbcr)
        
        mask = ((y < 145) | (y >= 170)) & (cb >= 50) & (cb <= 120) & (cr > 120) & (cr < 220)
        img[mask] = 255
        img[~mask] = 0
        return img

    def rgb_rules(self, image):
        """Apply RGB color space rules for fire detection
        OPTIMIZED THRESHOLDS: R≥210, G≥90, B≤130 (from hyperparameter tuning)
        """
        img = image.copy()
        b, g, r = cv2.split(image)
        
        rules = [
            r > g, g > b, r > 210, g > 90, b < 130,
            (g / (r + 1)) >= 0.1, (g / (r + 1)) <= 1.0,
            (b / (r + 1)) >= 0.1, (b / (r + 1)) <= 0.85,
            (b / (g + 1)) >= 0.1, (b / (g + 1)) <= 0.85
        ]
        
        mask = np.all(rules, axis=0)
        img[mask] = 255
        img[~mask] = 0
        
        # Morphological operations
        kernel = np.ones((4, 4), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def get_similarity_percentage(self, img1, img2):
        """Calculate pixel-wise similarity percentage between two binary images"""
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        percentage = np.sum(img1 == img2) / (img1.shape[0] * img1.shape[1])
        return percentage

    def rule_based_segmentation(self, image):
        """Apply complete rule-based fire segmentation pipeline"""
        # Low light enhancement
        low_light_img = self.low_light_enhancement(image)
        
        # Apply rules to both images
        rgb_original = self.rgb_rules(image)
        rgb_enhanced = self.rgb_rules(low_light_img)
        ycbcr_original = self.ycbcr_rules(image)
        ycbcr_enhanced = self.ycbcr_rules(low_light_img)
        
        # Combine results
        combined_original = cv2.bitwise_or(rgb_original, ycbcr_original)
        combined_enhanced = cv2.bitwise_or(rgb_enhanced, ycbcr_enhanced)
        ultimate_combined = cv2.bitwise_and(combined_original, combined_enhanced)
        rgb_combined = cv2.bitwise_or(rgb_original, rgb_enhanced)
        
        # Decide final result based on similarity
        similarity = self.get_similarity_percentage(ultimate_combined, rgb_combined)
        final_mask = rgb_combined if similarity >= 0.75 else ultimate_combined
        
        return final_mask, similarity

    def process_yolo_detection(self, frame):
        """Process frame with YOLO detection for both fire and smoke (NO RGB FIRE ANALYSIS)"""
        h, w = frame.shape[:2]
        
        # YOLO Detection with GPU optimization
        results = self.yolo_model(frame, device='cuda:0' if torch.cuda.is_available() else 'cpu', half=True if torch.cuda.is_available() else False, verbose=False)
        
        # Initialize masks
        yolo_fire_mask = np.zeros((h, w), dtype=np.uint8)
        yolo_smoke_regions = []  # Store smoke regions for enhanced DeepLabv3+ analysis
        detections = []
        
        # Process each detection
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.cpu().numpy()[0]
                    cls = int(box.cls.cpu().numpy()[0])
                    
                    if conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        class_name = self.yolo_model.names[cls].lower()
                        
                        # DISABLED: Rule-based RGB fire segmentation (NO RGB mode for better performance)
                        # Fire detection now relies only on DeepLabv3+ (more accurate, 2x faster)
                        # if cls == self.fire_class_id or 'fire' in class_name or 'flame' in class_name:
                        #     cropped_region = frame[y1:y2, x1:x2]
                        #     
                        #     if cropped_region.size > 0:
                        #         fire_mask, similarity = self.rule_based_segmentation(cropped_region)
                        #         
                        #         # Convert to binary mask
                        #         fire_mask_binary = cv2.cvtColor(fire_mask, cv2.COLOR_BGR2GRAY) if len(fire_mask.shape) == 3 else fire_mask
                        #         fire_mask_binary = (fire_mask_binary > 127).astype(np.uint8) * 255
                        #         
                        #         # Place mask back in original coordinates
                        #         yolo_fire_mask[y1:y2, x1:x2] = np.maximum(
                        #             yolo_fire_mask[y1:y2, x1:x2], 
                        #             fire_mask_binary
                        #         )
                        
                        # Store smoke regions for enhanced DeepLabv3+ analysis
                        if 'smoke' in class_name:
                            # Expand smoke bounding box slightly for better context
                            expand_factor = 0.15  # 15% expansion
                            box_w, box_h = x2 - x1, y2 - y1
                            expand_w, expand_h = int(box_w * expand_factor), int(box_h * expand_factor)
                            
                            exp_x1 = max(0, x1 - expand_w)
                            exp_y1 = max(0, y1 - expand_h)
                            exp_x2 = min(w, x2 + expand_w)
                            exp_y2 = min(h, y2 + expand_h)
                            
                            yolo_smoke_regions.append({
                                'bbox': (exp_x1, exp_y1, exp_x2, exp_y2),
                                'original_bbox': (x1, y1, x2, y2),
                                'confidence': conf
                            })
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': cls,
                            'class_name': class_name
                        })
        
        return yolo_fire_mask, yolo_smoke_regions, detections

    def test_time_augmentation(self, image_tensor):
        """Apply test-time augmentation for DeepLabv3+"""
        predictions = []
        
        # Original
        pred = self.deeplabv3_model(image_tensor.unsqueeze(0).to(self.device))
        predictions.append(torch.softmax(pred, dim=1))
        
        # Horizontal flip
        flipped = torch.flip(image_tensor, dims=[2])
        pred_flip = self.deeplabv3_model(flipped.unsqueeze(0).to(self.device))
        pred_flip = torch.flip(torch.softmax(pred_flip, dim=1), dims=[3])
        predictions.append(pred_flip)
        
        # Vertical flip
        v_flipped = torch.flip(image_tensor, dims=[1])
        pred_v_flip = self.deeplabv3_model(v_flipped.unsqueeze(0).to(self.device))
        pred_v_flip = torch.flip(torch.softmax(pred_v_flip, dim=1), dims=[2])
        predictions.append(pred_v_flip)
        
        return torch.mean(torch.stack(predictions), dim=0)

    def process_deeplabv3_segmentation(self, frame):
        """Process frame with DeepLabv3+ segmentation (NO TTA for sharper, more detailed results)"""
        # Preprocess image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_shape = image_rgb.shape[:2]
        
        augmented = self.val_transform(image=image_rgb)
        image_tensor = transforms.ToTensor()(augmented['image'])
        
        with torch.no_grad():
            # Single forward pass (NO test-time augmentation for sharper, more detailed results)
            prediction = self.deeplabv3_model(image_tensor.unsqueeze(0).to(self.device))
            prediction = torch.softmax(prediction, dim=1)
            
            # Convert to numpy and resize
            prediction = prediction.squeeze().cpu().numpy()
            prediction_resized = np.zeros((NUM_CLASSES, orig_shape[0], orig_shape[1]))
            
            for i in range(NUM_CLASSES):
                prediction_resized[i] = cv2.resize(prediction[i], (orig_shape[1], orig_shape[0]))
            
            # Create final mask
            final_mask = np.argmax(prediction_resized, axis=0).astype(np.uint8)
        
        return final_mask, prediction_resized
    
    def enhance_smoke_regions(self, frame, smoke_regions, base_prediction):
        """Enhanced DeepLabv3+ analysis for YOLO-detected smoke regions (NO TTA for sharper results)"""
        h, w = frame.shape[:2]
        enhanced_smoke_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not smoke_regions:
            return enhanced_smoke_mask
        
        for region in smoke_regions:
            x1, y1, x2, y2 = region['bbox']
            confidence = region['confidence']
            
            # Extract region
            region_frame = frame[y1:y2, x1:x2]
            
            if region_frame.size > 0:
                # Preprocess region
                region_rgb = cv2.cvtColor(region_frame, cv2.COLOR_BGR2RGB)
                region_shape = region_rgb.shape[:2]
                
                augmented = self.val_transform(image=region_rgb)
                region_tensor = transforms.ToTensor()(augmented['image'])
                
                with torch.no_grad():
                    # Single forward pass (NO TTA for sharper, more detailed results)
                    pred = self.deeplabv3_model(region_tensor.unsqueeze(0).to(self.device))
                    enhanced_prediction = torch.softmax(pred, dim=1)
                    
                    # Convert to numpy and resize back to region size
                    enhanced_prediction = enhanced_prediction.squeeze().cpu().numpy()
                    region_prediction = np.zeros((NUM_CLASSES, region_shape[0], region_shape[1]))
                    
                    for i in range(NUM_CLASSES):
                        region_prediction[i] = cv2.resize(enhanced_prediction[i], (region_shape[1], region_shape[0]))
                    
                    # Enhanced smoke extraction with confidence weighting
                    smoke_prob = region_prediction[1]  # Class 1 = smoke
                    
                    # Apply confidence-based threshold adjustment
                    # Higher YOLO confidence = lower threshold for DeepLabv3+ smoke detection
                    # OPTIMIZED THRESHOLDS: base=0.35, factor=0.3 (from hyperparameter tuning)
                    confidence_factor = min(confidence * 1.5, 1.0)  # Max factor of 1.0
                    adjusted_threshold = self.base_smoke_threshold * (1.0 - confidence_factor * self.confidence_adjustment_factor)
                    
                    # Create enhanced smoke mask for this region
                    region_smoke_mask = (smoke_prob > adjusted_threshold).astype(np.uint8) * 255
                    
                    # Apply morphological operations to clean up the mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    region_smoke_mask = cv2.morphologyEx(region_smoke_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                    region_smoke_mask = cv2.morphologyEx(region_smoke_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    # Place enhanced mask back in original coordinates
                    enhanced_smoke_mask[y1:y2, x1:x2] = np.maximum(
                        enhanced_smoke_mask[y1:y2, x1:x2],
                        region_smoke_mask
                    )
        
        return enhanced_smoke_mask

    def combine_results(self, yolo_fire_mask, deeplabv3_mask, enhanced_smoke_mask):
        """Intelligently combine results (NO RGB FIRE ANALYSIS - DeepLabv3+ fire only)"""
        h, w = deeplabv3_mask.shape
        
        # Create unified mask
        unified_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract individual class masks from DeepLabv3+
        base_smoke_mask = (deeplabv3_mask == 1).astype(np.uint8)  # Class 1 = smoke
        deeplabv3_fire_mask = (deeplabv3_mask == 2).astype(np.uint8)  # Class 2 = fire
        
        # Convert masks to binary
        yolo_fire_binary = (yolo_fire_mask > 127).astype(np.uint8)  # Will be all zeros in NO RGB mode
        enhanced_smoke_binary = (enhanced_smoke_mask > 127).astype(np.uint8)
        
        # Apply smoke fusion strategy
        if self.smoke_fusion_strategy == 'union':
            # STRATEGY 1: Union (OR) - Best balance (OPTIMIZED)
            combined_smoke_mask = np.logical_or(base_smoke_mask, enhanced_smoke_binary).astype(np.uint8)
        elif self.smoke_fusion_strategy == 'enhanced_only':
            # STRATEGY 2: Enhanced Only - Trust YOLO-guided smoke more
            combined_smoke_mask = enhanced_smoke_binary
        elif self.smoke_fusion_strategy == 'weighted_average':
            # STRATEGY 3: Weighted Average - Blend both predictions
            combined_smoke_mask = ((base_smoke_mask * 0.4 + enhanced_smoke_binary * 0.6) > 0.5).astype(np.uint8)
        else:
            # Default to union
            combined_smoke_mask = np.logical_or(base_smoke_mask, enhanced_smoke_binary).astype(np.uint8)
        
        # Fire detection: Only DeepLabv3+ (NO RGB rule-based fire)
        # yolo_fire_mask is always zeros in NO RGB mode
        combined_fire_mask = deeplabv3_fire_mask.astype(np.uint8)
        
        # Priority: Fire > Smoke > Background
        unified_mask[combined_smoke_mask == 1] = 1  # Enhanced smoke
        unified_mask[combined_fire_mask == 1] = 2   # Fire (overwrites smoke if overlap)
        
        return unified_mask, yolo_fire_binary, combined_smoke_mask, combined_fire_mask, enhanced_smoke_binary

    def create_colored_mask(self, unified_mask, show_sources=False):
        """Create colored visualization of the unified mask"""
        colored_mask = np.zeros((unified_mask.shape[0], unified_mask.shape[1], 3), dtype=np.uint8)
        
        # Apply colors based on class
        colored_mask[unified_mask == 1] = CLASS_COLORS['smoke']    # Blue for smoke
        colored_mask[unified_mask == 2] = CLASS_COLORS['fire']     # Red for fire
        
        return colored_mask

    def create_detection_visualization(self, frame, detections, unified_mask):
        """Create detection visualization with bounding boxes"""
        detection_vis = frame.copy()
        
        # Draw YOLO detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            
            # Set color based on class
            if 'fire' in class_name or 'flame' in class_name:
                color = (0, 0, 255)  # Red for fire
                label = f"Fire: {det['confidence']:.2f}"
            elif 'smoke' in class_name:
                color = (255, 0, 0)  # Blue for smoke
                label = f"Smoke: {det['confidence']:.2f}"
            else:
                color = (0, 255, 0)  # Green for others
                label = f"{class_name.title()}: {det['confidence']:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(detection_vis, label, 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add coverage information
        total_pixels = unified_mask.shape[0] * unified_mask.shape[1]
        fire_pixels = np.sum(unified_mask == 2)
        smoke_pixels = np.sum(unified_mask == 1)
        
        fire_coverage = (fire_pixels / total_pixels) * 100
        smoke_coverage = (smoke_pixels / total_pixels) * 100
        
        # Add coverage text with enhanced information
        cv2.putText(detection_vis, f"Fire: {fire_coverage:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(detection_vis, f"Smoke: {smoke_coverage:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Count YOLO detections for enhanced analysis info
        yolo_fire_count = len([det for det in detections if 'fire' in det['class_name'] or 'flame' in det['class_name']])
        yolo_smoke_count = len([det for det in detections if 'smoke' in det['class_name']])
        
        if yolo_fire_count > 0:
            cv2.putText(detection_vis, f"YOLO Fire Regions: {yolo_fire_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if yolo_smoke_count > 0:
            cv2.putText(detection_vis, f"YOLO Smoke Regions: {yolo_smoke_count}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return detection_vis

    def process_frame(self, frame):
        """Process a single frame with unified and optimized segmentation (no enhanced fire analysis)"""
        # YOLO detection for both fire and smoke
        yolo_fire_mask, yolo_smoke_regions, detections = self.process_yolo_detection(frame)
        
        # DeepLabv3+ segmentation (fire + smoke)
        deeplabv3_mask, deeplabv3_predictions = self.process_deeplabv3_segmentation(frame)
        
        # Enhanced smoke analysis on YOLO-detected smoke regions
        enhanced_smoke_mask = self.enhance_smoke_regions(frame, yolo_smoke_regions, deeplabv3_predictions)
        
        # Combine all results intelligently (no enhanced fire)
        unified_mask, yolo_fire_binary, combined_smoke_mask, combined_fire_mask, enhanced_smoke_binary = self.combine_results(
            yolo_fire_mask, deeplabv3_mask, enhanced_smoke_mask
        )
        
        return unified_mask, detections, {
            'yolo_fire': yolo_fire_binary,
            'base_smoke': (deeplabv3_mask == 1).astype(np.uint8),
            'enhanced_smoke': enhanced_smoke_binary,
            'combined_smoke': combined_smoke_mask,
            'combined_fire': combined_fire_mask,
            'deeplabv3_original': deeplabv3_mask,
            'yolo_smoke_regions': len(yolo_smoke_regions)
        }

def process_video(video_path, output_dir, yolo_model_path, deeplabv3_model_path, confidence_threshold=0.5):
    """Process video with unified fire and smoke segmentation (optimized version)"""
    
    # Convert .ts to .mp4 if needed
    print(f"=== VIDEO PREPROCESSING ===")
    converted_video_path = convert_ts_to_mp4(video_path, output_dir)
    if converted_video_path != video_path:
        print(f"Using converted video: {converted_video_path}")
        video_path = converted_video_path
    else:
        print(f"Using original video: {video_path}")
    
    # Initialize unified segmentation system
    unified_seg = UnifiedFireSmokeSegmentation(
        yolo_model_path, deeplabv3_model_path, confidence_threshold
    )
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output paths
    base_name = Path(video_path).stem
    mask_output = os.path.join(output_dir, f"{base_name}_masks.mp4")
    overlay_output = os.path.join(output_dir, f"{base_name}_overlay.mp4")
    detection_output = os.path.join(output_dir, f"{base_name}_detections.mp4")
    
    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mask_writer = cv2.VideoWriter(mask_output, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_output, fourcc, fps, (width, height))
    detection_writer = cv2.VideoWriter(detection_output, fourcc, fps, (width, height))
    
    # Statistics tracking
    frame_count = 0
    fire_detections = []
    smoke_detections = []
    processing_times = []
    yolo_detection_counts = []
    
    print("Processing video frames with optimized fire and smoke segmentation...")
    
    # Process every frame for best quality
    frame_skip = 1  # Process every frame
    print(f"Processing every frame for best quality (no frame skipping)")
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame with optimized approach
            unified_mask, detections, debug_masks = unified_seg.process_frame(frame)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Create visualizations
            colored_mask = unified_seg.create_colored_mask(unified_mask)
            
            # Create overlay
            alpha = 0.6
            overlay = cv2.addWeighted(frame, alpha, colored_mask, 1-alpha, 0)
            
            # Create detection visualization
            detection_vis = unified_seg.create_detection_visualization(frame, detections, unified_mask)
            
            # Write frames
            mask_writer.write(colored_mask)
            overlay_writer.write(overlay)
            detection_writer.write(detection_vis)
            
            # Calculate statistics
            total_pixels = unified_mask.shape[0] * unified_mask.shape[1]
            fire_pixels = np.sum(unified_mask == 2)
            smoke_pixels = np.sum(unified_mask == 1)
            
            fire_coverage = (fire_pixels / total_pixels) * 100
            smoke_coverage = (smoke_pixels / total_pixels) * 100
            
            fire_detections.append(fire_coverage)
            smoke_detections.append(smoke_coverage)
            yolo_detection_counts.append(len(detections))
            
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    mask_writer.release()
    overlay_writer.release()
    detection_writer.release()
    
    # Re-encode videos to H.264 for browser compatibility
    print("Re-encoding videos to H.264 for browser compatibility...")
    conversion_time_total = 0.0
    
    def reencode_video_to_h264(input_path, temp_suffix="_temp_h264"):
        """Re-encode video from mp4v to H.264 codec for browser compatibility"""
        temp_path = input_path.replace('.mp4', f'{temp_suffix}.mp4')
        start_time = time.time()
        conversion_success = False
        
        try:
            # Use ffmpeg to re-encode to H.264 (normalize timestamps)
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',       # Use H.264 codec
                '-preset', 'fast',       # Fast encoding preset
                '-crf', '23',           # Good quality setting
                '-vsync', 'cfr',        # Use constant frame rate for output
                '-r', str(fps),         # Explicitly set frame rate from input
                '-avoid_negative_ts', 'make_zero',  # Normalize timestamps to start at 0
                '-start_at_zero',       # Force video to start at timestamp 0
                '-c:a', 'copy',         # Copy audio (if any) without re-encoding
                '-y',                   # Overwrite output file if exists
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with re-encoded version
                os.remove(input_path)
                os.rename(temp_path, input_path)
                print(f"  ✓ Re-encoded: {os.path.basename(input_path)}")
                conversion_success = True
            else:
                print(f"  ✗ Failed to re-encode {os.path.basename(input_path)}: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            print(f"  ✗ Error re-encoding {os.path.basename(input_path)}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        finally:
            conversion_duration = time.time() - start_time
            print(f"[BENCH] segmentation_conversion_time_file={os.path.basename(input_path)}:{conversion_duration:.4f}")
        
        return conversion_success, conversion_duration
    
    # Re-encode all three output videos
    videos_to_reencode = [mask_output, overlay_output, detection_output]
    successful_reencodes = 0
    
    for video_path in videos_to_reencode:
        if os.path.exists(video_path):
            success, duration = reencode_video_to_h264(video_path)
            conversion_time_total += duration
            if success:
                successful_reencodes += 1
    
    print(f"Re-encoding complete: {successful_reencodes}/{len(videos_to_reencode)} videos successfully re-encoded to H.264")
    
    # Calculate final statistics
    avg_fire_coverage = np.mean(fire_detections)
    max_fire_coverage = np.max(fire_detections)
    avg_smoke_coverage = np.mean(smoke_detections)
    max_smoke_coverage = np.max(smoke_detections)
    avg_yolo_detections = np.mean(yolo_detection_counts)
    avg_processing_time = np.mean(processing_times)
    
    frames_with_fire = np.sum(np.array(fire_detections) > 0.1)
    frames_with_smoke = np.sum(np.array(smoke_detections) > 0.1)
    frames_with_yolo_detections = np.sum(np.array(yolo_detection_counts) > 0)
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"=== FIRE STATISTICS ===")
    print(f"Average fire coverage: {avg_fire_coverage:.2f}%")
    print(f"Maximum fire coverage: {max_fire_coverage:.2f}%")
    print(f"Frames with significant fire: {frames_with_fire}/{frame_count}")
    print(f"=== SMOKE STATISTICS ===")
    print(f"Average smoke coverage: {avg_smoke_coverage:.2f}%")
    print(f"Maximum smoke coverage: {max_smoke_coverage:.2f}%")
    print(f"Frames with significant smoke: {frames_with_smoke}/{frame_count}")
    print(f"=== YOLO DETECTION STATISTICS ===")
    print(f"Average YOLO detections per frame: {avg_yolo_detections:.1f}")
    print(f"Frames with YOLO detections: {frames_with_yolo_detections}/{frame_count}")
    print(f"=== PERFORMANCE ===")
    print(f"Average processing time: {avg_processing_time:.3f}s per frame")
    print("Re-encoding videos to H.264 for browser compatibility...")
    
    # Convert videos to H.264 for browser compatibility
    videos_to_convert = [
        (mask_output, "masks"),
        (overlay_output, "overlay"), 
        (detection_output, "detections")
    ]
    
    converted_count = 0
    for video_path, video_type in videos_to_convert:
        if os.path.exists(video_path):
            # Create H.264 version
            base_name = os.path.splitext(video_path)[0]
            h264_path = base_name + "_h264.mp4"
            
            try:
                conversion_stage_start = time.time()
                # Use ffmpeg to convert to H.264 (normalize timestamps)
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-c:v', 'libx264',        # H.264 codec
                    '-preset', 'fast',        # Fast encoding
                    '-crf', '23',             # Good quality
                    '-vsync', 'cfr',          # Constant frame rate
                    '-r', str(fps),           # Match original FPS
                    '-avoid_negative_ts', 'make_zero',  # Normalize timestamps to start at 0
                    '-start_at_zero',         # Force video to start at timestamp 0
                    '-pix_fmt', 'yuv420p',    # Web compatibility
                    '-movflags', '+faststart', # Optimize for streaming
                    h264_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and os.path.exists(h264_path):
                    # Replace original with H.264 version
                    os.remove(video_path)
                    os.rename(h264_path, video_path)
                    converted_count += 1
                    print(f"✓ Converted {video_type} video to H.264")
                else:
                    print(f"✗ Failed to convert {video_type} video: {result.stderr}")
            except Exception as e:
                print(f"✗ Error converting {video_type} video: {e}")
            finally:
                conversion_duration = time.time() - conversion_stage_start
                conversion_time_total += conversion_duration
                print(f"[BENCH] segmentation_conversion_time_file={os.path.basename(video_path)}:{conversion_duration:.4f}")
    
    print(f"Re-encoding complete: {converted_count}/3 videos successfully re-encoded to H.264")
    print(f"[BENCH] segmentation_conversion_time_total={conversion_time_total:.4f}")
    
    print(f"Outputs saved:")
    print(f"  Optimized masks video: {mask_output}")
    print(f"  Optimized overlay video: {overlay_output}")
    print(f"  Optimized detections video: {detection_output}")
    
    return mask_output, overlay_output, detection_output

def process_image(image_path, output_dir, yolo_model_path, deeplabv3_model_path, confidence_threshold=0.5):
    """Process image with unified fire and smoke segmentation"""
    
    print(f"=== IMAGE PREPROCESSING ===")
    print(f"Using image: {image_path}")
    
    # Initialize unified segmentation system
    unified_seg = UnifiedFireSmokeSegmentation(
        yolo_model_path, deeplabv3_model_path, confidence_threshold
    )
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"Image shape: {frame.shape}")
    
    # Process the single frame
    print("Processing image with optimized fire and smoke segmentation...")
    start_time = time.time()
    
    # Process frame with optimized approach
    unified_mask, detections, debug_masks = unified_seg.process_frame(frame)
    
    processing_time = time.time() - start_time
    
    # Create visualizations
    colored_mask = unified_seg.create_colored_mask(unified_mask)
    
    # Create overlay
    alpha = 0.6
    overlay = cv2.addWeighted(frame, alpha, colored_mask, 1-alpha, 0)
    
    # Create detection visualization
    detection_vis = unified_seg.create_detection_visualization(frame, detections, unified_mask)
    
    # Create output paths
    base_name = Path(image_path).stem
    mask_output = os.path.join(output_dir, f"{base_name}_masks.jpg")
    overlay_output = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    detection_output = os.path.join(output_dir, f"{base_name}_detections.jpg")
    
    # Save images
    cv2.imwrite(mask_output, colored_mask)
    cv2.imwrite(overlay_output, overlay)
    cv2.imwrite(detection_output, detection_vis)
    
    # Calculate statistics
    total_pixels = unified_mask.shape[0] * unified_mask.shape[1]
    fire_pixels = np.sum(unified_mask == 2)
    smoke_pixels = np.sum(unified_mask == 1)
    
    fire_coverage = (fire_pixels / total_pixels) * 100
    smoke_coverage = (smoke_pixels / total_pixels) * 100
    
    print(f"\nProcessing complete!")
    print(f"=== FIRE STATISTICS ===")
    print(f"Fire coverage: {fire_coverage:.2f}%")
    print(f"=== SMOKE STATISTICS ===")
    print(f"Smoke coverage: {smoke_coverage:.2f}%")
    print(f"=== YOLO DETECTION STATISTICS ===")
    print(f"YOLO detections: {len(detections)}")
    print(f"=== PERFORMANCE ===")
    print(f"Processing time: {processing_time:.3f}s")
    print(f"Outputs saved:")
    print(f"  Optimized masks image: {mask_output}")
    print(f"  Optimized overlay image: {overlay_output}")
    print(f"  Optimized detections image: {detection_output}")
    
    return mask_output, overlay_output, detection_output

def main():
    parser = argparse.ArgumentParser(description='Unified Fire & Smoke Segmentation Processing - Optimized Version')
    
    # Make video and image mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', help='Path to input video file')
    input_group.add_argument('--image', help='Path to input image file')
    
    parser.add_argument('--yolo_model', default='../models/best.pt', help='Path to YOLOv8 model file')
    parser.add_argument('--deeplabv3_model', default='../models/best_deeplabv3.pth', help='Path to DeepLabv3+ model file')
    parser.add_argument('--output_dir', default='optimized_results', help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold for YOLO detection (optimized: 0.1)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== OPTIMIZED FIRE & SMOKE SEGMENTATION ===")
    
    if args.video:
        print(f"Input video: {args.video}")
        input_path = args.video
        processing_type = "video"
    else:
        print(f"Input image: {args.image}")
        input_path = args.image
        processing_type = "image"
    
    print(f"YOLO model: {args.yolo_model}")
    print(f"DeepLabv3+ model: {args.deeplabv3_model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Optimization: Enhanced smoke analysis only (no enhanced fire)")
    
    try:
        if processing_type == "video":
            mask_output, overlay_output, detection_output = process_video(
                args.video, args.output_dir, args.yolo_model, args.deeplabv3_model, args.confidence
            )
            print(f"\nSuccess! Optimized video processing completed.")
        else:
            mask_output, overlay_output, detection_output = process_image(
                args.image, args.output_dir, args.yolo_model, args.deeplabv3_model, args.confidence
            )
            print(f"\nSuccess! Optimized image processing completed.")
        
    except Exception as e:
        print(f"Error processing {processing_type}: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 