/**
 * Video Synchronization Timeline Module
 * 
 * Provides a visual timeline bar that shows fire/smoke detection states:
 * - Gray: No detection
 * - Blue: Smoke detected 
 * - Red: Fire detected
 * - Violet: Both fire and smoke detected
 * 
 * The timeline allows clicking to seek to specific positions and updates
 * in real-time as the video plays.
 */

class VideoSynchronizerTimeline {
    constructor(synchronizer, options = {}) {
        this.synchronizer = synchronizer;
        this.options = {
            height: 40,
            showLabels: true,
            showTime: true,
            ...options
        };
        
        // Timeline state
        this.detectionData = null;
        this.videoDuration = 0;
        this.currentTime = 0;
        this.timelineElement = null;
        this.progressElement = null;
        this.canvasElement = null;
        this.ctx = null;
        
        // Color scheme for different detection types
        this.colors = {
            none: '#6c757d',      // Gray
            smoke: '#007bff',     // Blue  
            fire: '#dc3545',      // Red
            both: '#8A2BE2'       // Violet
        };
        
        // Event handlers
        this.boundUpdateTimeline = this.updateTimeline.bind(this);
        this.boundHandleTimelineClick = this.handleTimelineClick.bind(this);
        
        console.log(`Creating timeline for pair ${synchronizer.pairId}`);
        
        // Initialize when videos are loaded
        this.initialize();
    }
    
    /**
     * Initialize the timeline component
     */
    initialize() {
        // Wait for video metadata to load
        const checkVideoReady = () => {
            const masterVideo = this.synchronizer.masterVideo;
            if (masterVideo && !isNaN(masterVideo.duration) && masterVideo.duration > 0) {
                this.videoDuration = masterVideo.duration;
                this.createTimelineElement();
                this.loadDetectionMetadata();
                this.setupEventListeners();
                console.log(`Timeline initialized for pair ${this.synchronizer.pairId}, duration: ${this.videoDuration}s`);
            } else {
                // Try again in 100ms
                setTimeout(checkVideoReady, 100);
            }
        };
        
        checkVideoReady();
    }
    
    /**
     * Create the timeline DOM element
     */
    createTimelineElement() {
        // Find the video container (look for the closest common parent of both videos)
        const videoA = this.synchronizer.videoA;
        const videoB = this.synchronizer.videoB;
        
        // Find the row containing both videos
        let container = videoA.closest('.row');
        if (!container) {
            container = videoA.closest('.card-body');
        }
        if (!container) {
            container = videoA.parentElement;
        }
        
        // Create timeline container
        const timelineContainer = document.createElement('div');
        timelineContainer.className = 'video-timeline-container';
        timelineContainer.style.cssText = `
            margin: 15px 0;
            padding: 10px;
            background-color: var(--card-bg, #2a2a2a);
            border: 1px solid var(--card-border, #3d3d3d);
            border-radius: 8px;
        `;
        
        // Create title (hidden in fullscreen via CSS)
        if (this.options.showLabels) {
            const title = document.createElement('div');
            title.className = 'timeline-title';
            title.textContent = 'Fire & Smoke Detection Timeline';
            title.style.cssText = `
                color: var(--text-color, #eee);
                font-weight: 600;
                margin-bottom: 10px;
                text-align: center;
                font-size: 14px;
            `;
            timelineContainer.appendChild(title);
        }
        
        // Create timeline wrapper
        const timelineWrapper = document.createElement('div');
        timelineWrapper.style.cssText = 'position: relative; width: 100%;';
        
        // Create canvas for detection visualization
        this.canvasElement = document.createElement('canvas');
        this.canvasElement.style.cssText = `
            width: 100%;
            height: ${this.options.height}px;
            border: 1px solid var(--card-border, #444);
            border-radius: 4px;
            cursor: pointer;
            background-color: #1a1a1a;
        `;
        this.canvasElement.width = 800; // Will be adjusted based on container width
        this.canvasElement.height = this.options.height;
        
        // Create progress indicator
        this.progressElement = document.createElement('div');
        this.progressElement.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: ${this.options.height}px;
            background-color: #ffffff;
            border-radius: 1px;
            box-shadow: 0 0 4px rgba(255,255,255,0.8);
            pointer-events: none;
            z-index: 10;
        `;
        
        // Create time display
        const timeDisplay = document.createElement('div');
        timeDisplay.className = 'timeline-time-display';
        timeDisplay.style.cssText = `
            text-align: center;
            color: var(--text-color, #eee);
            font-size: 12px;
            margin-top: 5px;
            font-family: monospace;
        `;
        timeDisplay.textContent = '00:00 / 00:00';
        this.timeDisplayElement = timeDisplay;
        
        // Create legend
        if (this.options.showLabels) {
            const legend = this.createLegend();
            timelineContainer.appendChild(legend);
        }
        
        // Assemble timeline
        timelineWrapper.appendChild(this.canvasElement);
        timelineWrapper.appendChild(this.progressElement);
        timelineContainer.appendChild(timelineWrapper);
        
        if (this.options.showTime) {
            timelineContainer.appendChild(timeDisplay);
        }
        
        // Insert timeline into DOM - place it between the videos and the existing controls
        const existingControls = container.querySelector('.video-controls-container');
        if (existingControls) {
            container.insertBefore(timelineContainer, existingControls);
        } else {
            container.appendChild(timelineContainer);
        }
        
        this.timelineElement = timelineContainer;
        this.ctx = this.canvasElement.getContext('2d');
        
        // Adjust canvas size to fit container
        this.resizeCanvas();
    }
    
    /**
     * Create legend showing color meanings
     */
    createLegend() {
        const legend = document.createElement('div');
        legend.className = 'timeline-legend';
        legend.style.cssText = `
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        `;
        
        const items = [
            { color: this.colors.none, label: 'No Detection' },
            { color: this.colors.smoke, label: 'Smoke' },
            { color: this.colors.fire, label: 'Fire' },
            { color: this.colors.both, label: 'Both' }
        ];
        
        items.forEach(item => {
            const legendItem = document.createElement('div');
            legendItem.style.cssText = `
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 12px;
                color: var(--text-color, #eee);
            `;
            
            const colorBox = document.createElement('div');
            colorBox.style.cssText = `
                width: 12px;
                height: 12px;
                background-color: ${item.color};
                border-radius: 2px;
                border: 1px solid rgba(255,255,255,0.3);
            `;
            
            const label = document.createElement('span');
            label.textContent = item.label;
            
            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legend.appendChild(legendItem);
        });
        
        return legend;
    }
    
    /**
     * Resize canvas to fit container
     */
    resizeCanvas() {
        if (!this.canvasElement || !this.ctx) return;
        
        const rect = this.canvasElement.getBoundingClientRect();
        const containerWidth = this.canvasElement.parentElement.clientWidth;
        
        this.canvasElement.width = Math.max(containerWidth - 20, 400);
        this.canvasElement.height = this.options.height;
        
        // Redraw after resize
        this.drawTimeline();
    }
    
    /**
     * Load detection metadata from the server
     */
    async loadDetectionMetadata() {
        try {
            // Try to get metadata from video element data attributes
            const masterVideo = this.synchronizer.masterVideo;
            const metadataUrls = masterVideo.dataset.metadataUrls ? 
                masterVideo.dataset.metadataUrls.split(',') : [];
            
            // If no metadata URLs specified, try to construct from video filename
            if (metadataUrls.length === 0) {
                const videoSrc = masterVideo.src || (masterVideo.querySelector('source') ? masterVideo.querySelector('source').src : '');
                if (videoSrc) {
                    const filename = videoSrc.split('/').pop().replace(/\.[^/.]+$/, '');
                    const baseUrl = window.location.origin;
                    metadataUrls.push(`${baseUrl}/metadata/${filename}_metadata.json`);
                }
            }
            
            // Try each metadata URL until one works
            for (const url of metadataUrls) {
                try {
                    console.log(`Trying to load metadata from: ${url}`);
                    const response = await fetch(url);
                    if (response.ok) {
                        this.detectionData = await response.json();
                        console.log(`Loaded detection metadata:`, this.detectionData);
                        this.drawTimeline();
                        return;
                    }
                } catch (error) {
                    console.warn(`Failed to load metadata from ${url}:`, error);
                }
            }
            
            console.warn('No detection metadata found, showing empty timeline');
            this.drawTimeline();
            
        } catch (error) {
            console.error('Error loading detection metadata:', error);
            this.drawTimeline(); // Draw empty timeline
        }
    }
    
    /**
     * Draw the timeline visualization
     */
    drawTimeline() {
        if (!this.ctx) return;
        
        const canvas = this.canvasElement;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw background
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, width, height);
        
        // Draw border
        this.ctx.strokeStyle = '#444';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
        
        if (!this.detectionData || !this.detectionData.detections || this.videoDuration <= 0) {
            // Draw empty timeline with gray color
            this.ctx.fillStyle = this.colors.none;
            this.ctx.fillRect(1, 1, width - 2, height - 2);
            return;
        }
        
        // Create detection segments
        const segments = this.createDetectionSegments();
        
        // Draw segments
        segments.forEach(segment => {
            const startX = Math.floor((segment.startTime / this.videoDuration) * (width - 2)) + 1;
            const endX = Math.floor((segment.endTime / this.videoDuration) * (width - 2)) + 1;
            const segmentWidth = Math.max(endX - startX, 1);
            
            this.ctx.fillStyle = this.colors[segment.type];
            this.ctx.fillRect(startX, 1, segmentWidth, height - 2);
        });
        
        // Draw time markers every 10 seconds (optional)
        this.drawTimeMarkers();
    }
    
    /**
     * Create detection segments from metadata
     */
    createDetectionSegments() {
        if (!this.detectionData || !this.detectionData.detections) {
            return [{ startTime: 0, endTime: this.videoDuration, type: 'none' }];
        }
        
        const segments = [];
        const detections = this.detectionData.detections;
        const segmentDuration = 0.5; // Group detections in 0.5 second segments
        
        // Find the maximum time in detection data
        const maxDetectionTime = Math.max(...detections.map(d => d.time));
        const lastDetection = detections[detections.length - 1];
        
        // Create time-based segments
        for (let time = 0; time < this.videoDuration; time += segmentDuration) {
            const segmentEnd = Math.min(time + segmentDuration, this.videoDuration);
            
            // Find all detections in this time segment
            const segmentDetections = detections.filter(detection => 
                detection.time >= time && detection.time < segmentEnd
            );
            
            let segmentType = 'none';
            
            // If we're beyond the detection data but the last detection shows smoke/fire,
            // extend that detection type to cover the gap
            if (segmentDetections.length === 0 && time > maxDetectionTime && lastDetection) {
                // Use the last detection type to fill the gap at the end
                segmentType = lastDetection.detection_type || 'none';
            } else if (segmentDetections.length > 0) {
                // Determine dominant detection type in this segment
                const hasFireDetections = segmentDetections.some(d => 
                    d.detection_type === 'fire' || d.detection_type === 'both'
                );
                const hasSmokeDetections = segmentDetections.some(d => 
                    d.detection_type === 'smoke' || d.detection_type === 'both'
                );
                const hasBothDetections = segmentDetections.some(d => d.detection_type === 'both');
                
                if (hasBothDetections || (hasFireDetections && hasSmokeDetections)) {
                    segmentType = 'both';
                } else if (hasFireDetections) {
                    segmentType = 'fire';
                } else if (hasSmokeDetections) {
                    segmentType = 'smoke';
                }
            }
            
            segments.push({
                startTime: time,
                endTime: segmentEnd,
                type: segmentType
            });
        }
        
        // Merge consecutive segments of the same type
        const mergedSegments = [];
        let currentSegment = null;
        
        segments.forEach(segment => {
            if (!currentSegment || currentSegment.type !== segment.type) {
                if (currentSegment) {
                    mergedSegments.push(currentSegment);
                }
                currentSegment = { ...segment };
            } else {
                currentSegment.endTime = segment.endTime;
            }
        });
        
        if (currentSegment) {
            mergedSegments.push(currentSegment);
        }
        
        return mergedSegments;
    }
    
    /**
     * Draw time markers on the timeline
     */
    drawTimeMarkers() {
        if (this.videoDuration <= 10) return; // Don't show markers for short videos
        
        const canvas = this.canvasElement;
        const width = canvas.width;
        const height = canvas.height;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 1;
        
        // Draw marker every 10 seconds
        const markerInterval = 10;
        for (let time = markerInterval; time < this.videoDuration; time += markerInterval) {
            const x = Math.floor((time / this.videoDuration) * (width - 2)) + 1;
            
            this.ctx.beginPath();
            this.ctx.moveTo(x + 0.5, 1);
            this.ctx.lineTo(x + 0.5, height - 1);
            this.ctx.stroke();
        }
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Listen for timeline clicks
        this.canvasElement.addEventListener('click', this.boundHandleTimelineClick);
        
        // Listen for video time updates
        const masterVideo = this.synchronizer.masterVideo;
        masterVideo.addEventListener('timeupdate', this.boundUpdateTimeline);
        masterVideo.addEventListener('seeked', this.boundUpdateTimeline);
        
        // Listen for window resize
        window.addEventListener('resize', () => {
            setTimeout(() => this.resizeCanvas(), 100);
        });
    }
    
    /**
     * Handle timeline click events
     */
    handleTimelineClick(event) {
        if (!this.canvasElement || this.videoDuration <= 0) return;
        
        const rect = this.canvasElement.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const timePercent = clickX / rect.width;
        const seekTime = Math.max(0, Math.min(timePercent * this.videoDuration, this.videoDuration));
        
        console.log(`Timeline clicked: seeking to ${seekTime.toFixed(2)}s`);
        
        // Use the synchronizer to seek both videos
        if (this.synchronizer.core && this.synchronizer.core.seekTo) {
            this.synchronizer.core.seekTo(seekTime);
        } else {
            // Fallback: seek both videos directly
            this.synchronizer.masterVideo.currentTime = seekTime;
            this.synchronizer.slaveVideo.currentTime = seekTime;
        }
    }
    
    /**
     * Update timeline progress indicator
     */
    updateTimeline() {
        if (!this.progressElement || !this.timeDisplayElement || this.videoDuration <= 0) return;
        
        const currentTime = this.synchronizer.masterVideo.currentTime;
        const progress = Math.min(currentTime / this.videoDuration, 1);
        
        // Update progress indicator position
        const timelineWidth = this.canvasElement.clientWidth;
        const progressX = progress * timelineWidth;
        this.progressElement.style.left = `${progressX}px`;
        
        // Update time display
        if (this.options.showTime) {
            const currentTimeStr = this.formatTime(currentTime);
            const durationStr = this.formatTime(this.videoDuration);
            this.timeDisplayElement.textContent = `${currentTimeStr} / ${durationStr}`;
        }
        
        this.currentTime = currentTime;
    }
    
    /**
     * Format time in MM:SS format
     */
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * Destroy the timeline component
     */
    destroy() {
        // Remove event listeners
        if (this.canvasElement) {
            this.canvasElement.removeEventListener('click', this.boundHandleTimelineClick);
        }
        
        const masterVideo = this.synchronizer.masterVideo;
        if (masterVideo) {
            masterVideo.removeEventListener('timeupdate', this.boundUpdateTimeline);
            masterVideo.removeEventListener('seeked', this.boundUpdateTimeline);
        }
        
        // Remove timeline element from DOM
        if (this.timelineElement && this.timelineElement.parentNode) {
            this.timelineElement.parentNode.removeChild(this.timelineElement);
        }
        
        console.log(`Timeline destroyed for pair ${this.synchronizer.pairId}`);
    }
}

// Make available globally
window.VideoSynchronizerTimeline = VideoSynchronizerTimeline;
