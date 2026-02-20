/**
 * Video Synchronization UI Controls Module
 * 
 * Provides the UI controls functionality for video synchronization:
 * - Play/pause/seek controls
 * - Speed controls
 * - Time display
 * - Auto/manual speed toggle
 */

// UI Controls class definition
class VideoSynchronizerUI {
    constructor(synchronizer) {
        this.synchronizer = synchronizer;
        this.controls = null;
        
        // Keep track of UI state
        this.uiState = {
            lastUpdateTime: 0,
            isUpdating: false
        };
        
        console.log(`Creating UI controls for pair ${synchronizer.pairId}`);
    }
    
    /**
     * Create unified video controls
     */
    createControls() {
        // Create main container for controls
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'row video-controls-container';
        controlsContainer.style.cssText = 'margin-top: 10px; margin-bottom: 20px;';
        
        // Create a centered column for the controls
        const controlsCol = document.createElement('div');
        controlsCol.className = 'col-12';
        controlsCol.style.cssText = 'display: flex; justify-content: center;';
        
        // Create container for all controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'video-unified-controls';
        controlsDiv.style.cssText = 'padding: 10px; border: 1px solid var(--card-border); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 10px; background-color: var(--card-bg);';
        
        // Create video playback controls
        const playbackControlsDiv = document.createElement('div');
        playbackControlsDiv.className = 'playback-controls';
        playbackControlsDiv.style.cssText = 'display: flex; gap: 5px; align-items: center; margin-right: 15px;';
        
        // Create play/pause button
        const playPauseBtn = document.createElement('button');
        playPauseBtn.className = 'btn btn-sm btn-primary';
        playPauseBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            playPauseBtn.textContent = '▶';
        }
        playPauseBtn.title = 'Play/Pause';
        
        // Add event handlers - use both mousedown (faster) and click (better compatibility)
        playPauseBtn.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent focus issues
            this.synchronizer.handlePlayPauseClick();
        });
        
        // Still keep click for mobile devices and accessibility
        playPauseBtn.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default to avoid double-triggering
        });
        
        // Add a touchstart handler for mobile devices
        playPauseBtn.addEventListener('touchstart', (e) => {
            e.preventDefault(); // Prevent default behavior
            this.synchronizer.handlePlayPauseClick();
        });
        
        // Create rewind 5s button
        const rewindBtn = document.createElement('button');
        rewindBtn.className = 'btn btn-sm btn-outline-primary';
        rewindBtn.innerHTML = '<i class="bi bi-arrow-counterclockwise"></i> 5s';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            rewindBtn.textContent = '⟲ 5s';
        }
        rewindBtn.title = 'Rewind 5 seconds';
        
        // Create forward 5s button
        const forwardBtn = document.createElement('button');
        forwardBtn.className = 'btn btn-sm btn-outline-primary';
        forwardBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> 5s';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            forwardBtn.textContent = '⟳ 5s';
        }
        forwardBtn.title = 'Forward 5 seconds';
        
        // Create capture frame button
        const captureBtn = document.createElement('button');
        captureBtn.className = 'btn btn-sm btn-success';
        captureBtn.innerHTML = '<i class="bi bi-camera"></i> Frame';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            captureBtn.textContent = '📷 Frame';
        }
        captureBtn.title = 'Capture current frame from original video';
        
        // Create time display
        const timeDisplay = document.createElement('span');
        timeDisplay.className = 'time-display';
        timeDisplay.textContent = '0:00 / 0:00';
        timeDisplay.style.cssText = 'margin: 0 10px; min-width: 100px; text-align: center;';
        
        // Create metadata status container
        const metadataContainer = document.createElement('div');
        metadataContainer.className = 'metadata-container';
        metadataContainer.style.cssText = 'display: flex; align-items: center; margin-right: 15px;';
        
        // Create speed label
        const speedLabel = document.createElement('span');
        speedLabel.textContent = 'Speed: 1.0x';
        speedLabel.className = 'video-speed-label';
        speedLabel.style.cssText = 'margin: 0 10px;';
        
        // Create speed buttons
        const speedButtons = document.createElement('div');
        speedButtons.className = 'video-speed-buttons';
        speedButtons.style.cssText = 'display: flex; gap: 5px;';
        
        // Create speed options
        const speedOptions = [
            { label: '0.25x', value: 0.25 },
            { label: '0.5x', value: 0.5 },
            { label: '1.0x', value: 1.0 },
            { label: '1.5x', value: 1.5 },
            { label: '2.0x', value: 2.0 }
        ];
        
        speedOptions.forEach(option => {
            const btn = document.createElement('button');
            btn.textContent = option.label;
            btn.className = 'btn btn-sm ' + (option.value === 1.0 ? 'btn-primary' : 'btn-outline-primary');
            btn.style.cssText = 'min-width: 45px;';
            btn.dataset.speed = option.value;
            
            btn.addEventListener('click', () => {
                // Update UI
                speedButtons.querySelectorAll('button').forEach(b => {
                    b.className = 'btn btn-sm btn-outline-primary';
                });
                btn.className = 'btn btn-sm btn-primary';
                speedLabel.textContent = `Speed: ${option.label}`;
                
                // Disable auto speed mode when user manually selects speed
                if (this.synchronizer.detection) {
                    // Turn off auto speed completely
                    this.synchronizer.detection.disableAutoSpeed();
                    
                    // Update Auto button to show it's OFF
                    autoToggle.textContent = 'Auto: OFF';
                    autoToggle.className = 'btn btn-sm btn-danger';
                }
                
                // Apply the speed change
                this.synchronizer.setPlaybackSpeed(option.value);
                
                // Add visual feedback
                btn.classList.add('btn-active-feedback');
                setTimeout(() => {
                    btn.classList.remove('btn-active-feedback');
                }, 200);
            });
            
            speedButtons.appendChild(btn);
        });
        
        // Create auto speed toggle
        const autoToggle = document.createElement('button');
        autoToggle.textContent = 'Auto: ON';
        autoToggle.className = 'btn btn-sm btn-success';
        autoToggle.style.marginLeft = '10px';
        
        // Create fullscreen buttons container
        const fullscreenDiv = document.createElement('div');
        fullscreenDiv.className = 'fullscreen-controls';
        fullscreenDiv.style.cssText = 'display: flex; gap: 5px; margin-left: 15px;';
        
        // Detect if this is segmentation gallery (videos have data-video-type="overlay" or "mask")
        const isSegmentationGallery = this.synchronizer.videoA.dataset.videoType === 'overlay' || 
                                      this.synchronizer.videoB.dataset.videoType === 'mask';
        
        // Set labels based on gallery type
        const firstVideoLabel = isSegmentationGallery ? 'Overlay' : 'Orig';
        const secondVideoLabel = isSegmentationGallery ? 'Mask' : 'Proc';
        const firstVideoTitle = isSegmentationGallery ? 'Fullscreen overlay video' : 'Fullscreen original video';
        const secondVideoTitle = isSegmentationGallery ? 'Fullscreen mask video' : 'Fullscreen processed video';
        
        // Create fullscreen both button
        const fullscreenBothBtn = document.createElement('button');
        fullscreenBothBtn.className = 'btn btn-sm btn-outline-primary';
        fullscreenBothBtn.innerHTML = '<i class="bi bi-arrows-fullscreen"></i> Both';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            fullscreenBothBtn.textContent = '⤢ Both';
        }
        fullscreenBothBtn.title = 'Fullscreen both videos';
        
        // Create fullscreen original/overlay button
        const fullscreenOrigBtn = document.createElement('button');
        fullscreenOrigBtn.className = 'btn btn-sm btn-outline-primary';
        fullscreenOrigBtn.innerHTML = `<i class="bi bi-fullscreen"></i> ${firstVideoLabel}`;
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            fullscreenOrigBtn.textContent = `⤡ ${firstVideoLabel}`;
        }
        fullscreenOrigBtn.title = firstVideoTitle;
        
        // Create fullscreen processed/mask button
        const fullscreenProcBtn = document.createElement('button');
        fullscreenProcBtn.className = 'btn btn-sm btn-outline-primary';
        fullscreenProcBtn.innerHTML = `<i class="bi bi-fullscreen"></i> ${secondVideoLabel}`;
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            fullscreenProcBtn.textContent = `⤡ ${secondVideoLabel}`;
        }
        fullscreenProcBtn.title = secondVideoTitle;
        
        // Create exit fullscreen button (hidden by default, shown when in fullscreen)
        const exitFullscreenBtn = document.createElement('button');
        exitFullscreenBtn.className = 'btn btn-sm btn-outline-danger fullscreen-exit-control';
        exitFullscreenBtn.innerHTML = '<i class="bi bi-fullscreen-exit"></i> Exit';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            exitFullscreenBtn.textContent = '⤓ Exit';
        }
        exitFullscreenBtn.title = 'Exit fullscreen';
        exitFullscreenBtn.style.display = 'none';
        
        // Add event listeners
        
        // Rewind 5s
        rewindBtn.addEventListener('click', () => {
            this.synchronizer.seekRelative(-5);
            this.addVisualFeedback(rewindBtn);
        });
        
        // Forward 5s
        forwardBtn.addEventListener('click', () => {
            this.synchronizer.seekRelative(5);
            this.addVisualFeedback(forwardBtn);
        });
        
        // Capture frame
        captureBtn.addEventListener('click', () => {
            this.captureCurrentFrame();
            this.addVisualFeedback(captureBtn);
        });
        
        // Auto speed toggle
        autoToggle.addEventListener('click', () => {
            if (this.synchronizer.detection) {
                const isEnabled = this.synchronizer.detection.toggleAutoSpeed();
                autoToggle.textContent = isEnabled ? 'Auto: ON' : 'Auto: OFF';
                autoToggle.className = isEnabled ? 'btn btn-sm btn-success' : 'btn btn-sm btn-danger';
                
                // Clear user override when toggling auto speed
                this.synchronizer.detection.setUserOverride(false);
            }
            this.addVisualFeedback(autoToggle);
        });
        
        // Fullscreen both
        fullscreenBothBtn.addEventListener('click', () => {
            if (this.synchronizer.enterFullscreen) {
                this.synchronizer.enterFullscreen('both');
            }
            this.addVisualFeedback(fullscreenBothBtn);
        });
        
        // Fullscreen original
        fullscreenOrigBtn.addEventListener('click', () => {
            if (this.synchronizer.enterFullscreen) {
                this.synchronizer.enterFullscreen('original');
            }
            this.addVisualFeedback(fullscreenOrigBtn);
        });
        
        // Fullscreen processed
        fullscreenProcBtn.addEventListener('click', () => {
            if (this.synchronizer.enterFullscreen) {
                this.synchronizer.enterFullscreen('processed');
            }
            this.addVisualFeedback(fullscreenProcBtn);
        });
        
        // Exit fullscreen
        exitFullscreenBtn.addEventListener('click', () => {
            if (this.synchronizer.exitFullscreen) {
                this.synchronizer.exitFullscreen();
            }
            this.addVisualFeedback(exitFullscreenBtn);
        });
        
        // Add elements to containers
        playbackControlsDiv.appendChild(rewindBtn);
        playbackControlsDiv.appendChild(playPauseBtn);
        playbackControlsDiv.appendChild(forwardBtn);
        playbackControlsDiv.appendChild(captureBtn);
        playbackControlsDiv.appendChild(timeDisplay);
        
        metadataContainer.appendChild(speedLabel);
        
        fullscreenDiv.appendChild(fullscreenBothBtn);
        fullscreenDiv.appendChild(fullscreenOrigBtn);
        fullscreenDiv.appendChild(fullscreenProcBtn);
        fullscreenDiv.appendChild(exitFullscreenBtn);
        
        controlsDiv.appendChild(playbackControlsDiv);
        controlsDiv.appendChild(metadataContainer);
        controlsDiv.appendChild(speedButtons);
        controlsDiv.appendChild(autoToggle);
        controlsDiv.appendChild(fullscreenDiv);
        
        controlsCol.appendChild(controlsDiv);
        controlsContainer.appendChild(controlsCol);
        
        // Find the appropriate place to insert controls - in the middle between videos
        const videoContainer = this.synchronizer.masterVideo.closest('.card-body');
        if (videoContainer) {
            // Find the row that contains both video columns
            const videoRow = this.synchronizer.masterVideo.closest('.row');
            if (videoRow) {
                // Insert controls container after the row
                videoRow.parentNode.insertBefore(controlsContainer, videoRow.nextSibling);
            } else {
                // Fallback - add to the card body
                videoContainer.appendChild(controlsContainer);
            }
        }
        
        // Store references
        this.controls = {
            container: controlsDiv,
            containerParent: controlsContainer,
            playPauseBtn: playPauseBtn,
            captureBtn: captureBtn,
            timeDisplay: timeDisplay,
            label: speedLabel,
            buttons: speedButtons,
            autoToggle: autoToggle,
            metadataContainer: metadataContainer,
            exitFullscreenBtn: exitFullscreenBtn
        };
        
        // Start with a loading metadata status
        this.updateMetadataStatus(null); // null = loading state
        
        // Start time update interval
        this.startTimeUpdateInterval();
        
        return this.controls;
    }
    
    /**
     * Add visual feedback to a button
     */
    addVisualFeedback(button) {
        button.classList.add('btn-active-feedback');
        setTimeout(() => {
            button.classList.remove('btn-active-feedback');
        }, 200);
    }
    
    /**
     * Start interval to update time display
     */
    startTimeUpdateInterval() {
        // Update time display every 250ms
        this.timeUpdateInterval = setInterval(() => {
            this.updateTimeDisplay();
        }, 250);
        
        // Also attach to timeupdate event for smoother display
        if (this.synchronizer.masterVideo) {
            this.synchronizer.masterVideo.addEventListener('timeupdate', this.updateTimeDisplay.bind(this));
        }
    }
    
    /**
     * Update time display
     */
    updateTimeDisplay() {
        if (!this.controls || !this.controls.timeDisplay || this.synchronizer.destroyed) return;
        
        // Throttle updates to avoid excessive DOM operations
        const now = Date.now();
        if (this.uiState.isUpdating || now - this.uiState.lastUpdateTime < 100) return;
        
        this.uiState.isUpdating = true;
        
        try {
            const currentTime = this.synchronizer.masterVideo.currentTime || 0;
            const duration = this.synchronizer.masterVideo.duration || 0;
            
            this.controls.timeDisplay.textContent = `${this.formatTime(currentTime)} / ${this.formatTime(duration)}`;
            
            // Also update play/pause button state
            if (this.controls.playPauseBtn) {
                if (this.synchronizer.masterVideo.paused) {
                    this.controls.playPauseBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
                    if (!document.querySelector('link[href*="bootstrap-icons"]')) {
                        this.controls.playPauseBtn.textContent = '▶';
                    }
                } else {
                    this.controls.playPauseBtn.innerHTML = '<i class="bi bi-pause-fill"></i>';
                    if (!document.querySelector('link[href*="bootstrap-icons"]')) {
                        this.controls.playPauseBtn.textContent = '⏸';
                    }
                }
            }
        } finally {
            this.uiState.isUpdating = false;
            this.uiState.lastUpdateTime = now;
        }
    }
    
    /**
     * Format seconds into MM:SS
     */
    formatTime(seconds) {
        seconds = Math.max(0, Math.floor(seconds));
        const minutes = Math.floor(seconds / 60);
        seconds = seconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
    
    /**
     * Update metadata status indicator in controls
     */
    updateMetadataStatus(isLoaded) {
        if (!this.controls) return;
        
        // Remove existing status if any
        const existingStatus = this.controls.metadataContainer.querySelector('.metadata-status');
        if (existingStatus) {
            existingStatus.remove();
        }
        
        // Create status indicator as a button to match Auto button style
        const statusBtn = document.createElement('button');
        
        if (isLoaded === null) {
            // Loading state
            statusBtn.className = 'btn btn-sm btn-warning metadata-status';
            statusBtn.innerHTML = 'Metadata: <i class="bi bi-hourglass"></i>';
            if (!document.querySelector('link[href*="bootstrap-icons"]')) {
                statusBtn.textContent = 'Metadata: ...';
            }
        } else if (isLoaded) {
            // Success state
            statusBtn.className = 'btn btn-sm btn-success metadata-status';
            statusBtn.innerHTML = 'Metadata: <i class="bi bi-check"></i>';
            if (!document.querySelector('link[href*="bootstrap-icons"]')) {
                statusBtn.textContent = 'Metadata: ✓';
            }
        } else {
            // Failed state
            statusBtn.className = 'btn btn-sm btn-danger metadata-status';
            statusBtn.innerHTML = 'Metadata: <i class="bi bi-x"></i>';
            if (!document.querySelector('link[href*="bootstrap-icons"]')) {
                statusBtn.textContent = 'Metadata: ✕';
            }
            
            // Add retry functionality on click
            statusBtn.addEventListener('click', () => {
                if (this.synchronizer.detection && this.synchronizer.detection.loadDetectionMetadata) {
                    console.log('Retrying metadata load...');
                    this.synchronizer.detection.loadDetectionMetadata();
                    this.addVisualFeedback(statusBtn);
                }
            });
        }
        
        statusBtn.style.marginRight = '10px';
        
        // Add to controls container
        this.controls.metadataContainer.prepend(statusBtn);
    }
    
    /**
     * Destroy UI and clean up resources
     */
    destroy() {
        if (this.timeUpdateInterval) {
            clearInterval(this.timeUpdateInterval);
        }
        
        if (this.synchronizer.masterVideo) {
            this.synchronizer.masterVideo.removeEventListener('timeupdate', this.updateTimeDisplay);
        }
        
        // Remove controls from DOM
        if (this.controls && this.controls.containerParent) {
            this.controls.containerParent.remove();
        }
        
        this.controls = null;
    }
    
    /**
     * Capture current frame from the original video
     */
    captureCurrentFrame() {
        if (!this.synchronizer || this.synchronizer.destroyed) {
            console.error('Cannot capture frame: synchronizer not available');
            return;
        }
        
        const currentTime = this.synchronizer.masterVideo.currentTime;
        const originalVideo = this.synchronizer.videoA; // Original video (slave)
        
        console.log(`Capturing frame at ${currentTime.toFixed(3)}s from original video`);
        
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match video dimensions
        canvas.width = originalVideo.videoWidth || originalVideo.offsetWidth;
        canvas.height = originalVideo.videoHeight || originalVideo.offsetHeight;
        
        // Draw the current frame from original video to canvas
        try {
            ctx.drawImage(originalVideo, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to blob
            canvas.toBlob((blob) => {
                if (blob) {
                    this.saveFrameToServer(blob, currentTime);
                } else {
                    console.error('Failed to create frame blob');
                    this.showCaptureNotification('Failed to capture frame', 'error');
                }
            }, 'image/jpeg', 0.95); // High quality JPEG
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            this.showCaptureNotification('Error capturing frame', 'error');
        }
    }
    
    /**
     * Save captured frame to server
     */
    async saveFrameToServer(blob, timestamp) {
        try {
            // Get video filename for naming the frame
            const videoSrc = this.synchronizer.videoA.src || this.synchronizer.videoA.currentSrc;
            const videoFilename = videoSrc.split('/').pop().split('.')[0]; // Remove extension
            
            // Create filename with timestamp
            const frameFilename = `${videoFilename}_frame_${timestamp.toFixed(3)}s.jpg`;
            
            // Create FormData to send the frame
            const formData = new FormData();
            formData.append('frame', blob, frameFilename);
            formData.append('timestamp', timestamp.toString());
            formData.append('video_name', videoFilename);
            
            // Send to server
            const response = await fetch('/capture-frame/', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Frame saved successfully:', result);
                this.showCaptureNotification(`Frame captured at ${timestamp.toFixed(3)}s`, 'success');
            } else {
                throw new Error(`Server error: ${response.status}`);
            }
            
        } catch (error) {
            console.error('Error saving frame to server:', error);
            this.showCaptureNotification('Failed to save frame', 'error');
        }
    }
    
    /**
     * Show capture notification to user
     */
    showCaptureNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `capture-notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10001;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            color: white;
            background-color: ${type === 'success' ? '#28a745' : '#dc3545'};
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Make available globally
window.VideoSynchronizerUI = VideoSynchronizerUI; 