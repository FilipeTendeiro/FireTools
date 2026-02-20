/**
 * Video Synchronization Detection Module
 * 
 * Handles fire detection metadata loading, auto speed adjustment, and visual alarms
 */

// Detection handler class definition
class VideoSynchronizerDetection {
    constructor(synchronizer, options = {}) {
        this.synchronizer = synchronizer;
        this.options = options;
        
        // Speed settings for auto adjustment
        this.speedSettings = {
            normalSpeed: 1.0,     // Default normal speed
            fastSpeed: 2.0,       // Speed when no detection
            slowSpeed: 0.5,       // Speed when detection present
            currentSpeed: 1.0,    // Current playback rate
            autoSpeedEnabled: true, // Enable auto speed adjustment
            detectionState: false,  // Current detection state (true = detection active)
            detectionChangeTime: 0,  // Time of last detection state change
            lastCheckTime: 0,     // Last time we checked for detection
            checkInterval: 200,   // How often to check for detection (ms)
            adaptationDelay: 100, // Delay before applying speed change (ms)
            userOverride: false,  // Whether user manually set speed
            detectionData: null,  // Will hold the loaded metadata
            detectionRanges: [],  // Processed detection ranges for easier checking
            isMetadataLoaded: false // Flag to track if metadata is loaded
        };
        
        // Visual alarm settings
        this.alarmSettings = {
            enabled: true,           // Always enabled - no toggle
            currentAlarmState: false, // Current alarm state
            lastAlarmTime: 0,        // Last time alarm was triggered
            alarmCooldown: 500,      // Minimum time between alarm state changes (ms)
            strongAlarmThreshold: 0.8 // Confidence threshold for strong alarm
        };
        
        console.log(`Creating detection handler for pair ${synchronizer.pairId}`);
        console.log('UI available:', !!this.synchronizer.ui);
        
        // Bind methods
        this.checkForDetection = this.checkForDetection.bind(this);
        this.updateVisualAlarm = this.updateVisualAlarm.bind(this);
        
        // Try to load metadata with a small delay to ensure DOM attributes are set
        setTimeout(() => {
            if (this.options.disableDetection) {
                console.log('🚫 Detection/metadata loading disabled for this video pair');
                return;
            }
            this.loadDetectionMetadata();
        }, 100);
    }
    
    /**
     * Update visual alarm based on detection state (simple red border only)
     */
    updateVisualAlarm(isDetected, confidence = 0.5) {
        // Throttle alarm updates to prevent flickering
        const now = Date.now();
        if (now - this.alarmSettings.lastAlarmTime < this.alarmSettings.alarmCooldown) {
            return;
        }
        
        // Only update if state actually changed
        if (this.alarmSettings.currentAlarmState !== isDetected) {
            this.alarmSettings.currentAlarmState = isDetected;
            this.alarmSettings.lastAlarmTime = now;
            
            console.log(`Visual alarm ${isDetected ? 'activated' : 'deactivated'} (confidence: ${confidence.toFixed(2)})`);
            
            const processedVideo = this.synchronizer.masterVideo; // The processed video
            
            if (isDetected) {
                // Apply alarm effect to processed video only - just one border
                // Determine alarm strength based on confidence
                const useStrong = confidence >= this.alarmSettings.strongAlarmThreshold;
                const alarmClass = useStrong ? 'fire-alarm-strong' : 'fire-alarm';
                
                // Remove any existing alarm classes
                processedVideo.classList.remove('fire-alarm', 'fire-alarm-strong');
                
                // Add new alarm class to video only
                processedVideo.classList.add(alarmClass);
                
                console.log(`Applied ${alarmClass} to processed video`);
                
            } else {
                // Remove alarm effects from processed video
                processedVideo.classList.remove('fire-alarm', 'fire-alarm-strong');
                
                console.log('Removed alarm effects from processed video');
            }
        }
    }
    
    /**
     * Load detection metadata from JSON file
     */
    loadDetectionMetadata() {
        try {
            // First check if we have custom metadata URLs from the video element
            console.log('Checking master video for custom metadata URLs:', this.synchronizer.masterVideo);
            const customMetadataUrls = this.getCustomMetadataUrls(this.synchronizer.masterVideo);
            
            if (customMetadataUrls && customMetadataUrls.length > 0) {
                console.log('✅ Using custom metadata URLs:', customMetadataUrls);
                
                // Try each custom URL in sequence
                this.trySequentialUrls(customMetadataUrls, 0);
                return;
            } else {
                console.log('❌ No custom metadata URLs found, falling back to automatic detection');
            }
            
            // Fallback to automatic URL construction
            const videoSrc = this.getVideoSourceUrl(this.synchronizer.masterVideo);
            console.log('Video source URL:', videoSrc);
            if (!videoSrc) {
                console.warn('Could not determine video source URL for metadata loading');
                return;
            }
            
            // Convert video URL to metadata URL
            const metadataUrl = this.getMetadataUrl(videoSrc);
            console.log(`Attempting to load metadata from ${metadataUrl}`);
            
            // Fetch the metadata file
            fetch(metadataUrl)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Loaded detection metadata:', data);
                    this.speedSettings.detectionData = data;
                    this.speedSettings.isMetadataLoaded = true;
                    
                    // Process the detection data into time ranges for easier checking
                    this.processDetectionRanges();
                    
                    // Add status indicator to show metadata is loaded
                    console.log('Metadata loaded successfully, updating UI status');
                    if (this.synchronizer.ui && this.synchronizer.ui.updateMetadataStatus) {
                        this.synchronizer.ui.updateMetadataStatus(true);
                    } else {
                        console.warn('UI or updateMetadataStatus method not available');
                    }
                    
                    // Start detection checking if auto speed is enabled
                    if (this.speedSettings.autoSpeedEnabled) {
                        this.startDetectionChecking();
                    }
                })
                .catch(error => {
                    console.warn('Error loading detection metadata:', error);
                    
                    // Try alternate URL format if the first one failed
                    const altMetadataUrl = this.getAlternateMetadataUrl(videoSrc);
                    console.log(`Trying alternate metadata URL: ${altMetadataUrl}`);
                    
                    // Try the first alternative URL
                    this.tryLoadMetadata(altMetadataUrl)
                        .catch(err => {
                            console.warn(`Failed with first alternative URL: ${err}`);
                            
                            // Generate more alternative URLs to try
                            const urlParts = videoSrc.split('/');
                            const filename = urlParts[urlParts.length - 1];
                            const baseFilename = filename.includes('_metadata') 
                                ? filename.split('_metadata')[0] 
                                : filename.split('.')[0];
                            
                            const origin = window.location.origin;
                            
                            // Try several different URL patterns as a last resort
                            const fallbackUrls = [
                                `${origin}/static/outputs/videos/${baseFilename}_metadata.json`,
                                `${origin}/outputs/result/${baseFilename}_metadata.json`,
                                `/outputs/result/${baseFilename}_metadata.json`,
                                `/static/outputs/result/${baseFilename}_metadata.json`
                            ];
                            
                            console.log(`Trying fallback URLs:`, fallbackUrls);
                            
                            // Try each URL in sequence until one works
                            this.trySequentialUrls(fallbackUrls, 0);
                        });
                });
        } catch (error) {
            console.error('Error in loadDetectionMetadata:', error);
        }
    }
    
    /**
     * Get the source URL from a video element
     */
    getVideoSourceUrl(videoElement) {
        // First try to get src directly from the video element
        if (videoElement.src) {
            return videoElement.src;
        }
        
        // If that doesn't work, check all source elements
        const sources = videoElement.querySelectorAll('source');
        for (const source of sources) {
            if (source.src) {
                return source.src;
            }
        }
        
        return null;
    }
    
    /**
     * Get custom metadata URLs from video element data attribute
     */
    getCustomMetadataUrls(videoElement) {
        const metadataUrls = videoElement.dataset.metadataUrls;
        console.log('Checking for custom metadata URLs on video element:', videoElement);
        console.log('Data attribute metadataUrls:', metadataUrls);
        if (metadataUrls) {
            const urls = metadataUrls.split(',');
            console.log('Parsed custom metadata URLs:', urls);
            return urls;
        }
        console.log('No custom metadata URLs found');
        return null;
    }
    
    /**
     * Convert video URL to metadata URL
     */
    getMetadataUrl(videoUrl) {
        // Extract the filename from the URL
        const urlParts = videoUrl.split('/');
        const filename = urlParts[urlParts.length - 1]; 
        
        // Check if filename already contains "_metadata" to prevent duplication
        const baseFilename = filename.includes('_metadata') 
            ? filename.split('_metadata')[0] 
            : filename.split('.')[0]; // Remove extension
        
        // Construct the metadata URL - try the same directory as the video first
        const basePath = urlParts.slice(0, -1).join('/');
        return `${basePath}/${baseFilename}_metadata.json`;
    }
    
    /**
     * Get alternate metadata URL format
     */
    getAlternateMetadataUrl(videoUrl) {
        const urlParts = videoUrl.split('/');
        const filename = urlParts[urlParts.length - 1];
        const baseFilename = filename.includes('_metadata') 
            ? filename.split('_metadata')[0] 
            : filename.split('.')[0];
        
        const origin = window.location.origin;
        
        // Try different path structures
        const alternateUrls = [
            `${origin}/outputs/videos/${baseFilename}_metadata.json`,
            `${origin}/static/outputs/videos/${baseFilename}_metadata.json`,
            `/outputs/videos/${baseFilename}_metadata.json`,
            `/static/outputs/videos/${baseFilename}_metadata.json`
        ];
        
        return alternateUrls[0]; // Return first alternate for initial try
    }
    
    /**
     * Try loading metadata from a specific URL
     */
    tryLoadMetadata(url) {
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Loaded detection metadata from alternate URL:', data);
                this.speedSettings.detectionData = data;
                this.speedSettings.isMetadataLoaded = true;
                
                // Process the detection data into time ranges
                this.processDetectionRanges();
                
                // Update status indicator
                if (this.synchronizer.ui) {
                    this.synchronizer.ui.updateMetadataStatus(true);
                }
                
                // Start detection checking if auto speed is enabled
                if (this.speedSettings.autoSpeedEnabled) {
                    this.startDetectionChecking();
                }
            });
    }
    
    /**
     * Try multiple URLs sequentially until one works
     */
    trySequentialUrls(urls, index) {
        if (index >= urls.length) {
            console.error('All metadata URL attempts failed');
            if (this.synchronizer.ui && this.synchronizer.ui.updateMetadataStatus) {
                this.synchronizer.ui.updateMetadataStatus(false);
            } else {
                console.warn('UI not available to show metadata failure status');
            }
            return;
        }
        
        console.log(`Trying metadata URL ${index + 1}/${urls.length}: ${urls[index]}`);
        
        this.tryLoadMetadata(urls[index])
            .catch(error => {
                console.warn(`Failed to load from ${urls[index]}:`, error);
                // Try next URL
                this.trySequentialUrls(urls, index + 1);
            });
    }
    
    /**
     * Process detection data into time ranges for easier checking
     */
    processDetectionRanges() {
        if (!this.speedSettings.detectionData || !this.speedSettings.detectionData.detections) {
            return;
        }
        
        const { detections } = this.speedSettings.detectionData;
        
        // Sort detections by time
        const sortedDetections = [...detections].sort((a, b) => a.time - b.time);
        
        if (sortedDetections.length === 0) {
            this.speedSettings.detectionRanges = [];
            return;
        }
        
        // Check if this video has mostly continuous detection (high detection density)
        const videoDuration = this.speedSettings.detectionData.total_frames / this.speedSettings.detectionData.fps;
        const detectionDensity = sortedDetections.length / videoDuration; // detections per second
        
        console.log(`Video duration: ${videoDuration}s, Detections: ${sortedDetections.length}, Density: ${detectionDensity.toFixed(2)} det/sec`);
        
        // If we have high detection density (>15 detections per second), treat as continuous
        if (detectionDensity > 15) {
            console.log('High detection density detected - treating as continuous fire/smoke video');
            
            // For high-density videos, create fewer, larger ranges to avoid gaps
            const ranges = [];
            const maxGapSec = 2.0; // Allow up to 2 seconds gap before considering it a break
            let currentRange = null;
            
            sortedDetections.forEach(detection => {
                if (!currentRange) {
                    currentRange = { 
                        start: Math.max(0, detection.time - 0.5), 
                        end: detection.time + 0.5 
                    };
                } else {
                    // Check if this detection is within the allowable gap
                    if (detection.time - currentRange.end <= maxGapSec) {
                        // Extend current range
                        currentRange.end = detection.time + 0.5;
                    } else {
                        // Gap too large, close current range and start new one
                        ranges.push(currentRange);
                        currentRange = { 
                            start: Math.max(0, detection.time - 0.5), 
                            end: detection.time + 0.5 
                        };
                    }
                }
            });
            
            // Add the last range
            if (currentRange) {
                ranges.push(currentRange);
            }
            
            // Extend the last range to the end if there are detections near the end
            if (ranges.length > 0) {
                const lastRange = ranges[ranges.length - 1];
                const lastDetection = sortedDetections[sortedDetections.length - 1];
                
                // If last detection is within 3 seconds of video end, extend range to end
                if (lastDetection && (videoDuration - lastDetection.time) <= 3.0) {
                    console.log(`Extending last detection range to video end (last detection at ${lastDetection.time}s, video duration ${videoDuration}s)`);
                    lastRange.end = videoDuration;
                }
            }
            
            // If the entire video is essentially one big detection range, simplify it
            if (ranges.length === 1 && ranges[0].start < 1 && ranges[0].end > videoDuration - 1) {
                console.log('Video is essentially all fire/smoke - using full duration');
                this.speedSettings.detectionRanges = [{ start: 0, end: videoDuration }];
            } else {
                this.speedSettings.detectionRanges = ranges;
            }
        } else {
            // For sparse detections, use the original algorithm with buffers
            console.log('Sparse detection density - using original buffered approach');
            
            const ranges = [];
            const bufferSec = 1.0; // Add 1 second buffer before and after each detection
            let currentRange = null;
            
            sortedDetections.forEach(detection => {
                const start = Math.max(0, detection.time - bufferSec);
                const end = detection.time + bufferSec;
                
                if (!currentRange) {
                    currentRange = { start, end };
                } else if (start <= currentRange.end) {
                    // Extend current range if this detection overlaps
                    currentRange.end = Math.max(currentRange.end, end);
                } else {
                    // Start a new range
                    ranges.push(currentRange);
                    currentRange = { start, end };
                }
            });
            
            // Add the last range if it exists
            if (currentRange) {
                ranges.push(currentRange);
            }
            
            // Extend the last range to the end if there are detections near the end
            if (ranges.length > 0) {
                const lastRange = ranges[ranges.length - 1];
                const lastDetection = sortedDetections[sortedDetections.length - 1];
                
                // If last detection is within 3 seconds of video end, extend range to end
                if (lastDetection && (videoDuration - lastDetection.time) <= 3.0) {
                    console.log(`Extending sparse detection range to video end (last detection at ${lastDetection.time}s, video duration ${videoDuration}s)`);
                    lastRange.end = videoDuration;
                }
            }
            
            this.speedSettings.detectionRanges = ranges;
        }
        
        console.log('Processed detection ranges:', this.speedSettings.detectionRanges);
    }
    
    /**
     * Start detection checking interval
     */
    startDetectionChecking() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
        }
        
        // Check for detection regularly
        this.detectionInterval = setInterval(this.checkForDetection, this.speedSettings.checkInterval);
        console.log('Started detection checking interval');
        
        // Immediately check current state if video is already playing
        if (!this.synchronizer.masterVideo.paused) {
            console.log('Video already playing, checking initial detection state');
            setTimeout(() => {
                this.checkForDetection();
            }, 100);
        }
    }
    
    /**
     * Stop detection checking
     */
    stopDetectionChecking() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
            console.log('Stopped detection checking interval');
        }
    }
    
    /**
     * Check if current time is within a detection range
     */
    checkForDetection() {
        if (this.synchronizer.destroyed || this.synchronizer.masterVideo.paused || 
            !this.speedSettings.autoSpeedEnabled || 
            this.speedSettings.userOverride) {
            return;
        }
        
        // Skip if metadata isn't loaded
        if (!this.speedSettings.isMetadataLoaded) {
            return;
        }
        
        // Limit how often we check
        const now = Date.now();
        if (now - this.speedSettings.lastCheckTime < this.speedSettings.checkInterval) {
            return;
        }
        this.speedSettings.lastCheckTime = now;
        
        try {
            const currentTime = this.synchronizer.masterVideo.currentTime;
            const videoDuration = this.synchronizer.masterVideo.duration;
            
            // Special handling for near the end of video (last 2 seconds)
            const isNearEnd = videoDuration && (videoDuration - currentTime) <= 2.0;
            
            // Check if current time is within any detection range
            let isInDetectionRange = false;
            let highestConfidence = 0;
            
            for (const range of this.speedSettings.detectionRanges) {
                if (currentTime >= range.start && currentTime <= range.end) {
                    isInDetectionRange = true;
                    
                    // Find the detection with highest confidence in current time range
                    if (this.speedSettings.detectionData && this.speedSettings.detectionData.detections) {
                        for (const detection of this.speedSettings.detectionData.detections) {
                            if (Math.abs(detection.time - currentTime) <= 1.0) { // Within 1 second
                                highestConfidence = Math.max(highestConfidence, detection.confidence || 0.5);
                            }
                        }
                    }
                    break;
                }
            }
            
            // If we're near the end and the last detection range extends close to the end,
            // consider it as still in detection range to avoid speed jumps
            if (isNearEnd && !isInDetectionRange && this.speedSettings.detectionRanges.length > 0) {
                const lastRange = this.speedSettings.detectionRanges[this.speedSettings.detectionRanges.length - 1];
                if (lastRange.end >= videoDuration - 3.0) { // If last range ends within 3 seconds of video end
                    console.log(`Near end of video, maintaining detection state based on last range`);
                    isInDetectionRange = true;
                    highestConfidence = 0.7; // Default confidence for end-of-video detection
                }
            }
            
            // Update visual alarm based on detection state
            this.updateVisualAlarm(isInDetectionRange, highestConfidence);
            
            // If detection state has changed, update speed
            if (isInDetectionRange !== this.speedSettings.detectionState) {
                console.log(`Detection state changed to: ${isInDetectionRange ? 'DETECTED' : 'NONE'} at ${currentTime.toFixed(2)}s`);
                
                this.speedSettings.detectionState = isInDetectionRange;
                this.speedSettings.detectionChangeTime = now;
                
                // Apply speed change after a small delay to avoid rapid fluctuations
                setTimeout(() => {
                    // Only apply if the state hasn't changed again during the delay
                    if (this.speedSettings.detectionState === isInDetectionRange && 
                        !this.speedSettings.userOverride && 
                        this.speedSettings.autoSpeedEnabled) {
                        
                        const newSpeed = isInDetectionRange ? 
                            this.speedSettings.slowSpeed : 
                            this.speedSettings.fastSpeed;
                        
                        this.synchronizer.setPlaybackSpeed(newSpeed);
                        
                        // Update speed display in UI
                        if (this.synchronizer.ui && this.synchronizer.ui.controls) {
                            const controls = this.synchronizer.ui.controls;
                            
                            if (controls.label) {
                                controls.label.textContent = `Speed: ${newSpeed.toFixed(1)}x`;
                            }
                            
                            // Update button highlights
                            if (controls.buttons) {
                                const buttons = controls.buttons.querySelectorAll('button');
                                buttons.forEach(btn => {
                                    const btnSpeed = parseFloat(btn.dataset.speed);
                                    if (btnSpeed === newSpeed) {
                                        btn.className = 'btn btn-sm btn-primary';
                                    } else {
                                        btn.className = 'btn btn-sm btn-outline-primary';
                                    }
                                });
                            }
                        }
                    }
                }, this.speedSettings.adaptationDelay);
            }
            
        } catch (error) {
            console.error('Error checking for detection:', error);
        }
    }
    
    /**
     * Toggle auto speed functionality
     */
    toggleAutoSpeed() {
        this.speedSettings.autoSpeedEnabled = !this.speedSettings.autoSpeedEnabled;
        console.log(`Auto speed ${this.speedSettings.autoSpeedEnabled ? 'enabled' : 'disabled'}`);
        
        if (this.speedSettings.autoSpeedEnabled && this.speedSettings.isMetadataLoaded) {
            // Immediately check current detection state and apply appropriate speed
            this.applyCurrentDetectionSpeed();
            this.startDetectionChecking();
        } else {
            this.stopDetectionChecking();
            // Don't change speed when disabling auto mode - keep current speed
            console.log('Auto mode disabled - keeping current speed');
            
            // Update UI to show the current speed correctly
            this.updateUIForCurrentSpeed();
        }
        
        return this.speedSettings.autoSpeedEnabled;
    }
    
    /**
     * Disable auto speed functionality (called when user manually selects speed)
     */
    disableAutoSpeed() {
        this.speedSettings.autoSpeedEnabled = false;
        this.speedSettings.userOverride = false; // Clear any user override flags
        console.log('Auto speed disabled by user manual speed selection');
        
        this.stopDetectionChecking();
    }
    
    /**
     * Immediately apply the correct speed based on current video time and detection state
     */
    applyCurrentDetectionSpeed() {
        if (!this.speedSettings.isMetadataLoaded || this.synchronizer.destroyed) {
            return;
        }
        
        try {
            const currentTime = this.synchronizer.masterVideo.currentTime;
            const videoDuration = this.synchronizer.masterVideo.duration;
            
            // Special handling for near the end of video (last 2 seconds)
            const isNearEnd = videoDuration && (videoDuration - currentTime) <= 2.0;
            
            // Check if current time is within any detection range
            let isInDetectionRange = false;
            
            for (const range of this.speedSettings.detectionRanges) {
                if (currentTime >= range.start && currentTime <= range.end) {
                    isInDetectionRange = true;
                    break;
                }
            }
            
            // If we're near the end and the last detection range extends close to the end,
            // consider it as still in detection range to avoid speed jumps
            if (isNearEnd && !isInDetectionRange && this.speedSettings.detectionRanges.length > 0) {
                const lastRange = this.speedSettings.detectionRanges[this.speedSettings.detectionRanges.length - 1];
                if (lastRange.end >= videoDuration - 3.0) { // If last range ends within 3 seconds of video end
                    console.log(`Near end of video (apply method), maintaining detection state based on last range`);
                    isInDetectionRange = true;
                }
            }
            
            // Apply the appropriate speed immediately
            const newSpeed = isInDetectionRange ? 
                this.speedSettings.slowSpeed : 
                this.speedSettings.fastSpeed;
            
            console.log(`Auto mode re-enabled: Applying ${newSpeed}x speed for current time ${currentTime.toFixed(2)}s (detection: ${isInDetectionRange})`);
            
            // Update detection state
            this.speedSettings.detectionState = isInDetectionRange;
            
            // Apply speed
            this.synchronizer.setPlaybackSpeed(newSpeed);
            
            // Update speed display in UI
            if (this.synchronizer.ui && this.synchronizer.ui.controls) {
                const controls = this.synchronizer.ui.controls;
                
                if (controls.label) {
                    controls.label.textContent = `Speed: ${newSpeed.toFixed(1)}x`;
                }
                
                // Update button highlights
                if (controls.buttons) {
                    const buttons = controls.buttons.querySelectorAll('button');
                    buttons.forEach(btn => {
                        const btnSpeed = parseFloat(btn.dataset.speed);
                        if (btnSpeed === newSpeed) {
                            btn.className = 'btn btn-sm btn-primary';
                        } else {
                            btn.className = 'btn btn-sm btn-outline-primary';
                        }
                    });
                }
            }
            
        } catch (error) {
            console.error('Error applying current detection speed:', error);
        }
    }
    
    /**
     * Set user override flag when user manually changes speed
     */
    setUserOverride(override) {
        this.speedSettings.userOverride = override;
        console.log(`User override ${override ? 'enabled' : 'disabled'}`);
        
        // If user override is disabled and auto speed is enabled, restart detection checking
        if (!override && this.speedSettings.autoSpeedEnabled && this.speedSettings.isMetadataLoaded) {
            this.startDetectionChecking();
        }
    }
    
    /**
     * Update UI to reflect the current playback speed
     */
    updateUIForCurrentSpeed() {
        if (!this.synchronizer.ui || !this.synchronizer.ui.controls) {
            return;
        }
        
        try {
            // Get the actual current playback rate from the video
            const currentSpeed = this.synchronizer.masterVideo.playbackRate;
            const controls = this.synchronizer.ui.controls;
            
            // Update speed label
            if (controls.label) {
                controls.label.textContent = `Speed: ${currentSpeed.toFixed(1)}x`;
            }
            
            // Update button highlights to match current speed
            if (controls.buttons) {
                const buttons = controls.buttons.querySelectorAll('button');
                buttons.forEach(btn => {
                    const btnSpeed = parseFloat(btn.dataset.speed);
                    if (Math.abs(btnSpeed - currentSpeed) < 0.01) { // Account for floating point precision
                        btn.className = 'btn btn-sm btn-primary';
                    } else {
                        btn.className = 'btn btn-sm btn-outline-primary';
                    }
                });
            }
            
            console.log(`Updated UI to show current speed: ${currentSpeed}x`);
            
        } catch (error) {
            console.error('Error updating UI for current speed:', error);
        }
    }
    
    /**
     * Get current detection status
     */
    getDetectionStatus() {
        return {
            isMetadataLoaded: this.speedSettings.isMetadataLoaded,
            autoSpeedEnabled: this.speedSettings.autoSpeedEnabled,
            detectionState: this.speedSettings.detectionState,
            currentSpeed: this.speedSettings.currentSpeed,
            detectionRanges: this.speedSettings.detectionRanges,
            currentAlarmState: this.alarmSettings.currentAlarmState
        };
    }
    
    /**
     * Destroy and clean up resources
     */
    destroy() {
        this.stopDetectionChecking();
        
        // Clean up visual alarm effects
        this.updateVisualAlarm(false); // Clear any active alarms
        
        // Remove alarm classes from videos
        [this.synchronizer.videoA, this.synchronizer.videoB].forEach(video => {
            if (video) {
                const wrapper = video.closest('.video-container-wrapper');
                if (wrapper) {
                    wrapper.classList.remove('fire-alarm', 'fire-alarm-strong');
                }
                video.classList.remove('fire-alarm', 'fire-alarm-strong');
            }
        });
        
        // Clear references
        this.speedSettings.detectionData = null;
        this.speedSettings.detectionRanges = [];
        
        console.log(`Detection handler for pair ${this.synchronizer.pairId} destroyed`);
    }
}

// Make available globally
window.VideoSynchronizerDetection = VideoSynchronizerDetection; 