/**
 * Video Synchronization Core Module
 * 
 * Provides the core functionality for video synchronization:
 * - Master-slave relationship setup
 * - Basic video synchronization
 * - Polling for sync issues
 * - Play/pause/seek handling
 */

// Core VideoSynchronizer class definition
class VideoSynchronizerCore {
    constructor(videoA, videoB, pairId) {
        // Setup variables
        this.videoA = videoA;
        this.videoB = videoB;
        this.pairId = pairId;
        this.isSyncing = false;
        this.destroyed = false;
        this.masterVideo = videoB; // videoB (processed) is the master
        this.slaveVideo = videoA;  // videoA (original) is the slave
        this.isHandlingPlayPause = false;
        
        // Video time offset compensation (for videos with different start_time in container)
        this.timeOffset = 0;
        this.timeOffsetDetected = false;
        
        // Adaptive playback rate compensation (for VFR vs CFR or frame rate mismatches)
        this.playbackRateAdjustment = 1.0;
        this.driftHistory = [];
        this.maxDriftHistory = 10;
        this.lastDriftCorrection = 0;
        
        // Video state tracking
        this.state = {
            playing: false,
            ended: false,
            currentTime: 0,
            lastPlayEvent: 0,
            lastPauseEvent: 0
        };
        
        console.log(`Creating core synchronizer for pair ${pairId}`);
        
        // Add video classes for debugging
        this.videoA.classList.add('sync-video-a');
        this.videoB.classList.add('sync-video-b');
        
        // Initialize core functionality
        this.initializeCore();
    }
    
    /**
     * Initialize the core synchronizer
     */
    initializeCore() {
        // Bind methods to preserve 'this' context
        this.handleMasterPlay = this.handleMasterPlay.bind(this);
        this.handleMasterPause = this.handleMasterPause.bind(this);
        this.handleMasterSeek = this.handleMasterSeek.bind(this);
        this.handleMasterEnded = this.handleMasterEnded.bind(this);
        this.handleClick = this.handleClick.bind(this);
        this.handleVideoMouseDown = this.handleVideoMouseDown.bind(this);
        this.pollForSync = this.pollForSync.bind(this);
        
        // Detect initial time offset when videos are ready
        this.detectTimeOffset();
        
        // Set up master-slave relationship
        this.setupMasterSlaveRelationship();
        
        // Start polling for sync
        this.startPolling();
        
        console.log(`VideoSynchronizerCore for pair ${this.pairId} initialized`);
    }
    
    /**
     * Detect time offset between videos
     * Some videos have different start_time in their container (e.g., original video starts at 0.533s, processed at 0.0s)
     */
    detectTimeOffset() {
        // Wait for both videos to be ready
        const checkReady = () => {
            if (this.videoA.readyState >= 2 && this.videoB.readyState >= 2 && !this.timeOffsetDetected) {
                // Measure the natural time difference when both videos are at position 0
                // This captures container-level timing offsets
                const timeA = this.videoA.currentTime;
                const timeB = this.videoB.currentTime;
                
                // Record ANY initial difference (lowered threshold from 0.1 to 0.01)
                const initialDiff = Math.abs(timeA - timeB);
                if (initialDiff > 0.01) {
                    this.timeOffset = timeA - timeB;
                    console.log(`Detected time offset: ${this.timeOffset.toFixed(3)}s (videoA: ${timeA.toFixed(3)}s, videoB: ${timeB.toFixed(3)}s)`);
                    this.timeOffsetDetected = true;
                    
                    // Immediately apply the offset to sync the videos
                    if (this.slaveVideo.currentTime !== this.getCompensatedSlaveTime()) {
                        console.log(`Applying initial offset correction`);
                        this.slaveVideo.currentTime = this.getCompensatedSlaveTime();
                    }
                } else {
                    console.log(`No significant time offset detected (diff: ${initialDiff.toFixed(3)}s)`);
                    this.timeOffsetDetected = true;
                }
            }
        };
        
        // Check immediately
        checkReady();
        
        // Also listen for loadedmetadata and loadeddata events
        const onLoaded = () => {
            setTimeout(checkReady, 100); // Small delay to ensure currentTime is accurate
        };
        
        this.videoA.addEventListener('loadedmetadata', onLoaded);
        this.videoB.addEventListener('loadedmetadata', onLoaded);
        this.videoA.addEventListener('loadeddata', onLoaded);
        this.videoB.addEventListener('loadeddata', onLoaded);
        
        // Also check after a short delay
        setTimeout(checkReady, 200);
        setTimeout(checkReady, 500);
        setTimeout(checkReady, 1000);
        
        // Remove listeners after detection
        setTimeout(() => {
            this.videoA.removeEventListener('loadedmetadata', onLoaded);
            this.videoB.removeEventListener('loadedmetadata', onLoaded);
            this.videoA.removeEventListener('loadeddata', onLoaded);
            this.videoB.removeEventListener('loadeddata', onLoaded);
        }, 5000);
    }
    
    /**
     * Get the compensated current time for slave video accounting for offset
     */
    getCompensatedSlaveTime() {
        return this.masterVideo.currentTime + this.timeOffset;
    }
    
    /**
     * Set up the master-slave event listeners
     */
    setupMasterSlaveRelationship() {
        // Remove existing listeners if any
        this.removeMasterListeners();
        
        // Add event listeners to master video
        this.masterVideo.addEventListener('play', this.handleMasterPlay);
        this.masterVideo.addEventListener('pause', this.handleMasterPause);
        this.masterVideo.addEventListener('seeked', this.handleMasterSeek);
        this.masterVideo.addEventListener('ended', this.handleMasterEnded);
        
        // Add click handlers to both videos (both click and mousedown for better responsiveness)
        this.videoA.addEventListener('click', this.handleClick);
        this.videoB.addEventListener('click', this.handleClick);
        this.videoA.addEventListener('mousedown', this.handleVideoMouseDown);
        this.videoB.addEventListener('mousedown', this.handleVideoMouseDown);
        
        // Add double-click handlers for reset
        this.videoA.addEventListener('dblclick', this.handleDoubleClick.bind(this));
        this.videoB.addEventListener('dblclick', this.handleDoubleClick.bind(this));
        
        // If master video is already playing, sync slave now
        if (!this.masterVideo.paused) {
            this.syncSlave('play');
        }
        
        console.log(`Master (${this.masterVideo === this.videoA ? 'A' : 'B'}) - Slave (${this.slaveVideo === this.videoA ? 'A' : 'B'}) relationship established`);
    }
    
    /**
     * Remove master event listeners
     */
    removeMasterListeners() {
        if (this.masterVideo) {
            this.masterVideo.removeEventListener('play', this.handleMasterPlay);
            this.masterVideo.removeEventListener('pause', this.handleMasterPause);
            this.masterVideo.removeEventListener('seeked', this.handleMasterSeek);
            this.masterVideo.removeEventListener('ended', this.handleMasterEnded);
        }
        
        // Also remove video click handlers
        if (this.videoA) {
            this.videoA.removeEventListener('mousedown', this.handleVideoMouseDown);
        }
        if (this.videoB) {
            this.videoB.removeEventListener('mousedown', this.handleVideoMouseDown);
        }
    }
    
    /**
     * Switch master and slave roles
     */
    switchMasterSlave() {
        console.log(`Switching master-slave roles for pair ${this.pairId}`);
        
        // Remove current master listeners
        this.removeMasterListeners();
        
        // Swap master and slave
        const temp = this.masterVideo;
        this.masterVideo = this.slaveVideo;
        this.slaveVideo = temp;
        
        // Set up new master-slave relationship
        this.setupMasterSlaveRelationship();
    }
    
    /**
     * Synchronize the slave video with the master
     */
    syncSlave(action) {
        if (this.isSyncing || this.destroyed) return;
        this.isSyncing = true;
        
        try {
            console.log(`Syncing slave for action: ${action}`);
            
            // Always sync time first (with offset compensation)
            const targetSlaveTime = this.getCompensatedSlaveTime();
            const timeDiff = Math.abs(targetSlaveTime - this.slaveVideo.currentTime);
            if (timeDiff > 0.05) {
                console.log(`Correcting time difference: ${timeDiff.toFixed(3)}s (offset: ${this.timeOffset.toFixed(3)}s)`);
                this.slaveVideo.currentTime = targetSlaveTime;
            }
            
            // Sync playback rate (with adaptive adjustment)
            const targetRate = this.masterVideo.playbackRate * this.playbackRateAdjustment;
            if (Math.abs(this.slaveVideo.playbackRate - targetRate) > 0.001) {
                this.slaveVideo.playbackRate = targetRate;
            }
            
            // Then handle the action
            switch (action) {
                case 'play':
                    // Update state
                    this.state.playing = true;
                    this.state.ended = false;
                    this.state.lastPlayEvent = Date.now();
                    this.state.currentTime = this.masterVideo.currentTime;
                    
                    // Play the slave
                    const playPromise = this.slaveVideo.play();
                    if (playPromise) {
                        playPromise.catch(e => {
                            console.warn('Slave play failed:', e);
                            // If slave play fails, try switching roles
                            if (!this.isSyncing) {
                                this.switchMasterSlave();
                            }
                        });
                    }
                    break;
                    
                case 'pause':
                    // Update state
                    this.state.playing = false;
                    this.state.lastPauseEvent = Date.now();
                    this.state.currentTime = this.masterVideo.currentTime;
                    
                    // Pause the slave
                    this.slaveVideo.pause();
                    break;
                    
                case 'seeked':
                    // Just update time
                    this.state.currentTime = this.masterVideo.currentTime;
                    break;
                    
                case 'ended':
                    // Mark as ended and pause both
                    this.state.playing = false;
                    this.state.ended = true;
                    this.state.currentTime = this.masterVideo.duration || 0;
                    
                    // Ensure slave is also at end and paused
                    this.slaveVideo.pause();
                    if (this.slaveVideo.duration) {
                        this.slaveVideo.currentTime = this.slaveVideo.duration;
                    }
                    break;
                    
                case 'reset':
                    // Reset both videos to beginning
                    this.state.ended = false;
                    this.state.currentTime = 0;
                    
                    this.masterVideo.currentTime = 0;
                    this.slaveVideo.currentTime = 0;
                    
                    // If master is playing, play slave too
                    if (!this.masterVideo.paused) {
                        this.slaveVideo.play().catch(e => console.warn('Slave play after reset failed:', e));
                    }
                    break;
                    
                case 'ratechange':
                    // Sync playback rate
                    this.slaveVideo.playbackRate = this.masterVideo.playbackRate;
                    break;
            }
        } catch (error) {
            console.error('Error syncing slave:', error);
        } finally {
            // Reset sync flag after a delay
            setTimeout(() => {
                this.isSyncing = false;
            }, 50);
        }
    }
    
    /**
     * Calculate adaptive playback rate adjustment based on drift history
     */
    calculatePlaybackRateAdjustment() {
        if (this.driftHistory.length < 2) return this.playbackRateAdjustment;
        
        // Calculate average drift over recent history
        const avgDrift = this.driftHistory.reduce((a, b) => a + b, 0) / this.driftHistory.length;
        
        // If there's consistent drift in one direction, adjust playback rate
        // Positive drift = slave is ahead, needs to slow down
        // Negative drift = slave is behind, needs to speed up
        if (Math.abs(avgDrift) > 0.03) {
            // Calculate adjustment factor (more aggressive for faster convergence)
            // For every 100ms of drift, adjust by 0.2%
            const adjustment = -avgDrift * 0.002;
            
            // Clamp adjustment to reasonable range (0.98 to 1.02 = ±2%)
            this.playbackRateAdjustment = Math.max(0.98, Math.min(1.02, this.playbackRateAdjustment + adjustment));
            
            console.log(`Adaptive playback rate: ${this.playbackRateAdjustment.toFixed(4)}x (drift: ${avgDrift.toFixed(3)}s)`);
        }
        
        return this.playbackRateAdjustment;
    }
    
    /**
     * Start polling for synchronization
     */
    startPolling() {
        // Poll every 100ms for tighter synchronization
        this.pollInterval = setInterval(this.pollForSync, 100);
    }
    
    /**
     * Poll for sync issues and correct them
     */
    pollForSync() {
        if (this.isSyncing || this.destroyed) return;
        
        try {
            // Check for play/pause state mismatch
            if (!this.masterVideo.paused && this.slaveVideo.paused) {
                console.log('Poll detected slave paused while master playing, fixing...');
                this.syncSlave('play');
            } else if (this.masterVideo.paused && !this.slaveVideo.paused) {
                console.log('Poll detected slave playing while master paused, fixing...');
                this.syncSlave('pause');
            }
            
            // Check for playback rate mismatch
            if (Math.abs(this.masterVideo.playbackRate - this.slaveVideo.playbackRate) > 0.01) {
                console.log('Poll detected playback rate mismatch, fixing...');
                this.syncSlave('ratechange');
            }
            
            // Check for time drift if both are playing (with offset compensation and adaptive rate)
            if (!this.masterVideo.paused && !this.slaveVideo.paused) {
                const targetSlaveTime = this.getCompensatedSlaveTime();
                const currentDrift = targetSlaveTime - this.slaveVideo.currentTime;
                const timeDiff = Math.abs(currentDrift);
                
                // Track drift for adaptive playback rate
                this.driftHistory.push(currentDrift);
                if (this.driftHistory.length > this.maxDriftHistory) {
                    this.driftHistory.shift();
                }
                
                // If drift is significant, do a hard correction
                if (timeDiff > 0.2) {
                    console.log(`Poll detected large time drift: ${timeDiff.toFixed(3)}s, hard correcting...`);
                    this.slaveVideo.currentTime = targetSlaveTime;
                    // Reset drift history after hard correction
                    this.driftHistory = [];
                }
                // For smaller drift, use adaptive playback rate adjustment
                else if (timeDiff > 0.03) {
                    const now = Date.now();
                    // Update playback rate every 1.5 seconds (more responsive)
                    if (now - this.lastDriftCorrection > 1500) {
                        const newRate = this.calculatePlaybackRateAdjustment();
                        const baseRate = this.masterVideo.playbackRate;
                        const adjustedRate = baseRate * newRate;
                        
                        // Only update if the change is significant
                        if (Math.abs(this.slaveVideo.playbackRate - adjustedRate) > 0.0001) {
                            this.slaveVideo.playbackRate = adjustedRate;
                            this.lastDriftCorrection = now;
                            console.log(`Adjusting slave playback rate to ${adjustedRate.toFixed(4)}x to reduce drift (${currentDrift.toFixed(3)}s)`);
                        }
                    }
                }
            }
            
            // Check for ended state inconsistency and duration mismatches
            const masterNearEnd = this.masterVideo.duration && (this.masterVideo.currentTime >= this.masterVideo.duration - 0.2);
            const slaveNearEnd = this.slaveVideo.duration && (this.slaveVideo.currentTime >= this.slaveVideo.duration - 0.2);
            
            // Handle duration mismatches - if one video ends significantly before the other
            const durationDiff = Math.abs((this.masterVideo.duration || 0) - (this.slaveVideo.duration || 0));
            if (durationDiff > 0.5) { // More than 0.5 second difference
                console.log(`Duration mismatch detected: master=${this.masterVideo.duration?.toFixed(2)}s, slave=${this.slaveVideo.duration?.toFixed(2)}s`);
                
                // If master ends first, pause slave when master ends
                if (masterNearEnd && this.masterVideo.duration < this.slaveVideo.duration) {
                    console.log('Master ended first due to duration mismatch, pausing slave');
                    this.slaveVideo.pause();
                    this.state.ended = true;
                }
                // If slave ends first, pause master when slave ends  
                else if (slaveNearEnd && this.slaveVideo.duration < this.masterVideo.duration) {
                    console.log('Slave ended first due to duration mismatch, pausing master');
                    this.masterVideo.pause();
                    this.state.ended = true;
                }
            } else {
                // Normal synchronization for videos with similar durations
                if (masterNearEnd && !slaveNearEnd) {
                    console.log('Poll detected master at end but slave not, fixing...');
                    if (this.slaveVideo.duration) {
                        this.slaveVideo.currentTime = this.slaveVideo.duration - 0.1;
                    }
                } else if (!masterNearEnd && slaveNearEnd) {
                    console.log('Poll detected slave at end but master not, fixing...');
                    if (this.masterVideo.duration) {
                        this.masterVideo.currentTime = this.slaveVideo.currentTime;
                    }
                }
            }
            
            // Detect restart after end
            if (this.state.ended && this.masterVideo.currentTime < 0.5) {
                console.log('Poll detected restart after end, resetting both videos');
                this.state.ended = false;
                this.slaveVideo.currentTime = this.getCompensatedSlaveTime();
                
                // If master is playing, ensure slave plays too
                if (!this.masterVideo.paused) {
                    this.slaveVideo.play().catch(e => console.warn('Slave play after restart failed:', e));
                }
            }
        } catch (error) {
            console.error('Error in polling:', error);
        }
    }
    
    // Event handlers
    
    handleMasterPlay() {
        console.log(`Master play event (${this.masterVideo === this.videoA ? 'A' : 'B'})`);
        
        // Check if this is a restart after end
        if (this.state.ended) {
            console.log('Detected play after end, resetting both videos');
            this.masterVideo.currentTime = 0;
            this.slaveVideo.currentTime = 0;
            this.state.ended = false;
        }
        
        this.syncSlave('play');
    }
    
    handleMasterPause() {
        console.log(`Master pause event (${this.masterVideo === this.videoA ? 'A' : 'B'})`);
        
        // Don't sync pause if it's at the end (handled by ended event)
        if (this.masterVideo.duration && 
            Math.abs(this.masterVideo.currentTime - this.masterVideo.duration) < 0.2) {
            console.log('Ignoring pause at end of video');
            return;
        }
        
        this.syncSlave('pause');
    }
    
    handleMasterSeek() {
        console.log(`Master seek event (${this.masterVideo === this.videoA ? 'A' : 'B'}) to ${this.masterVideo.currentTime.toFixed(2)}`);
        
        // Reset ended state if seeking to beginning
        if (this.state.ended && this.masterVideo.currentTime < 0.5) {
            console.log('Seeking to beginning after end, resetting ended state');
            this.state.ended = false;
        }
        
        this.syncSlave('seeked');
    }
    
    handleMasterEnded() {
        console.log(`Master ended event (${this.masterVideo === this.videoA ? 'A' : 'B'})`);
        this.state.ended = true;
        this.syncSlave('ended');
        
        // Auto-reset to beginning after a short delay for better UX
        setTimeout(() => {
            if (this.state.ended && !this.destroyed) {
                console.log('Auto-resetting videos to beginning after end');
                this.masterVideo.currentTime = 0;
                this.slaveVideo.currentTime = 0;
                this.state.ended = false;
            }
        }, 100);
    }
    
    handleClick(event) {
        // If videos are ended, clicking either should reset both
        if (this.state.ended) {
            console.log('Click after end detected, preparing for restart');
            
            // Don't reset immediately, just prepare state
            // The actual reset will happen when play is triggered
            this.state.ended = true;
            
            // If click was on slave, switch roles so the clicked video becomes master
            if (event.target === this.slaveVideo) {
                console.log('Click was on slave, switching master/slave roles');
                this.switchMasterSlave();
            }
        }
    }
    
    /**
     * Handle video mousedown - faster response than click
     */
    handleVideoMouseDown(event) {
        // Prevent default to avoid text selection, etc.
        event.preventDefault();
        
        // If videos are ended, clicking either should reset both
        if (this.state.ended) {
            console.log('Click after end detected, preparing for restart');
            
            // Don't reset immediately, just prepare state
            // The actual reset will happen when play is triggered
            this.state.ended = false;
            
            // If click was on slave, switch roles so the clicked video becomes master
            if (event.target === this.slaveVideo) {
                console.log('Click was on slave, switching master/slave roles');
                this.switchMasterSlave();
            }
            
            // Play videos
            setTimeout(() => {
                if (!this.destroyed) {
                    this.playVideos();
                }
            }, 0);
            
            return;
        }
        
        // If not ended, toggle play/pause
        if (this.masterVideo.paused) {
            this.playVideos();
        } else {
            this.pauseVideos();
        }
    }
    
    handleDoubleClick(event) {
        event.preventDefault();
        console.log('Double-click reset detected');
        
        // Force reset both videos
        this.masterVideo.pause();
        this.slaveVideo.pause();
        this.masterVideo.currentTime = 0;
        this.slaveVideo.currentTime = 0;
        this.state.ended = false;
        
        // Play both videos after reset
        setTimeout(() => {
            if (!this.destroyed) {
                this.masterVideo.play().catch(e => console.warn('Master play after dblclick failed:', e));
                this.syncSlave('play');
            }
        }, 100);
        
        return false;
    }
    
    /**
     * Handle play/pause button click with improved responsiveness
     */
    handlePlayPauseClick() {
        // Don't process if we're already handling a click
        if (this.isHandlingPlayPause) return;
        
        // Set flag to prevent multiple simultaneous clicks
        this.isHandlingPlayPause = true;
        
        // Check master video state
        if (this.masterVideo.paused) {
            this.playVideos();
        } else {
            this.pauseVideos();
        }
        
        // Clear the flag after a short delay
        setTimeout(() => {
            this.isHandlingPlayPause = false;
        }, 200);
    }
    
    /**
     * Play both videos
     */
    playVideos() {
        if (this.destroyed) return;
        
        console.log('Playing both videos');
        
        // If videos are at end, reset to beginning
        if (this.state.ended) {
            this.masterVideo.currentTime = 0;
            this.slaveVideo.currentTime = 0;
            this.state.ended = false;
        }
        
        // Play master video directly first for improved responsiveness
        const masterPlayPromise = this.masterVideo.play();
        
        // Then handle the slave synchronization
        if (masterPlayPromise !== undefined) {
            masterPlayPromise.catch(e => {
                console.warn('Master play failed:', e);
                // If master fails, no need to try slave
            });
        }
        
        // Force update state immediately rather than waiting for events
        this.state.playing = true;
        this.state.lastPlayEvent = Date.now();
        
        // Sync slave immediately with master for tighter synchronization
        if (!this.destroyed && !this.masterVideo.paused) {
            this.syncSlave('play');
        }
    }
    
    /**
     * Pause both videos
     */
    pauseVideos() {
        if (this.destroyed) return;
        
        console.log('Pausing both videos');
        
        // Pause master video immediately
        this.masterVideo.pause();
        
        // Force update state immediately rather than waiting for events
        this.state.playing = false;
        this.state.lastPauseEvent = Date.now();
        
        // Pause slave immediately too (don't wait for sync)
        this.slaveVideo.pause();
    }
    
    /**
     * Seek by a relative amount of seconds
     */
    seekRelative(seconds) {
        if (this.destroyed) return;
        
        console.log(`Seeking ${seconds > 0 ? 'forward' : 'backward'} by ${Math.abs(seconds)} seconds`);
        
        // Temporarily disable sync to prevent interference during seek
        this.isSyncing = true;
        
        const currentTime = this.masterVideo.currentTime;
        const duration = this.masterVideo.duration;
        
        let newTime = currentTime + seconds;
        
        // Clamp to valid range
        newTime = Math.max(0, Math.min(duration - 0.1, newTime)); // Leave small buffer before end
        
        // Store playing state
        const wasPlaying = !this.masterVideo.paused;
        
        // Pause both videos to avoid conflicts during seek
        if (wasPlaying) {
            this.masterVideo.pause();
            this.slaveVideo.pause();
        }
        
        // Apply seek to both videos simultaneously
        try {
            console.log(`Setting both videos to time: ${newTime.toFixed(2)}s`);
            
            // Set time on both videos at the same time (with offset compensation)
            this.masterVideo.currentTime = newTime;
            this.slaveVideo.currentTime = newTime + this.timeOffset;
            
            // Update internal state
            this.state.currentTime = newTime;
            
            // Reset ended state if seeking away from end
            if (this.state.ended && newTime < (duration - 1)) {
                this.state.ended = false;
            }
            
        } catch (error) {
            console.error('Error during seek:', error);
        }
        
        // Re-enable sync after a delay and resume playback if needed
        setTimeout(() => {
            this.isSyncing = false;
            
            if (wasPlaying && !this.destroyed) {
                console.log('Resuming playback after seek');
                this.masterVideo.play().catch(e => console.warn('Master play after seek failed:', e));
                this.slaveVideo.play().catch(e => console.warn('Slave play after seek failed:', e));
            }
        }, 100); // Longer delay to ensure seek completes
    }
    
    /**
     * Seek to an absolute time position in seconds
     */
    seekTo(timeInSeconds) {
        if (this.destroyed) return;
        
        console.log(`Seeking to absolute position: ${timeInSeconds.toFixed(2)}s`);
        
        // Temporarily disable sync to prevent interference during seek
        this.isSyncing = true;
        
        const duration = this.masterVideo.duration;
        
        // Clamp to valid range
        let newTime = Math.max(0, Math.min(duration - 0.1, timeInSeconds)); // Leave small buffer before end
        
        // Store playing state
        const wasPlaying = !this.masterVideo.paused;
        
        // Pause both videos to avoid conflicts during seek
        if (wasPlaying) {
            this.masterVideo.pause();
            this.slaveVideo.pause();
        }
        
        // Apply seek to both videos simultaneously
        try {
            console.log(`Setting both videos to time: ${newTime.toFixed(2)}s`);
            
            // Set time on both videos at the same time (with offset compensation)
            this.masterVideo.currentTime = newTime;
            this.slaveVideo.currentTime = newTime + this.timeOffset;
            
            // Update internal state
            this.state.currentTime = newTime;
            
            // Reset ended state if seeking away from end
            if (this.state.ended && newTime < (duration - 1)) {
                this.state.ended = false;
            }
            
        } catch (error) {
            console.error('Error during absolute seek:', error);
        }
        
        // Re-enable sync after a delay and resume playback if needed
        setTimeout(() => {
            this.isSyncing = false;
            
            if (wasPlaying && !this.destroyed) {
                console.log('Resuming playback after absolute seek');
                this.masterVideo.play().catch(e => console.warn('Master play after seek failed:', e));
                this.slaveVideo.play().catch(e => console.warn('Slave play after seek failed:', e));
            }
        }, 100); // Longer delay to ensure seek completes
    }
    
    /**
     * Set playback speed for both videos
     */
    setPlaybackSpeed(speed) {
        if (this.destroyed) return;
        
        console.log(`Setting playback speed to: ${speed}x`);
        
        try {
            // Set speed for both videos (master at requested speed, slave with adjustment)
            this.masterVideo.playbackRate = speed;
            this.slaveVideo.playbackRate = speed * this.playbackRateAdjustment;
        } catch (error) {
            console.error('Error setting playback speed:', error);
        }
    }
    
    /**
     * Clean up core resources
     */
    destroyCore() {
        if (this.destroyed) return;
        
        // Clear intervals
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        
        // Remove master listeners
        this.removeMasterListeners();
        
        // Remove click handlers
        if (this.videoA) {
            this.videoA.removeEventListener('click', this.handleClick);
            this.videoA.removeEventListener('dblclick', this.handleDoubleClick);
        }
        
        if (this.videoB) {
            this.videoB.removeEventListener('click', this.handleClick);
            this.videoB.removeEventListener('dblclick', this.handleDoubleClick);
        }
        
        console.log(`VideoSynchronizerCore for pair ${this.pairId} destroyed`);
    }
}

// Make available globally
window.VideoSynchronizerCore = VideoSynchronizerCore; 