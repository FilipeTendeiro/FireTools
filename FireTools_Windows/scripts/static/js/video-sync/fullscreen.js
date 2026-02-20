/**
 * Video Synchronization Fullscreen Module
 * 
 * Provides the fullscreen functionality for video synchronization:
 * - Enter/exit fullscreen for videos
 * - Tracks fullscreen state
 * - Handles browser compatibility
 */

// Fullscreen handler class definition
class VideoSynchronizerFullscreen {
    constructor(synchronizer) {
        this.synchronizer = synchronizer;
        this.fullscreenState = null;
        this.exitFullscreenBtn = null;
        this.fullscreenChangeListener = null;
        this.isTransitioning = false;
        
        console.log(`Creating fullscreen handler for pair ${synchronizer.pairId}`);
    }
    
    /**
     * Handle fullscreen mode
     */
    enterFullscreen(mode) {
        if (this.synchronizer.destroyed) return;
        
        // Prevent rapid mode switching
        if (this.isTransitioning) {
            console.log('Mode transition in progress, ignoring request');
            return;
        }
        
        this.isTransitioning = true;
        console.log(`Entering fullscreen mode: ${mode}`);
        
        // If we're already in fullscreen, clean up the previous state without exiting fullscreen
        if (this.fullscreenState) {
            console.log('Already in fullscreen, cleaning up previous state without exiting');
            this.cleanupPreviousMode();
        }
        
        try {
            // Get the current video elements and their container
            const videoA = this.synchronizer.videoA;
            const videoB = this.synchronizer.videoB;
            
            // Get parent container that holds both videos
            const cardBody = videoA.closest('.card-body');
            const videoRow = videoA.closest('.row');
            
            if (!cardBody || !videoRow) {
                console.error('Could not find parent container for videos');
                return;
            }
            
            // Store the current fullscreen state
            this.fullscreenState = {
                mode: mode,
                container: cardBody // Always use cardBody as container for layout-preserving fullscreen
            };
            
            // Create and add exit fullscreen button for all modes (both, original, processed)
            this.createExitFullscreenButton(cardBody);
            
            // Mode-specific handling
            if (mode === 'both') {
                // For 'both' mode, we make the card-body fullscreen
                // This will include both videos side by side
                
                if (cardBody.requestFullscreen) {
                    cardBody.requestFullscreen();
                } else if (cardBody.webkitRequestFullscreen) {
                    cardBody.webkitRequestFullscreen();
                } else if (cardBody.msRequestFullscreen) {
                    cardBody.msRequestFullscreen();
                } else if (cardBody.mozRequestFullScreen) {
                    cardBody.mozRequestFullScreen();
                }
                
                // Add a class to adjust the layout in fullscreen mode
                cardBody.classList.add('fullscreen-mode');
                
                // Add inline styles for better fullscreen display with vertical centering
                cardBody.style.cssText += `
                    display: flex !important;
                    flex-direction: column !important;
                    justify-content: center !important;
                    align-items: center !important;
                    height: 100vh !important;
                    background-color: #000 !important;
                    padding: 20px !important;
                `;
                
                // Style the video container (row)
                if (videoRow) {
                    videoRow.style.cssText += `
                        width: 100% !important;
                        max-width: 100% !important;
                        height: auto !important;
                        margin: 0 0 145px 0 !important;
                        display: flex !important;
                        justify-content: center !important;
                        align-items: center !important;
                    `;
                    
                    // Style the video columns
                    const videoCols = videoRow.querySelectorAll('.col-md-6');
                    videoCols.forEach(col => {
                        col.style.cssText += `
                            flex: 1 !important;
                            max-width: 50% !important;
                            height: auto !important;
                            padding: 10px !important;
                        `;
                    });
                    
                    // Style the videos themselves
                    const videos = videoRow.querySelectorAll('video');
                    videos.forEach(video => {
                        video.style.cssText += `
                            max-width: 100% !important;
                            height: auto !important;
                            max-height: 65vh !important;
                            object-fit: contain !important;
                        `;
                    });
                }
                
                // Position timeline container consistently for Both mode
                const timelineContainerBoth = cardBody.querySelector('.video-timeline-container');
                if (timelineContainerBoth) {
                    timelineContainerBoth.style.cssText += `
                        position: absolute !important;
                        bottom: 90px !important;
                        left: 50% !important;
                        transform: translateX(-50%) !important;
                        width: 85% !important;
                        max-width: 900px !important;
                        margin: 0 !important;
                        z-index: 150 !important;
                        background-color: rgba(0, 0, 0, 0.8) !important;
                        border-radius: 8px !important;
                        padding: 12px !important;
                    `;
                }
                
                // Add fullscreen change listener for all browsers
                this.addFullscreenChangeListener();
                
            } else if (mode === 'original') {
                // For 'original', use layout-preserving fullscreen like 'both' mode
                // but show only the original video centered and larger
                
                if (cardBody.requestFullscreen) {
                    cardBody.requestFullscreen();
                } else if (cardBody.webkitRequestFullscreen) {
                    cardBody.webkitRequestFullscreen();
                } else if (cardBody.msRequestFullscreen) {
                    cardBody.msRequestFullscreen();
                } else if (cardBody.mozRequestFullScreen) {
                    cardBody.mozRequestFullScreen();
                }
                
                // Add a class to adjust the layout in single video fullscreen mode
                cardBody.classList.add('fullscreen-mode', 'single-video-original');
                
                // Add inline styles for single video display (original)
                cardBody.style.cssText += `
                    display: flex !important;
                    flex-direction: column !important;
                    justify-content: flex-start !important;
                    align-items: center !important;
                    height: 100vh !important;
                    background-color: #000 !important;
                    padding: 0px !important;
                `;
                
                // Style the video container (row) for single video
                if (videoRow) {
                    videoRow.style.cssText += `
                        width: 100% !important;
                        max-width: 100% !important;
                        height: auto !important;
                        margin: 0 0 145px 0 !important;
                        display: flex !important;
                        justify-content: center !important;
                        align-items: center !important;
                    `;
                    
                    // Hide the processed video column and let the original column fill the width
                    const videoCols = videoRow.querySelectorAll('.col-md-6');
                    videoCols.forEach((col, index) => {
                        if (index === 0) { // Original video column
                            col.style.cssText += `
                                flex: 1 1 auto !important;
                                max-width: 75% !important;
                                width: 75% !important;
                                height: auto !important;
                                padding: 0px !important;
                                display: flex !important;
                                flex-direction: column !important;
                                align-items: center !important;
                            `;
                        } else { // Processed video column
                            col.style.cssText += `
                                display: none !important;
                            `;
                        }
                    });
                    
                    // Style the original video to be larger
                    videoA.style.cssText += `
                        max-width: 100% !important;
                        width: 100vw !important;
                        height: auto !important;
                        max-height: 96vh !important;
                        object-fit: contain !important;
                    `;
                    
                    // Style the title to align with the video
                    const titleElement = videoCols[0].querySelector('h5');
                    if (titleElement) {
                        titleElement.style.cssText += `
                            align-self: flex-start !important;
                            width: 100% !important;
                            text-align: left !important;
                        `;
                    }
                }
                
                // Position timeline container consistently (same as Both mode)
                const timelineContainer = cardBody.querySelector('.video-timeline-container');
                if (timelineContainer) {
                    timelineContainer.style.cssText += `
                        position: absolute !important;
                        bottom: 90px !important;
                        left: 50% !important;
                        transform: translateX(-50%) !important;
                        width: 85% !important;
                        max-width: 900px !important;
                        margin: 0 !important;
                        z-index: 150 !important;
                        background-color: rgba(0, 0, 0, 0.8) !important;
                        border-radius: 8px !important;
                        padding: 12px !important;
                    `;
                }
                
                // Add fullscreen change listener for all browsers
                this.addFullscreenChangeListener();
                
            } else if (mode === 'processed') {
                // For 'processed', use layout-preserving fullscreen like 'both' mode
                // but show only the processed video centered and larger
                
                if (cardBody.requestFullscreen) {
                    cardBody.requestFullscreen();
                } else if (cardBody.webkitRequestFullscreen) {
                    cardBody.webkitRequestFullscreen();
                } else if (cardBody.msRequestFullscreen) {
                    cardBody.msRequestFullscreen();
                } else if (cardBody.mozRequestFullScreen) {
                    cardBody.mozRequestFullScreen();
                }
                
                // Add a class to adjust the layout in single video fullscreen mode
                cardBody.classList.add('fullscreen-mode', 'single-video-processed');
                
                // Add inline styles for single video display (processed)
                cardBody.style.cssText += `
                    display: flex !important;
                    flex-direction: column !important;
                    justify-content: flex-start !important;
                    align-items: center !important;
                    height: 100vh !important;
                    background-color: #000 !important;
                    padding: 0px !important;
                `;
                
                // Style the video container (row) for single video
                if (videoRow) {
                    videoRow.style.cssText += `
                        width: 100% !important;
                        max-width: 100% !important;
                        height: auto !important;
                        margin: 0 0 145px 0 !important;
                        display: flex !important;
                        justify-content: center !important;
                        align-items: center !important;
                    `;
                    
                    // Hide the original video column and let the processed column fill the width
                    const videoCols = videoRow.querySelectorAll('.col-md-6');
                    videoCols.forEach((col, index) => {
                        if (index === 1) { // Processed video column
                            col.style.cssText += `
                                flex: 1 1 auto !important;
                                max-width: 75% !important;
                                width: 75% !important;
                                height: auto !important;
                                padding: 0px !important;
                                display: flex !important;
                                flex-direction: column !important;
                                align-items: center !important;
                            `;
                        } else { // Original video column
                            col.style.cssText += `
                                display: none !important;
                            `;
                        }
                    });
                    
                    // Style the processed video to be larger (same as original)
                    videoB.style.cssText += `
                        max-width: 100% !important;
                        width: 100vw !important;
                        height: auto !important;
                        max-height: 96vh !important;
                        object-fit: contain !important;
                    `;
                    
                    // Style the title to align with the video
                    const titleElement = videoCols[1].querySelector('h5');
                    if (titleElement) {
                        titleElement.style.cssText += `
                            align-self: flex-start !important;
                            width: 100% !important;
                            text-align: left !important;
                        `;
                    }
                }
                
                // Position timeline container consistently (same as Both mode)
                const timelineContainerProc = cardBody.querySelector('.video-timeline-container');
                if (timelineContainerProc) {
                    timelineContainerProc.style.cssText += `
                        position: absolute !important;
                        bottom: 90px !important;
                        left: 50% !important;
                        transform: translateX(-50%) !important;
                        width: 85% !important;
                        max-width: 900px !important;
                        margin: 0 !important;
                        z-index: 150 !important;
                        background-color: rgba(0, 0, 0, 0.8) !important;
                        border-radius: 8px !important;
                        padding: 12px !important;
                    `;
                }
                
                // Add fullscreen change listener for all browsers
                this.addFullscreenChangeListener();
            }
            
            // Update UI components if they exist
            if (this.synchronizer.ui && 
                this.synchronizer.ui.controls && 
                this.synchronizer.ui.controls.exitFullscreenBtn) {
                this.synchronizer.ui.controls.exitFullscreenBtn.style.display = 'inline-block';
            }
            
            // Hide segmentation button in fullscreen modes
            this.hideSegmentationButton();
            
        } catch (error) {
            console.error('Error entering fullscreen:', error);
        }
        
        // Clear transition flag after a short delay
        setTimeout(() => {
            this.isTransitioning = false;
        }, 400);
    }
    
    /**
     * Add fullscreen change event listener for all browsers
     */
    addFullscreenChangeListener() {
        // Remove existing listeners if any
        this.removeFullscreenChangeListeners();
        
        // Store event references for cleanup
        this.fullscreenChangeListener = this.handleFullscreenChange.bind(this);
        
        // Add listeners for all browser prefixes
        document.addEventListener('fullscreenchange', this.fullscreenChangeListener);
        document.addEventListener('webkitfullscreenchange', this.fullscreenChangeListener);
        document.addEventListener('mozfullscreenchange', this.fullscreenChangeListener);
        document.addEventListener('MSFullscreenChange', this.fullscreenChangeListener);
    }
    
    /**
     * Remove fullscreen change event listeners
     */
    removeFullscreenChangeListeners() {
        if (this.fullscreenChangeListener) {
            document.removeEventListener('fullscreenchange', this.fullscreenChangeListener);
            document.removeEventListener('webkitfullscreenchange', this.fullscreenChangeListener);
            document.removeEventListener('mozfullscreenchange', this.fullscreenChangeListener);
            document.removeEventListener('MSFullscreenChange', this.fullscreenChangeListener);
            this.fullscreenChangeListener = null;
        }
    }
    
    /**
     * Handle fullscreen change event
     */
    handleFullscreenChange() {
        // Check if we're in fullscreen
        const isInFullscreen = !!(document.fullscreenElement || 
                                document.webkitFullscreenElement || 
                                document.mozFullScreenElement || 
                                document.msFullscreenElement);
        
        console.log('Fullscreen state changed: ' + (isInFullscreen ? 'entered' : 'exited'));
        
        // If exiting fullscreen
        if (!isInFullscreen) {
            console.log('Fullscreen exited');
            
            // Clean up fullscreen mode (this will restore scroll position)
            this.exitFullscreenCleanup();
            
            // Additional cleanup for any remaining fullscreen artifacts
            setTimeout(() => {
                this.forceLayoutReset();
            }, 100);
        } else {
            // Entering fullscreen
            console.log('Fullscreen entered');
            
            // Update UI for fullscreen mode
            if (this.synchronizer.ui && 
                this.synchronizer.ui.controls && 
                this.synchronizer.ui.controls.exitFullscreenBtn) {
                this.synchronizer.ui.controls.exitFullscreenBtn.style.display = 'inline-block';
            }
        }
    }
    
    /**
     * Force a complete layout reset to fix any remaining fullscreen artifacts
     */
    forceLayoutReset() {
        console.log('Forcing complete layout reset');
        
        // Find all video containers and reset their styles
        const videoContainers = document.querySelectorAll('.card-body, .video-container, .row');
        videoContainers.forEach(container => {
            if (container.style) {
                // Remove any remaining fullscreen-related styles
                container.style.cssText = '';
            }
        });
        
        // Find all video columns and ensure proper Bootstrap classes
        const videoCols = document.querySelectorAll('.col-md-6');
        if (videoCols.length === 2) {
            videoCols.forEach(col => {
                // Force remove any remaining inline styles
                col.removeAttribute('style');
                
                // Ensure proper Bootstrap classes
                col.className = 'col-md-6';
                
                // Make sure both columns are visible
                col.style.display = 'block';
                col.style.visibility = 'visible';
                col.style.opacity = '1';
                
                // Force a reflow for each column
                col.offsetHeight;
            });
            
            // Force the parent row to use proper layout
            const videoRow = document.querySelector('.row');
            if (videoRow) {
                videoRow.style.display = 'flex';
                videoRow.style.flexWrap = 'wrap';
                videoRow.offsetHeight;
                videoRow.style.display = '';
            }
        }
        
        // Find all videos and reset their styles
        const videos = document.querySelectorAll('video');
        videos.forEach(video => {
            if (video.style) {
                video.style.cssText = '';
            }
        });
        
        // Force a complete page reflow
        document.body.style.display = 'none';
        document.body.offsetHeight; // Force reflow
        document.body.style.display = '';
        
        // Trigger resize event
        window.dispatchEvent(new Event('resize'));
        
        console.log('Layout reset complete');
    }
    
    /**
     * Create and add exit fullscreen button
     */
    createExitFullscreenButton(container) {
        // Always create exit button for all modes now
        if (!container) return;
        
        // Remove existing exit button if any
        if (this.exitFullscreenBtn) {
            this.exitFullscreenBtn.remove();
        }
        
        // Create exit button
        const exitBtn = document.createElement('button');
        exitBtn.className = 'exit-fullscreen-btn';
        exitBtn.innerHTML = '<i class="bi bi-fullscreen-exit"></i> Exit Fullscreen';
        if (!document.querySelector('link[href*="bootstrap-icons"]')) {
            exitBtn.textContent = '✕ Exit Fullscreen';
        }
        exitBtn.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 10000;
            background-color: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
        `;
        
        // Add click event
        exitBtn.addEventListener('click', () => {
            this.exitFullscreen();
        });
        
        // Add to document body
        document.body.appendChild(exitBtn);
        this.exitFullscreenBtn = exitBtn;
    }
    
    /**
     * Exit fullscreen
     */
    exitFullscreen() {
        console.log('Exiting fullscreen manually');
        
        try {
            // Use the appropriate browser-specific method to exit fullscreen
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            } else {
                // Fallback for browsers that don't support the standard methods
                // Force cleanup manually
                console.log('No standard exit fullscreen method found, forcing cleanup');
                this.exitFullscreenCleanup();
            }
        } catch (error) {
            console.error('Error exiting fullscreen:', error);
            // If there's an error, force cleanup manually
            this.exitFullscreenCleanup();
        }
    }
    
    /**
     * Clean up after exiting fullscreen
     */
    exitFullscreenCleanup() {
        console.log('Running fullscreen cleanup');
        
        // Clean up based on fullscreen mode
        if (this.fullscreenState) {
            const container = this.fullscreenState.container;
            
            if ((this.fullscreenState.mode === 'both' || 
                 this.fullscreenState.mode === 'original' || 
                 this.fullscreenState.mode === 'processed') && container) {
                // Clean up all fullscreen modes (both, original, processed)
                if (container.classList) {
                    container.classList.remove('fullscreen-mode', 'single-video-original', 'single-video-processed');
                    console.log('Removed fullscreen-mode classes from card-body');
                    
                    // Completely reset all inline styles for cardBody
                    container.style.display = '';
                    container.style.flexDirection = '';
                    container.style.justifyContent = '';
                    container.style.alignItems = '';
                    container.style.height = '';
                    container.style.backgroundColor = '';
                    container.style.padding = '';
                    container.style.cssText = ''; // Clear any remaining styles
                    
                    // Find and reset video row styles
                    const videoRow = container.querySelector('.row');
                    if (videoRow) {
                        videoRow.style.width = '';
                        videoRow.style.maxWidth = '';
                        videoRow.style.height = '';
                        videoRow.style.margin = '';
                        videoRow.style.display = '';
                        videoRow.style.justifyContent = '';
                        videoRow.style.alignItems = '';
                        videoRow.style.cssText = ''; // Clear any remaining styles
                        
                        // Reset video column styles and make sure both are visible again
                    const videoCols = videoRow.querySelectorAll('.col-md-6');
                    videoCols.forEach(col => {
                            // Force remove any remaining inline styles
                            col.removeAttribute('style');
                            
                            // Ensure proper Bootstrap classes
                            col.className = 'col-md-6';
                    });
                    
                    // Reset video styles
                    const videos = videoRow.querySelectorAll('video');
                    videos.forEach(video => {
                            video.style.width = '';
                            video.style.height = '';
                            video.style.maxHeight = '';
                            video.style.objectFit = '';
                            video.style.cssText = ''; // Clear any remaining styles
                        });
                    
                    // Reset title/h5 styles to prevent floating labels
                    const titles = videoRow.querySelectorAll('h5');
                    titles.forEach(title => {
                        title.style.cssText = ''; // Clear all inline styles
                    });
                    
                    // Reset timeline container styles
                    const timelineContainer = videoRow.querySelector('.video-timeline-container');
                    if (timelineContainer) {
                        timelineContainer.style.position = '';
                        timelineContainer.style.bottom = '';
                        timelineContainer.style.left = '';
                        timelineContainer.style.transform = '';
                        timelineContainer.style.width = '';
                        timelineContainer.style.maxWidth = '';
                        timelineContainer.style.margin = '';
                        timelineContainer.style.zIndex = '';
                        timelineContainer.style.backgroundColor = '';
                        timelineContainer.style.borderRadius = '';
                        timelineContainer.style.padding = '';
                        timelineContainer.style.cssText = ''; // Clear any remaining styles
                    }
                    }
                }
            }
            
            // Force a re-layout for all modes
                setTimeout(() => {
                if (container) {
                    const originalDisplay = container.style.display;
                    container.style.display = 'none';
                    container.offsetHeight; // Force reflow
                    container.style.display = originalDisplay || '';
                    
                    // Additional aggressive cleanup for Bootstrap layout
                    const videoRow = container.querySelector('.row');
                    if (videoRow) {
                        // Force Bootstrap to recalculate the layout
                        videoRow.style.display = 'none';
                        videoRow.offsetHeight; // Force reflow
                        videoRow.style.display = '';
                        
                        // Ensure all columns are properly reset
                        const videoCols = videoRow.querySelectorAll('.col-md-6');
                        if (videoCols.length === 2) {
                            videoCols.forEach((col, index) => {
                                // Force remove any remaining inline styles
                                col.removeAttribute('style');
                                
                                // Ensure proper Bootstrap classes
                                col.className = 'col-md-6';
                                
                                // Make sure both columns are visible
                                col.style.display = 'block';
                                col.style.visibility = 'visible';
                                col.style.opacity = '1';
                                
                                // Force a reflow for each column
                                col.offsetHeight;
                            });
                            
                            // Force the row to use flexbox layout
                            videoRow.style.display = 'flex';
                            videoRow.style.flexWrap = 'wrap';
                            videoRow.offsetHeight;
                            videoRow.style.display = '';
                        }
                    }
                    
                    // Trigger a resize event to ensure proper layout
                    window.dispatchEvent(new Event('resize'));
                    }
                }, 10);
                
            console.log('Reset all fullscreen styles for mode:', this.fullscreenState.mode);
        }
        
        // Remove exit button
        if (this.exitFullscreenBtn) {
            this.exitFullscreenBtn.remove();
            this.exitFullscreenBtn = null;
            console.log('Removed exit fullscreen button');
        }
        
        // Update control button state if needed
        if (this.synchronizer.ui && 
            this.synchronizer.ui.controls && 
            this.synchronizer.ui.controls.exitFullscreenBtn) {
            this.synchronizer.ui.controls.exitFullscreenBtn.style.display = 'none';
            console.log('Hidden exit button in controls');
        }
        
        // Clear fullscreen state
        this.fullscreenState = null;
        
        // Clear transition flag
        this.isTransitioning = false;
        
        // Show segmentation button again
        this.showSegmentationButton();
    }
    
    /**
     * Clean up previous fullscreen mode without exiting fullscreen
     */
    cleanupPreviousMode() {
        console.log('Cleaning up previous fullscreen mode');
        
        if (this.fullscreenState && this.fullscreenState.container) {
            const container = this.fullscreenState.container;
            
            // Remove fullscreen mode classes
            container.classList.remove('fullscreen-mode', 'single-video-original', 'single-video-processed');
            
            // Reset container styles
            container.style.display = '';
            container.style.flexDirection = '';
            container.style.justifyContent = '';
            container.style.alignItems = '';
            container.style.height = '';
            container.style.backgroundColor = '';
            container.style.padding = '';
            
            // Reset video row and columns
            const videoRow = container.querySelector('.row');
            if (videoRow) {
                // Reset row styles
                videoRow.style.width = '';
                videoRow.style.maxWidth = '';
                videoRow.style.height = '';
                videoRow.style.margin = '';
                videoRow.style.display = '';
                videoRow.style.justifyContent = '';
                videoRow.style.alignItems = '';
                
                // Reset column styles
                const videoCols = videoRow.querySelectorAll('.col-md-6');
                videoCols.forEach(col => {
                    col.removeAttribute('style');
                    col.className = 'col-md-6';
                    col.style.display = 'block';
                    col.style.visibility = 'visible';
                    col.style.opacity = '1';
                });
                
                // Reset video styles
                const videos = videoRow.querySelectorAll('video');
                videos.forEach(video => {
                    video.style.width = '';
                    video.style.height = '';
                    video.style.maxHeight = '';
                    video.style.objectFit = '';
                });
                
                // Reset timeline container styles when switching modes
                const timelineContainer = videoRow.querySelector('.video-timeline-container');
                if (timelineContainer) {
                    timelineContainer.style.position = '';
                    timelineContainer.style.bottom = '';
                    timelineContainer.style.left = '';
                    timelineContainer.style.transform = '';
                    timelineContainer.style.width = '';
                    timelineContainer.style.maxWidth = '';
                    timelineContainer.style.margin = '';
                    timelineContainer.style.zIndex = '';
                    timelineContainer.style.backgroundColor = '';
                    timelineContainer.style.borderRadius = '';
                    timelineContainer.style.padding = '';
                }
            }
        }
        
        // Clear the fullscreen state but don't remove exit button or exit fullscreen
        this.fullscreenState = null;
        
        // Important: Re-hide segmentation button after cleanup to keep it hidden when switching modes
        // We're still in fullscreen, just switching between Both/Orig/Processed
        setTimeout(() => {
            this.hideSegmentationButton();
        }, 10);
    }

    /**
     * Hide the segmentation button in fullscreen modes
     */
    hideSegmentationButton() {
        // Find all possible segmentation buttons
        const segmentationBtns = document.querySelectorAll('#start-segmentation-btn, .start-segmentation-btn, button[data-action="start-segmentation"]');
        segmentationBtns.forEach(btn => {
            btn.style.display = 'none';
            btn.setAttribute('data-hidden-by-fullscreen', 'true');
        });
        if (segmentationBtns.length > 0) {
            console.log(`Hidden ${segmentationBtns.length} segmentation button(s) in fullscreen mode`);
        }
    }
    
    /**
     * Show the segmentation button when exiting fullscreen
     */
    showSegmentationButton() {
        // Only show buttons that were hidden by fullscreen
        const segmentationBtns = document.querySelectorAll('[data-hidden-by-fullscreen="true"]');
        segmentationBtns.forEach(btn => {
            btn.style.display = '';
            btn.removeAttribute('data-hidden-by-fullscreen');
        });
        if (segmentationBtns.length > 0) {
            console.log(`Showed ${segmentationBtns.length} segmentation button(s) after exiting fullscreen`);
        }
    }
    
    /**
     * Destroy and clean up resources
     */
    destroy() {
        // Clean up fullscreen if active
        this.exitFullscreenCleanup();
        this.removeFullscreenChangeListeners();
    }
}

// Make available globally
window.VideoSynchronizerFullscreen = VideoSynchronizerFullscreen; 

// Global fullscreen exit handler to catch browser-initiated exits
document.addEventListener('fullscreenchange', function() {
    if (!document.fullscreenElement) {
        // Fullscreen was exited, force cleanup of any remaining artifacts
        setTimeout(() => {
            const videoContainers = document.querySelectorAll('.card-body, .video-container, .row');
            videoContainers.forEach(container => {
                if (container.style) {
                    container.style.cssText = '';
                }
            });
            
            // Reset video columns and ensure Bootstrap classes
            const videoCols = document.querySelectorAll('.col-md-6');
            videoCols.forEach(col => {
                // Force remove any remaining inline styles
                col.removeAttribute('style');
                
                // Ensure proper Bootstrap classes
                col.className = 'col-md-6';
                
                // Force a reflow for each column
                col.style.display = 'none';
                col.offsetHeight;
                col.style.display = '';
            });
            
            const videos = document.querySelectorAll('video');
            videos.forEach(video => {
                if (video.style) {
                    video.style.cssText = '';
                }
            });
            
            // Force reflow
            document.body.style.display = 'none';
            document.body.offsetHeight;
            document.body.style.display = '';
            
            window.dispatchEvent(new Event('resize'));
        }, 50);
    }
});

// Also handle webkit fullscreen changes
document.addEventListener('webkitfullscreenchange', function() {
    if (!document.webkitFullscreenElement) {
        setTimeout(() => {
            const videoContainers = document.querySelectorAll('.card-body, .video-container, .row');
            videoContainers.forEach(container => {
                if (container.style) {
                    container.style.cssText = '';
                }
            });
            
            // Reset video columns and ensure Bootstrap classes
            const videoCols = document.querySelectorAll('.col-md-6');
            videoCols.forEach(col => {
                // Force remove any remaining inline styles
                col.removeAttribute('style');
                
                // Ensure proper Bootstrap classes
                col.className = 'col-md-6';
                
                // Force a reflow for each column
                col.style.display = 'none';
                col.offsetHeight;
                col.style.display = '';
            });
            
            const videos = document.querySelectorAll('video');
            videos.forEach(video => {
                if (video.style) {
                    video.style.cssText = '';
                }
            });
            
            document.body.style.display = 'none';
            document.body.offsetHeight;
            document.body.style.display = '';
            
            window.dispatchEvent(new Event('resize'));
        }, 50);
    }
}); 