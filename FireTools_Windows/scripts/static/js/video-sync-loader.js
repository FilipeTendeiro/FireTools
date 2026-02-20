/**
 * Video Synchronization Loader
 * 
 * This file orchestrates loading all the required video synchronization 
 * modules and initializes the system.
 */

// Define paths to modules
const MODULE_PATHS = {
    core: '/static/js/video-sync/core.js',
    ui: '/static/js/video-sync/ui-controls.js',
    fullscreen: '/static/js/video-sync/fullscreen.js',
    detection: '/static/js/video-sync/detection.js',
    timeline: '/static/js/video-sync/timeline.js',
    utils: '/static/js/video-sync/utils.js',
};

// Track loaded state
let isLoaded = false;
let activeCleanupFunction = null;

// Main setup function (backward compatible with existing code)
function setupVideoSync() {
    console.log("Setting up video synchronization (modular version)");
    
    // If modules aren't loaded yet, start loading
    if (!window.VideoSynchronizerCore) {
        console.log("Video sync modules not loaded yet, loading dynamically");
        loadModulesAndSetup();
        return function() {
            console.log("Cleanup requested but setup not complete");
            // Will be replaced with real cleanup function when modules are loaded
        };
    }
    
    // When all modules are loaded, this code will run
    return initializeVideoSync();
}

// Load all modules dynamically
function loadModulesAndSetup() {
    if (isLoaded) return;
    
    // Load the modular files
    let loadedCount = 0;
    const totalModules = Object.keys(MODULE_PATHS).length;
    
    // Load each module
    Object.entries(MODULE_PATHS).forEach(([key, path]) => {
        loadScript(path, () => {
            console.log(`Loaded ${key} module`);
            loadedCount++;
            
            // When all modules are loaded, initialize
            if (loadedCount === totalModules) {
                console.log('All modules loaded, initializing video sync');
                isLoaded = true;
                
                // Once loaded, attempt to setup if there are videos on the page
                activeCleanupFunction = initializeVideoSync();
            }
        });
    });
}

// Initialize synchronization once modules are loaded
function initializeVideoSync() {
    console.log("Initializing video synchronization");
    
    // Create the composite VideoSynchronizer class that combines all modules
    class VideoSynchronizer {
        constructor(videoA, videoB, pairId, options = {}) {
            // Store options
            this.options = options;
            
            // Initialize core functionality
            this.core = new VideoSynchronizerCore(videoA, videoB, pairId);
            
            // Copy core properties to this instance for backward compatibility
            this.videoA = this.core.videoA;
            this.videoB = this.core.videoB;
            this.pairId = this.core.pairId;
            this.masterVideo = this.core.masterVideo;
            this.slaveVideo = this.core.slaveVideo;
            this.destroyed = this.core.destroyed;
            
            // Initialize fullscreen functionality
            this.fullscreen = new VideoSynchronizerFullscreen(this);
            
            // Initialize timeline first (needs to be before UI controls)
            this.timeline = new VideoSynchronizerTimeline(this, options);
            
            // Initialize UI controls after timeline
            this.ui = new VideoSynchronizerUI(this);
            this.ui.createControls();
            
            // Initialize detection functionality after UI is ready (with options)
            this.detection = new VideoSynchronizerDetection(this, options);
            
            // Bind fullscreen methods for easy access
            this.enterFullscreen = this.fullscreen.enterFullscreen.bind(this.fullscreen);
            this.exitFullscreen = this.fullscreen.exitFullscreen.bind(this.fullscreen);
            
            // Bind detection methods for easy access
            this.loadDetectionMetadata = this.detection.loadDetectionMetadata.bind(this.detection);
            this.toggleAutoSpeed = this.detection.toggleAutoSpeed.bind(this.detection);
            this.disableAutoSpeed = this.detection.disableAutoSpeed.bind(this.detection);
            this.getDetectionStatus = this.detection.getDetectionStatus.bind(this.detection);
            
            // Bind core methods for easy access
            this.handlePlayPauseClick = this.core.handlePlayPauseClick.bind(this.core);
            this.seekRelative = this.core.seekRelative.bind(this.core);
            this.seekTo = this.core.seekTo.bind(this.core);
            this.setPlaybackSpeed = this.core.setPlaybackSpeed.bind(this.core);
            this.playVideos = this.core.playVideos.bind(this.core);
            this.pauseVideos = this.core.pauseVideos.bind(this.core);
            
            console.log(`Composite VideoSynchronizer created for pair ${pairId}`);
        }
        
        destroy() {
            if (this.detection) {
                this.detection.destroy();
            }
            if (this.timeline) {
                this.timeline.destroy();
            }
            if (this.fullscreen) {
                this.fullscreen.destroy();
            }
            if (this.ui) {
                this.ui.destroy();
            }
            if (this.core) {
                this.core.destroyCore();
            }
            this.destroyed = true;
        }
    }
    
    // Make VideoSynchronizer available globally
    window.VideoSynchronizer = VideoSynchronizer;
    
    // Store active synchronizers for cleanup
    const activeSynchronizers = [];
    
    // Find all sync-video elements and group by pair index
    const videoPairs = {};
    
    document.querySelectorAll('.sync-video').forEach(video => {
        const pairIndex = video.dataset.pairIndex;
        if (!pairIndex) return;
        
        if (!videoPairs[pairIndex]) {
            videoPairs[pairIndex] = [];
        }
        
        // Only add if not already in array
        if (!videoPairs[pairIndex].includes(video)) {
            videoPairs[pairIndex].push(video);
            
            // When we have a pair, create a synchronizer
            if (videoPairs[pairIndex].length === 2) {
                const synchronizer = new VideoSynchronizer(
                    videoPairs[pairIndex][0], 
                    videoPairs[pairIndex][1],
                    pairIndex
                );
                activeSynchronizers.push(synchronizer);
                
                // Hide the native video controls when using our custom controls
                videoPairs[pairIndex][0].controls = false;
                videoPairs[pairIndex][1].controls = false;
            }
        }
    });
    
    // Return cleanup function
    return function cleanup() {
        activeSynchronizers.forEach(sync => sync.destroy());
        console.log("Video sync cleanup complete");
    };
}

// Helper to load a script dynamically
function loadScript(src, callback) {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = callback;
    script.onerror = (error) => {
        console.error(`Failed to load script: ${src}`, error);
    };
    document.head.appendChild(script);
}

// Automatically load modules when this script is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadModulesAndSetup();
}); 