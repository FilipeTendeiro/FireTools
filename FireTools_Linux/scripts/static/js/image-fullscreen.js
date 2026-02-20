/**
 * Image Fullscreen Modal
 * Provides fullscreen viewing with toggle between original and processed images
 */

class ImageFullscreenViewer {
    constructor() {
        this.modal = null;
        this.originalImg = null;
        this.processedImg = null;
        this.currentView = 'processed'; // 'original' or 'processed'
        this.originalUrl = '';
        this.processedUrl = '';
        this.label = '';
        this.firstLabel = 'Original Image'; // Label for first image (original/overlay)
        this.secondLabel = 'Processed Image'; // Label for second image (processed/mask)
        
        this.initModal();
        this.attachEventListeners();
    }
    
    initModal() {
        // Check if modal already exists
        if (document.getElementById('imageFullscreenModal')) {
            this.modal = document.getElementById('imageFullscreenModal');
            return;
        }
        
        // Create modal HTML
        const modalHTML = `
            <div class="modal fade" id="imageFullscreenModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-fullscreen">
                    <div class="modal-content bg-dark">
                        <div class="modal-header border-0">
                            <h5 class="modal-title text-white" id="imageModalLabel">Image Viewer</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body d-flex flex-column align-items-center justify-content-center p-0">
                            <!-- Image Container -->
                            <div class="position-relative w-100 h-100 d-flex align-items-center justify-content-center">
                                <img id="fullscreenImage" 
                                     class="img-fluid" 
                                     style="height: 75vh; max-width: 95vw; object-fit: contain;"
                                     alt="Fullscreen view">
                            </div>
                            
                            <!-- Control Bar -->
                            <div class="position-absolute bottom-0 w-100 bg-dark bg-opacity-75 p-3">
                                <div class="container">
                                    <div class="row align-items-center">
                                        <div class="col-md-4 text-start">
                                            <span class="text-white" id="currentViewLabel">
                                                <i class="bi bi-image"></i> Processed Image
                                            </span>
                                        </div>
                                        <div class="col-md-4 text-center">
                                            <button id="toggleImageBtn" class="btn btn-primary">
                                                <i class="bi bi-arrow-left-right"></i> Toggle to Original
                                            </button>
                                        </div>
                                        <div class="col-md-4 text-end">
                                            <button class="btn btn-secondary me-2" data-bs-dismiss="modal">
                                                <i class="bi bi-x-lg"></i> Close
                                            </button>
                                            <button id="downloadImageBtn" class="btn btn-success">
                                                <i class="bi bi-download"></i> Download
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to document
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.modal = document.getElementById('imageFullscreenModal');
    }
    
    attachEventListeners() {
        // Toggle button
        const toggleBtn = document.getElementById('toggleImageBtn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleImage());
        }
        
        // Download button
        const downloadBtn = document.getElementById('downloadImageBtn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadImage());
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.modal && this.modal.classList.contains('show')) {
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === ' ') {
                    e.preventDefault();
                    this.toggleImage();
                } else if (e.key === 'Escape') {
                    this.close();
                }
            }
        });
        
        // Reset view when modal is closed
        this.modal.addEventListener('hidden.bs.modal', () => {
            this.currentView = 'processed';
        });
    }
    
    open(originalUrl, processedUrl = null, label = 'Image', initialView = 'processed', firstLabel = 'Original Image', secondLabel = 'Processed Image') {
        this.originalUrl = originalUrl;
        this.processedUrl = processedUrl || originalUrl; // If no processed URL, use original for both
        this.label = label;
        this.currentView = initialView; // Use the specified initial view
        this.firstLabel = firstLabel; // Custom label for first image
        this.secondLabel = secondLabel; // Custom label for second image
        
        // Update modal title
        document.getElementById('imageModalLabel').textContent = label;
        
        // Hide/show toggle button based on whether we have two different images
        const toggleBtn = document.getElementById('toggleImageBtn');
        if (originalUrl === this.processedUrl || !processedUrl) {
            toggleBtn.style.display = 'none';
        } else {
            toggleBtn.style.display = 'inline-block';
        }
        
        // Show the specified initial view
        this.updateImage();
        
        // Open modal using Bootstrap
        const bsModal = new bootstrap.Modal(this.modal);
        bsModal.show();
    }
    
    toggleImage() {
        this.currentView = this.currentView === 'processed' ? 'original' : 'processed';
        this.updateImage();
    }
    
    updateImage() {
        const img = document.getElementById('fullscreenImage');
        const label = document.getElementById('currentViewLabel');
        const toggleBtn = document.getElementById('toggleImageBtn');
        
        if (this.currentView === 'processed') {
            img.src = this.processedUrl;
            img.alt = this.secondLabel;
            label.innerHTML = `<i class="bi bi-image"></i> ${this.secondLabel}`;
            toggleBtn.innerHTML = `<i class="bi bi-arrow-left-right"></i> Toggle to ${this.firstLabel.replace(' Image', '')}`;
        } else {
            img.src = this.originalUrl;
            img.alt = this.firstLabel;
            label.innerHTML = `<i class="bi bi-image"></i> ${this.firstLabel}`;
            toggleBtn.innerHTML = `<i class="bi bi-arrow-left-right"></i> Toggle to ${this.secondLabel.replace(' Image', '')}`;
        }
    }
    
    downloadImage() {
        const currentUrl = this.currentView === 'processed' ? this.processedUrl : this.originalUrl;
        const filename = `${this.label}_${this.currentView}.jpg`;
        
        // Create temporary link and trigger download
        const a = document.createElement('a');
        a.href = currentUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    close() {
        const bsModal = bootstrap.Modal.getInstance(this.modal);
        if (bsModal) {
            bsModal.hide();
        }
    }
}

// Initialize global instance
window.imageFullscreenViewer = new ImageFullscreenViewer();

// Helper function to attach to gallery items
function attachFullscreenToImages() {
    // Find all View Full Size buttons/links and update them
    document.querySelectorAll('a[href][target="_blank"], a[href][download]').forEach(link => {
        const text = link.textContent.trim();
        if (text === 'View Full Size' || text.includes('View Full Size')) {
            // Convert anchor to button
            const button = document.createElement('button');
            button.className = link.className.replace('btn-sm', 'btn-sm');
            button.innerHTML = '<i class="bi bi-arrows-fullscreen"></i> View Full Size';
            
            // Determine which column this button is in
            const column = link.closest('.col-md-6');
            
            // Store URLs from parent structure
            const card = link.closest('.card-body') || link.closest('.card');
            if (card) {
                const imgs = card.querySelectorAll('img');
                const headerEl = card.closest('.card')?.querySelector('.card-header span');
                const label = headerEl?.textContent.trim() || 'Image';
                
                if (imgs.length >= 2) {
                    // Paired images (original + processed or overlay + mask)
                    const originalImg = imgs[0];
                    const processedImg = imgs[1];
                    
                    // Determine which image this button is associated with
                    // by checking if the button's column contains the first or second image
                    const isFirstColumn = column && column.contains(originalImg);
                    const initialView = isFirstColumn ? 'original' : 'processed';
                    
                    // Detect labels based on h5 headers in the columns
                    const h5Elements = card.querySelectorAll('h5');
                    let firstLabel = 'Original Image';
                    let secondLabel = 'Processed Image (with Detection)';
                    
                    if (h5Elements.length >= 2) {
                        const firstHeader = h5Elements[0].textContent.trim();
                        const secondHeader = h5Elements[1].textContent.trim();
                        
                        // Check if this is segmentation gallery (has "Overlay" and "Mask" labels)
                        if (firstHeader === 'Overlay' || firstHeader.includes('Overlay')) {
                            firstLabel = 'Overlay Image';
                            secondLabel = 'Mask Image';
                        }
                    }
                    
                    button.addEventListener('click', (e) => {
                        e.preventDefault();
                        window.imageFullscreenViewer.open(originalImg.src, processedImg.src, label, initialView, firstLabel, secondLabel);
                    });
                } else if (imgs.length === 1) {
                    // Single image
                    const img = imgs[0];
                    
                    button.addEventListener('click', (e) => {
                        e.preventDefault();
                        window.imageFullscreenViewer.open(img.src, null, label, 'original');
                    });
                }
            }
            
            link.replaceWith(button);
        }
    });
    
    // Frame images: No click handler - only use the View Full Size button
}

// Auto-attach when document loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(attachFullscreenToImages, 500);
    });
} else {
    setTimeout(attachFullscreenToImages, 500);
}

