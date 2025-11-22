// Simple Grid Background - Working Version
class SimpleGridBackground {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.dots = [];
        this.mouse = { x: 0, y: 0 };
        this.ripples = [];
        
        this.init();
    }
    
    init() {
        console.log('Initializing simple grid background...');
        this.createCanvas();
        this.resize();
        this.createDots();
        this.setupEvents();
        this.animate();
        console.log('Simple grid initialized successfully');
    }
    
    createCanvas() {
        // Remove existing grid
        const existing = document.querySelector('.grid-background');
        if (existing) existing.remove();
        
        // Create grid container
        const gridContainer = document.createElement('div');
        gridContainer.className = 'grid-background';
        gridContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
        `;
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'grid-canvas';
        this.canvas.style.cssText = `
            width: 100%;
            height: 100%;
            display: block;
            opacity: 0.8;
        `;
        
        this.ctx = this.canvas.getContext('2d');
        gridContainer.appendChild(this.canvas);
        document.body.appendChild(gridContainer);
        
        console.log('Grid canvas created and added to DOM');
    }
    
    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        console.log(`Grid resized to: ${window.innerWidth}x${window.innerHeight}`);
    }
    
    createDots() {
        this.dots = [];
        const spacing = 50;
        const cols = Math.ceil(this.canvas.width / spacing);
        const rows = Math.ceil(this.canvas.height / spacing);
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                this.dots.push({
                    x: col * spacing,
                    y: row * spacing,
                    opacity: 0.3,
                    size: 1
                });
            }
        }
        console.log(`Created ${this.dots.length} dots`);
    }
    
    setupEvents() {
        document.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
        
        document.addEventListener('click', (e) => {
            this.ripples.push({
                x: e.clientX,
                y: e.clientY,
                radius: 0,
                opacity: 1
            });
        });
        
        window.addEventListener('resize', () => {
            this.resize();
            this.createDots();
        });
    }
    
    animate() {
        if (!this.ctx || !this.canvas) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw dots
        this.dots.forEach(dot => {
            const distance = Math.sqrt(
                Math.pow(dot.x - this.mouse.x, 2) + 
                Math.pow(dot.y - this.mouse.y, 2)
            );
            
            if (distance < 100) {
                dot.opacity = 0.8;
                dot.size = 2;
            } else {
                dot.opacity = Math.max(0.3, dot.opacity - 0.02);
                dot.size = Math.max(1, dot.size - 0.05);
            }
            
            this.ctx.fillStyle = `rgba(169, 162, 156, ${dot.opacity})`;
            this.ctx.beginPath();
            this.ctx.arc(dot.x, dot.y, dot.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw ripples
        this.ripples = this.ripples.filter(ripple => {
            ripple.radius += 3;
            ripple.opacity *= 0.98;
            
            if (ripple.opacity > 0.01) {
                this.ctx.strokeStyle = `rgba(213, 204, 199, ${ripple.opacity})`;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
                this.ctx.stroke();
                return true;
            }
            return false;
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// Simple Upload Handler
class SimpleUploadHandler {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.selectedFile = null;
        
        if (this.uploadArea && this.fileInput) {
            this.init();
        }
    }
    
    init() {
        console.log('Initializing simple upload handler...');
        
        // Click to upload
        this.uploadArea.addEventListener('click', (e) => {
            console.log('Upload area clicked');
            if (e.target.closest('.remove-file')) return;
            this.fileInput.click();
        });
        
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            console.log('File input changed');
            if (e.target.files && e.target.files[0]) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '#d5ccc7';
            this.uploadArea.style.backgroundColor = 'rgba(213, 204, 199, 0.1)';
        });
        
        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '';
            this.uploadArea.style.backgroundColor = '';
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = '';
            this.uploadArea.style.backgroundColor = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
        
        // Form submission
        const form = document.getElementById('uploadForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                if (!this.selectedFile) {
                    e.preventDefault();
                    alert('Please select a file first');
                    return;
                }
                
                if (this.analyzeBtn) {
                    this.analyzeBtn.innerHTML = `
                        <i class="fas fa-spinner fa-spin"></i>
                        <span>Analyzing...</span>
                    `;
                    this.analyzeBtn.disabled = true;
                }
            });
        }
        
        console.log('Simple upload handler initialized');
    }
    
    handleFileSelect(file) {
        console.log('File selected:', file.name);
        
        // Validate file type
        const allowedTypes = ['.csv', '.xlsx', '.xls'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(ext)) {
            alert('Please select a CSV or Excel file');
            return;
        }
        
        // Validate file size (50MB limit)
        if (file.size > 50 * 1024 * 1024) {
            alert('File must be less than 50MB');
            return;
        }
        
        this.selectedFile = file;
        
        // Update upload area
        this.uploadArea.classList.add('file-selected');
        const uploadContent = this.uploadArea.querySelector('.upload-content');
        if (uploadContent) {
            uploadContent.innerHTML = `
                <div class="upload-icon mb-6">
                    <i class="fas fa-check-circle" style="color: #22c55e; font-size: 4rem;"></i>
                </div>
                <div class="upload-text">
                    <h3 class="mb-3">✅ File Selected</h3>
                    <p class="mb-4"><strong>${file.name}</strong></p>
                    <p>Size: ${this.formatFileSize(file.size)} • Ready to analyze</p>
                </div>
            `;
        }
        
        // Show file info
        if (this.fileName) this.fileName.textContent = file.name;
        if (this.fileSize) this.fileSize.textContent = this.formatFileSize(file.size);
        if (this.fileInfo) this.fileInfo.style.display = 'block';
        
        // Enable analyze button
        if (this.analyzeBtn) {
            this.analyzeBtn.disabled = false;
            const shortName = file.name.length > 20 ? file.name.substring(0, 20) + '...' : file.name;
            this.analyzeBtn.innerHTML = `
                <i class="fas fa-magic"></i>
                <span>Analyze "${shortName}"</span>
            `;
        }
        
        // Update file input with the selected file
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        this.fileInput.files = dataTransfer.files;
        
        console.log('File processed successfully');
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Navigation mobile menu handler
function initMobileMenu() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }
}

// Initialize everything when DOM is ready
function initializeApp() {
    console.log('Initializing simple app...');
    
    // Initialize grid background
    new SimpleGridBackground();
    
    // Initialize upload handler
    new SimpleUploadHandler();
    
    // Initialize mobile menu
    initMobileMenu();
    
    console.log('Simple app initialized successfully');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM is already loaded
    initializeApp();
}

// Also initialize after a short delay as backup
setTimeout(() => {
    if (!document.querySelector('.grid-background')) {
        console.log('Grid not found, initializing backup...');
        new SimpleGridBackground();
    }
}, 1000);