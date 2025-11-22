/**
 * DataScope - Interactive Grid Background & Modern Interactions
 * Advanced JavaScript for professional user experience
 */

class DBVisualizerApp {
    constructor() {
        this.gridCanvas = null;
        this.gridCtx = null;
        this.gridDots = [];
        this.animationId = null;
        this.mouse = { x: 0, y: 0 };
        this.ripples = [];
        this.particles = [];
        
        this.config = {
            grid: {
                spacing: 50,
                dotSize: 1,
                maxActiveDistance: 150,
                rippleSpeed: 3,
                maxRippleSize: 200,
                particleCount: 30
            },
            colors: {
                gridDefault: 'rgba(169, 162, 156, 0.2)',
                gridActive: 'rgba(213, 204, 199, 0.6)',
                ripple: 'rgba(213, 204, 199, 0.4)',
                particle: 'rgba(213, 204, 199, 0.3)'
            }
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeGrid();
        this.initializeNavigation();
        this.initializeCards();
        this.initializeButtons();
        this.initializeForms();
        this.startAnimationLoop();
    }
    
    setupEventListeners() {
        // Check if DOM is already loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.onDOMContentLoaded();
            });
        } else {
            // DOM is already loaded
            this.onDOMContentLoaded();
        }
        
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
        
        window.addEventListener('scroll', () => {
            this.onWindowScroll();
        });
        
        document.addEventListener('mousemove', (e) => {
            this.onMouseMove(e);
        });
        
        document.addEventListener('click', (e) => {
            this.onMouseClick(e);
        });
    }
    
    onDOMContentLoaded() {
        this.initializeGrid();
        this.initializeIntersectionObserver();
        this.initializeParticles();
    }
    
    onWindowResize() {
        this.resizeGrid();
        this.throttle(() => {
            this.updateGridDots();
        }, 250)();
    }
    
    onWindowScroll() {
        this.updateNavbar();
        this.createScrollRipples();
    }
    
    onMouseMove(e) {
        this.mouse.x = e.clientX;
        this.mouse.y = e.clientY;
        this.activateNearbyDots();
        this.updateParticles();
    }
    
    onMouseClick(e) {
        this.createRipple(e.clientX, e.clientY);
        this.createParticleBurst(e.clientX, e.clientY);
    }
    
    // Advanced Grid Background System
    initializeGrid() {
        console.log('Initializing grid background...');
        
        // Create grid background container
        let gridContainer = document.querySelector('.grid-background');
        if (!gridContainer) {
            gridContainer = document.createElement('div');
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
            document.body.appendChild(gridContainer);
            console.log('Grid container created');
        }
        
        // Create canvas
        if (!this.gridCanvas) {
            this.gridCanvas = document.createElement('canvas');
            this.gridCanvas.className = 'grid-canvas';
            this.gridCanvas.style.cssText = `
                width: 100%;
                height: 100%;
                display: block;
                opacity: 0.8;
            `;
            this.gridCtx = this.gridCanvas.getContext('2d');
            gridContainer.appendChild(this.gridCanvas);
            console.log('Grid canvas created');
        }
        
        this.resizeGrid();
        this.updateGridDots();
        console.log('Grid initialized successfully');
    }
    
    resizeGrid() {
        if (!this.gridCanvas || !this.gridCtx) return;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Set canvas size
        this.gridCanvas.width = width;
        this.gridCanvas.height = height;
        this.gridCanvas.style.width = width + 'px';
        this.gridCanvas.style.height = height + 'px';
        
        console.log(`Grid resized to: ${width}x${height}`);
    }
    
    updateGridDots() {
        if (!this.gridCanvas) return;
        
        this.gridDots = [];
        const spacing = this.config.grid.spacing;
        const cols = Math.ceil(this.gridCanvas.width / spacing) + 1;
        const rows = Math.ceil(this.gridCanvas.height / spacing) + 1;
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                this.gridDots.push({
                    x: col * spacing,
                    y: row * spacing,
                    opacity: 0.3,
                    targetOpacity: 0.3,
                    active: false,
                    pulse: 0,
                    baseSize: this.config.grid.dotSize,
                    currentSize: this.config.grid.dotSize,
                    lastActivation: 0
                });
            }
        }
    }
    
    activateNearbyDots() {
        const maxDistance = this.config.grid.maxActiveDistance;
        const currentTime = Date.now();
        
        this.gridDots.forEach(dot => {
            const distance = Math.sqrt(
                Math.pow(dot.x - this.mouse.x, 2) + 
                Math.pow(dot.y - this.mouse.y, 2)
            );
            
            if (distance < maxDistance) {
                const intensity = 1 - (distance / maxDistance);
                dot.targetOpacity = Math.max(0.3, 0.8 * intensity);
                dot.active = true;
                dot.lastActivation = currentTime;
                dot.currentSize = dot.baseSize * (1 + intensity * 2);
            } else if (currentTime - dot.lastActivation > 2000) {
                dot.active = false;
                dot.targetOpacity = 0.3;
                dot.currentSize = dot.baseSize;
            }
        });
    }
    
    createRipple(x, y) {
        this.ripples.push({
            x: x,
            y: y,
            radius: 0,
            maxRadius: this.config.grid.maxRippleSize,
            opacity: 1,
            speed: this.config.grid.rippleSpeed,
            created: Date.now()
        });
    }
    
    createScrollRipples() {
        const scrollPercent = window.pageYOffset / (document.documentElement.scrollHeight - window.innerHeight);
        const y = scrollPercent * window.innerHeight;
        
        // Create ripples along the scroll position
        if (Math.random() < 0.1) { // 10% chance per scroll event
            const x = Math.random() * window.innerWidth;
            this.createRipple(x, y);
        }
    }
    
    // Particle System
    initializeParticles() {
        for (let i = 0; i < this.config.grid.particleCount; i++) {
            this.particles.push({
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                opacity: Math.random() * 0.3,
                size: Math.random() * 2 + 1,
                life: 1
            });
        }
    }
    
    updateParticles() {
        this.particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Attract to mouse
            const dx = this.mouse.x - particle.x;
            const dy = this.mouse.y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 100) {
                const force = 0.01;
                particle.vx += dx * force / distance;
                particle.vy += dy * force / distance;
            }
            
            // Boundary wrapping
            if (particle.x < 0) particle.x = window.innerWidth;
            if (particle.x > window.innerWidth) particle.x = 0;
            if (particle.y < 0) particle.y = window.innerHeight;
            if (particle.y > window.innerHeight) particle.y = 0;
            
            // Damping
            particle.vx *= 0.99;
            particle.vy *= 0.99;
        });
    }
    
    createParticleBurst(x, y) {
        for (let i = 0; i < 8; i++) {
            const angle = (Math.PI * 2 / 8) * i;
            const speed = Math.random() * 2 + 1;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                opacity: 0.8,
                size: Math.random() * 3 + 2,
                life: 1,
                decay: 0.02
            });
        }
    }
    
    // Animation Loop
    startAnimationLoop() {
        const animate = () => {
            this.renderGrid();
            this.renderRipples();
            this.renderParticles();
            this.updateRipples();
            this.updateGridAnimations();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    renderGrid() {
        if (!this.gridCtx || !this.gridCanvas) return;
        
        // Clear canvas
        this.gridCtx.clearRect(0, 0, this.gridCanvas.width, this.gridCanvas.height);
        
        // Draw grid dots
        this.gridDots.forEach(dot => {
            // Smooth opacity transition
            dot.opacity += (dot.targetOpacity - dot.opacity) * 0.1;
            
            // Pulse animation for active dots
            if (dot.active) {
                dot.pulse += 0.1;
                const pulseIntensity = Math.sin(dot.pulse) * 0.3;
                dot.opacity = dot.targetOpacity + pulseIntensity;
                dot.currentSize = dot.baseSize * (1 + Math.abs(pulseIntensity));
            }
            
            // Draw dot
            const opacity = Math.max(0, Math.min(1, dot.opacity));
            this.gridCtx.fillStyle = `rgba(169, 162, 156, ${opacity})`;
            this.gridCtx.beginPath();
            this.gridCtx.arc(dot.x, dot.y, dot.currentSize, 0, Math.PI * 2);
            this.gridCtx.fill();
            
            // Draw connections between nearby active dots
            if (dot.active && dot.opacity > 0.5) {
                this.drawConnections(dot);
            }
        });
    }
    
    drawConnections(dot) {
        const maxConnectionDistance = this.config.grid.spacing * 2;
        
        this.gridDots.forEach(otherDot => {
            if (otherDot === dot || !otherDot.active || otherDot.opacity <= 0.5) return;
            
            const distance = Math.sqrt(
                Math.pow(dot.x - otherDot.x, 2) + 
                Math.pow(dot.y - otherDot.y, 2)
            );
            
            if (distance < maxConnectionDistance) {
                const lineOpacity = (1 - distance / maxConnectionDistance) * 0.3;
                this.gridCtx.strokeStyle = `rgba(213, 204, 199, ${lineOpacity})`;
                this.gridCtx.lineWidth = 0.5;
                this.gridCtx.beginPath();
                this.gridCtx.moveTo(dot.x, dot.y);
                this.gridCtx.lineTo(otherDot.x, otherDot.y);
                this.gridCtx.stroke();
            }
        });
    }
    
    renderRipples() {
        this.ripples.forEach(ripple => {
            const opacity = ripple.opacity * (1 - ripple.radius / ripple.maxRadius);
            
            this.gridCtx.strokeStyle = `rgba(213, 204, 199, ${opacity})`;
            this.gridCtx.lineWidth = 2;
            this.gridCtx.beginPath();
            this.gridCtx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
            this.gridCtx.stroke();
        });
    }
    
    renderParticles() {
        this.particles.forEach(particle => {
            if (particle.life <= 0) return;
            
            this.gridCtx.fillStyle = `rgba(213, 204, 199, ${particle.opacity * particle.life})`;
            this.gridCtx.beginPath();
            this.gridCtx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.gridCtx.fill();
        });
    }
    
    updateRipples() {
        this.ripples = this.ripples.filter(ripple => {
            ripple.radius += ripple.speed;
            ripple.opacity *= 0.98;
            return ripple.radius < ripple.maxRadius && ripple.opacity > 0.01;
        });
    }
    
    updateGridAnimations() {
        // Random activation for ambient effect
        if (Math.random() < 0.01) { // 1% chance per frame
            const randomDot = this.gridDots[Math.floor(Math.random() * this.gridDots.length)];
            if (randomDot) {
                randomDot.active = true;
                randomDot.targetOpacity = 0.6;
                randomDot.lastActivation = Date.now();
                
                setTimeout(() => {
                    randomDot.active = false;
                    randomDot.targetOpacity = 0.3;
                }, 2000);
            }
        }
        
        // Update particle lifetimes
        this.particles.forEach(particle => {
            if (particle.decay) {
                particle.life -= particle.decay;
                if (particle.life <= 0) {
                    // Respawn particle
                    particle.x = Math.random() * window.innerWidth;
                    particle.y = Math.random() * window.innerHeight;
                    particle.vx = (Math.random() - 0.5) * 0.5;
                    particle.vy = (Math.random() - 0.5) * 0.5;
                    particle.life = 1;
                    particle.opacity = Math.random() * 0.3;
                    delete particle.decay;
                }
            }
        });
    }
    
    // Navigation Enhancement
    initializeNavigation() {
        const navbar = document.querySelector('.navbar');
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                this.animateNavToggle(navToggle, navMenu.classList.contains('active'));
            });
            
            // Close menu on link click (mobile)
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', () => {
                    navMenu.classList.remove('active');
                    this.animateNavToggle(navToggle, false);
                });
            });
        }
        
        // Add active states to nav links
        this.updateActiveNavLink();
    }
    
    updateNavbar() {
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        }
    }
    
    updateActiveNavLink() {
        const navLinks = document.querySelectorAll('.nav-link');
        const currentPath = window.location.pathname;
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }
    
    animateNavToggle(toggle, isActive) {
        const icon = toggle.querySelector('i');
        if (icon) {
            icon.style.transform = isActive ? 'rotate(90deg)' : 'rotate(0deg)';
        }
    }
    
    // Card Interactions
    initializeCards() {
        const cards = document.querySelectorAll('.card, .feature-card, .step-card');
        
        cards.forEach(card => {
            this.addCardParallax(card);
            this.addCardIntersectionObserver(card);
        });
    }
    
    addCardParallax(card) {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    }
    
    addCardIntersectionObserver(card) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animationDelay = `${Math.random() * 0.5}s`;
                    entry.target.classList.add('animate-fade-in-up');
                }
            });
        }, { threshold: 0.1 });
        
        observer.observe(card);
    }
    
    // Button Enhancements
    initializeButtons() {
        const buttons = document.querySelectorAll('.btn');
        
        buttons.forEach(button => {
            this.addButtonRipple(button);
            this.addButtonMagnetic(button);
        });
    }
    
    addButtonRipple(button) {
        button.addEventListener('click', (e) => {
            const ripple = document.createElement('div');
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                left: ${x}px;
                top: ${y}px;
                width: ${size}px;
                height: ${size}px;
                background: rgba(213, 204, 199, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: buttonRipple 0.6s ease-out;
                pointer-events: none;
                z-index: 1;
            `;
            
            button.style.position = 'relative';
            button.style.overflow = 'hidden';
            button.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    }
    
    addButtonMagnetic(button) {
        button.addEventListener('mousemove', (e) => {
            const rect = button.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            
            const strength = 0.2;
            button.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = '';
        });
    }
    
    // Form Improvements
    initializeForms() {
        const forms = document.querySelectorAll('form');
        const uploadAreas = document.querySelectorAll('.upload-area');
        
        forms.forEach(form => {
            this.addFormValidation(form);
        });
        
        uploadAreas.forEach(area => {
            this.addDragAndDrop(area);
        });
    }
    
    addFormValidation(form) {
        const inputs = form.querySelectorAll('.form-control');
        
        inputs.forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });
            
            input.addEventListener('blur', () => {
                input.parentElement.classList.remove('focused');
                if (input.value) {
                    input.parentElement.classList.add('filled');
                } else {
                    input.parentElement.classList.remove('filled');
                }
            });
        });
    }
    
    addDragAndDrop(uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0], uploadArea);
            }
        });
    }
    
    handleFileUpload(file, uploadArea) {
        const fileInput = uploadArea.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.files = file;
            
            // Visual feedback
            const uploadText = uploadArea.querySelector('.upload-text');
            if (uploadText) {
                uploadText.innerHTML = `
                    <h3>File Selected</h3>
                    <p>${file.name}</p>
                    <small>Ready to upload</small>
                `;
            }
        }
    }
    
    // Intersection Observer for animations
    initializeIntersectionObserver() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in-up');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);
        
        // Observe elements that should animate on scroll
        document.querySelectorAll('.section-title, .section-subtitle, .feature-card, .step-card, .hero-content, .hero-visual')
            .forEach(el => observer.observe(el));
    }
    
    // Utility Functions
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    debounce(func, wait, immediate) {
        let timeout;
        return function() {
            const context = this;
            const args = arguments;
            const later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    }
}

// CSS Animation Keyframes
const styleElement = document.createElement('style');
styleElement.textContent = `
    @keyframes buttonRipple {
        to {
            transform: scale(2);
            opacity: 0;
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in-up {
        animation: fadeInUp 0.8s ease-out forwards;
    }
    
    .form-group.focused .form-label {
        color: var(--text-primary);
        transform: translateY(-2px);
    }
    
    .form-group.filled .form-label {
        color: var(--text-secondary);
    }
`;
document.head.appendChild(styleElement);

// Simple initialization function
function initializeApp() {
    console.log('Starting DataScope App initialization...');
    try {
        const app = new DBVisualizerApp();
        window.DBVisualizerApp = app;
        console.log('DataScope App initialized successfully');
    } catch (error) {
        console.error('Error initializing DataScope App:', error);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM is already loaded
    setTimeout(initializeApp, 100);
}

// Also try to initialize after a short delay
setTimeout(() => {
    if (!window.DBVisualizerApp) {
        console.log('Attempting delayed initialization...');
        initializeApp();
    }
}, 500);

// Export class for potential external use
window.DBVisualizerAppClass = DBVisualizerApp;