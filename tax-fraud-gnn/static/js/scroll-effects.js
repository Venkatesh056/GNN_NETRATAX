/* ============================================================================
   Scroll Effects & Interactions
   ============================================================================ */

document.addEventListener('DOMContentLoaded', function() {
    initScrollProgress();
    initNavbarScroll();
    initParallaxScroll();
    initCounterOnScroll();
});

/* ============================================================================
   Scroll Progress Bar
   ============================================================================ */
function initScrollProgress() {
    // Create progress bar element
    const progressBar = document.createElement('div');
    progressBar.classList.add('scroll-progress');
    document.body.appendChild(progressBar);
    
    // Update progress on scroll
    window.addEventListener('scroll', () => {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    });
}

/* ============================================================================
   Navbar Scroll Effect
   ============================================================================ */
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    
    let lastScroll = 0;
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.scrollY;
        
        // Add scrolled class when scrolled down
        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        // Hide navbar on scroll down, show on scroll up
        if (currentScroll > lastScroll && currentScroll > 100) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScroll = currentScroll;
    });
}

/* ============================================================================
   Parallax Scroll Effect
   ============================================================================ */
function initParallaxScroll() {
    const parallaxElements = document.querySelectorAll('.hero-background, .floating-shape');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;
        
        parallaxElements.forEach((el, index) => {
            const speed = (index + 1) * 0.5;
            el.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
}

/* ============================================================================
   Counter Animation on Scroll
   ============================================================================ */
function initCounterOnScroll() {
    const counters = document.querySelectorAll('.stat-value, .metric-value');
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                animateCounter(entry.target);
                entry.target.classList.add('counted');
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => counterObserver.observe(counter));
}

function animateCounter(element) {
    const text = element.textContent;
    const number = parseFloat(text.replace(/[^0-9.]/g, ''));
    
    if (isNaN(number)) return;
    
    const duration = 2000;
    const steps = 60;
    const increment = number / steps;
    let current = 0;
    let step = 0;
    
    const timer = setInterval(() => {
        current += increment;
        step++;
        
        if (step >= steps) {
            element.textContent = text;
            clearInterval(timer);
        } else {
            // Preserve any text formatting
            const formatted = text.includes('%') 
                ? current.toFixed(1) + '%'
                : text.includes(',')
                ? Math.floor(current).toLocaleString()
                : Math.floor(current).toString();
            
            element.textContent = formatted;
        }
    }, duration / steps);
}

/* ============================================================================
   Smooth Anchor Scrolling
   ============================================================================ */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#' || href === '#!') return;
        
        e.preventDefault();
        const target = document.querySelector(href);
        
        if (target) {
            const offsetTop = target.offsetTop - 80; // Account for fixed navbar
            
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

/* ============================================================================
   Lazy Load Images
   ============================================================================ */
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.classList.add('loaded');
                    imageObserver.unobserve(img);
                }
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

/* ============================================================================
   Scroll Reveal with Stagger
   ============================================================================ */
function initScrollReveal() {
    const revealElements = document.querySelectorAll('.reveal, .stagger-item');
    
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('active');
                }, index * 100);
                revealObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    });
    
    revealElements.forEach(el => revealObserver.observe(el));
}

initScrollReveal();

/* ============================================================================
   Back to Top Button
   ============================================================================ */
function createBackToTop() {
    const button = document.createElement('button');
    button.classList.add('back-to-top');
    button.innerHTML = '↑';
    button.setAttribute('aria-label', 'Back to top');
    
    button.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: #700034;
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 24px;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 999;
        box-shadow: 0 4px 12px rgba(112, 0, 52, 0.3);
    `;
    
    button.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    document.body.appendChild(button);
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            button.style.opacity = '1';
            button.style.visibility = 'visible';
        } else {
            button.style.opacity = '0';
            button.style.visibility = 'hidden';
        }
    });
    
    button.addEventListener('mouseenter', () => {
        button.style.transform = 'scale(1.1)';
        button.style.boxShadow = '0 6px 20px rgba(112, 0, 52, 0.4)';
    });
    
    button.addEventListener('mouseleave', () => {
        button.style.transform = 'scale(1)';
        button.style.boxShadow = '0 4px 12px rgba(112, 0, 52, 0.3)';
    });
}

createBackToTop();

/* ============================================================================
   Mouse Trail Effect (Subtle)
   ============================================================================ */
function initMouseTrail() {
    const trail = [];
    const trailLength = 10;
    
    document.addEventListener('mousemove', (e) => {
        trail.push({ x: e.clientX, y: e.clientY, time: Date.now() });
        
        if (trail.length > trailLength) {
            trail.shift();
        }
        
        // Clean up old trail elements
        document.querySelectorAll('.mouse-trail').forEach(el => {
            if (Date.now() - parseInt(el.dataset.time) > 1000) {
                el.remove();
            }
        });
    });
}

// Uncomment to enable mouse trail
// initMouseTrail();

/* ============================================================================
   Export for Global Use
   ============================================================================ */
window.ScrollEffects = {
    animateCounter,
    initScrollReveal
};
