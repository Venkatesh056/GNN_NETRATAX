/* ============================================================================
   UI Enhancements - Interactive Effects & Animations
   ============================================================================ */

// Initialize all enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initRippleEffect();
    initScrollReveal();
    initParallax();
    initNumberCounters();
    initSmoothScroll();
    initTooltips();
    initCardHoverEffects();
    initLoadingStates();
    initNotifications();
    initProgressBars();
});

/* ============================================================================
   Ripple Effect on Buttons
   ============================================================================ */
function initRippleEffect() {
    const buttons = document.querySelectorAll('.btn, .nav-link, .metric-card');
    
    buttons.forEach(button => {
        button.classList.add('ripple-container');
        
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

/* ============================================================================
   Scroll Reveal Animation
   ============================================================================ */
function initScrollReveal() {
    const reveals = document.querySelectorAll('.metric-card, .chart-container, .stat-card, .feature-card');
    
    reveals.forEach(el => el.classList.add('reveal'));
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    reveals.forEach(el => observer.observe(el));
}

/* ============================================================================
   Parallax Effect on Mouse Move
   ============================================================================ */
function initParallax() {
    const parallaxElements = document.querySelectorAll('.metric-card, .chart-container');
    
    document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;
        
        parallaxElements.forEach((el, index) => {
            const speed = (index + 1) * 0.5;
            const x = (mouseX - 0.5) * speed;
            const y = (mouseY - 0.5) * speed;
            
            el.style.transform = `translate(${x}px, ${y}px)`;
        });
    });
}

/* ============================================================================
   Animated Number Counters
   ============================================================================ */
function initNumberCounters() {
    const counters = document.querySelectorAll('.metric-value, .stat-value');
    
    const animateCounter = (element) => {
        const target = parseInt(element.textContent.replace(/,/g, ''));
        if (isNaN(target)) return;
        
        const duration = 2000;
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target.toLocaleString();
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current).toLocaleString();
            }
        }, 16);
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => {
        counter.classList.add('counter');
        observer.observe(counter);
    });
}

/* ============================================================================
   Smooth Scroll
   ============================================================================ */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/* ============================================================================
   Enhanced Tooltips
   ============================================================================ */
function initTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(el => {
        el.classList.add('tooltip');
        
        const tooltipText = document.createElement('span');
        tooltipText.classList.add('tooltip-text');
        tooltipText.textContent = el.getAttribute('data-tooltip');
        
        el.appendChild(tooltipText);
    });
}

/* ============================================================================
   Card Hover Effects
   ============================================================================ */
function initCardHoverEffects() {
    const cards = document.querySelectorAll('.metric-card, .chart-container, .stat-card');
    
    cards.forEach(card => {
        card.classList.add('hover-lift');
        
        card.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        });
    });
}

/* ============================================================================
   Loading States
   ============================================================================ */
function initLoadingStates() {
    // Add loading spinner to buttons on click
    const actionButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
    
    actionButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (this.classList.contains('loading')) return;
            
            const originalText = this.innerHTML;
            this.classList.add('loading');
            this.disabled = true;
            
            this.innerHTML = '<div class="loading-spinner"></div>';
            
            // Simulate loading (remove this in production, let actual API calls handle it)
            setTimeout(() => {
                this.innerHTML = originalText;
                this.classList.remove('loading');
                this.disabled = false;
            }, 2000);
        });
    });
}

/* ============================================================================
   Notification System
   ============================================================================ */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.classList.add('notification-slide');
    
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    const color = type === 'success' ? '#27AE60' : type === 'error' ? '#700034' : '#2F4156';
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 40px; height: 40px; background: ${color}; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold;">
                ${icon}
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #2F4156; margin-bottom: 4px;">
                    ${type.charAt(0).toUpperCase() + type.slice(1)}
                </div>
                <div style="font-size: 14px; color: #3D5368;">
                    ${message}
                </div>
            </div>
        </div>
    `;
    
    notification.style.borderLeftColor = color;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 500);
    }, duration);
}

function initNotifications() {
    // Example: Show notification on page load
    setTimeout(() => {
        showNotification('Dashboard loaded successfully!', 'success');
    }, 1000);
}

/* ============================================================================
   Progress Bars
   ============================================================================ */
function initProgressBars() {
    const progressBars = document.querySelectorAll('[data-progress]');
    
    progressBars.forEach(bar => {
        const progress = bar.getAttribute('data-progress');
        const fill = bar.querySelector('.progress-bar-fill');
        
        if (fill) {
            setTimeout(() => {
                fill.style.width = progress + '%';
            }, 500);
        }
    });
}

/* ============================================================================
   Stagger Animation for Lists
   ============================================================================ */
function staggerAnimation(selector, delay = 100) {
    const items = document.querySelectorAll(selector);
    
    items.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            item.style.transition = 'all 0.5s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, index * delay);
    });
}

/* ============================================================================
   Floating Action Button
   ============================================================================ */
function createFAB() {
    const fab = document.createElement('div');
    fab.classList.add('fab');
    fab.innerHTML = '↑';
    fab.setAttribute('title', 'Scroll to top');
    
    fab.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    document.body.appendChild(fab);
    
    // Show/hide based on scroll position
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            fab.style.opacity = '1';
            fab.style.pointerEvents = 'all';
        } else {
            fab.style.opacity = '0';
            fab.style.pointerEvents = 'none';
        }
    });
    
    fab.style.opacity = '0';
    fab.style.pointerEvents = 'none';
}

// Initialize FAB
setTimeout(createFAB, 1000);

/* ============================================================================
   Table Row Animations
   ============================================================================ */
function animateTableRows() {
    const tables = document.querySelectorAll('.companies-table tbody');
    
    tables.forEach(tbody => {
        const rows = tbody.querySelectorAll('tr');
        
        rows.forEach((row, index) => {
            row.style.opacity = '0';
            row.style.transform = 'translateX(-20px)';
            
            setTimeout(() => {
                row.style.transition = 'all 0.4s ease';
                row.style.opacity = '1';
                row.style.transform = 'translateX(0)';
            }, index * 50);
        });
    });
}

// Call on table load
if (document.querySelector('.companies-table')) {
    setTimeout(animateTableRows, 500);
}

/* ============================================================================
   Chart Loading Animation
   ============================================================================ */
function animateCharts() {
    const charts = document.querySelectorAll('.chart-container');
    
    charts.forEach((chart, index) => {
        chart.style.opacity = '0';
        chart.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            chart.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            chart.style.opacity = '1';
            chart.style.transform = 'scale(1)';
        }, index * 200);
    });
}

// Call on page load
setTimeout(animateCharts, 300);

/* ============================================================================
   Skeleton Loading
   ============================================================================ */
function createSkeleton(container) {
    const skeleton = document.createElement('div');
    skeleton.classList.add('skeleton');
    skeleton.style.height = '100%';
    skeleton.style.minHeight = '200px';
    
    container.appendChild(skeleton);
    
    return skeleton;
}

function removeSkeleton(skeleton) {
    skeleton.style.opacity = '0';
    setTimeout(() => skeleton.remove(), 300);
}

/* ============================================================================
   Page Transition Effect
   ============================================================================ */
function initPageTransition() {
    const links = document.querySelectorAll('a:not([target="_blank"])');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            if (href && href !== '#' && !href.startsWith('#') && !href.startsWith('javascript:')) {
                e.preventDefault();
                
                document.body.style.opacity = '0';
                document.body.style.transform = 'scale(0.98)';
                document.body.style.transition = 'all 0.3s ease';
                
                setTimeout(() => {
                    window.location.href = href;
                }, 300);
            }
        });
    });
}

initPageTransition();

/* ============================================================================
   Export Functions for Global Use
   ============================================================================ */
window.UIEnhancements = {
    showNotification,
    staggerAnimation,
    createSkeleton,
    removeSkeleton,
    animateTableRows,
    animateCharts
};
