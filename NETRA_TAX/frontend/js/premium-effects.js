/**
 * NETRA TAX - Premium UI Effects JavaScript
 * Handles dynamic animations, counter effects, scroll interactions
 * NO API modifications - pure UI/UX enhancements
 */

(function () {
    'use strict';

    // ============================================================================
    // CONFIGURATION
    // ============================================================================

    const CONFIG = {
        counterDuration: 1500,
        scrollRevealThreshold: 0.15,
        scrollTopThreshold: 300,
        debounceDelay: 10
    };

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    /**
     * Debounce function to limit execution frequency
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Check if element is in viewport
     */
    function isInViewport(element, threshold = 0) {
        const rect = element.getBoundingClientRect();
        const windowHeight = window.innerHeight || document.documentElement.clientHeight;
        return (
            rect.top <= windowHeight * (1 - threshold) &&
            rect.bottom >= 0
        );
    }

    /**
     * Easing function for smooth animations
     */
    function easeOutQuart(t) {
        return 1 - Math.pow(1 - t, 4);
    }

    // ============================================================================
    // COUNTER ANIMATION
    // ============================================================================

    /**
     * Animate counting numbers from 0 to target value
     */
    function animateCounter(element) {
        if (element.dataset.animated === 'true') return;

        const text = element.textContent.trim();

        // Extract number and any prefix/suffix
        const match = text.match(/^([^\d]*)([\d,]+(?:\.\d+)?)(.*?)$/);
        if (!match) return;

        const prefix = match[1];
        const numberStr = match[2].replace(/,/g, '');
        const suffix = match[3];
        const targetNumber = parseFloat(numberStr);

        if (isNaN(targetNumber)) return;

        element.dataset.animated = 'true';

        const hasDecimals = numberStr.includes('.');
        const decimalPlaces = hasDecimals ? numberStr.split('.')[1].length : 0;
        const hasCommas = match[2].includes(',');

        const startTime = performance.now();

        function updateCounter(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / CONFIG.counterDuration, 1);
            const easedProgress = easeOutQuart(progress);
            const currentValue = targetNumber * easedProgress;

            let displayValue;
            if (hasDecimals) {
                displayValue = currentValue.toFixed(decimalPlaces);
            } else {
                displayValue = Math.round(currentValue).toString();
            }

            if (hasCommas) {
                displayValue = Number(displayValue).toLocaleString();
            }

            element.textContent = prefix + displayValue + suffix;

            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        }

        requestAnimationFrame(updateCounter);
    }

    /**
     * Initialize counter animations for all metric values
     */
    function initCounterAnimations() {
        const counterElements = document.querySelectorAll('.metric-value, .stat-value, .gauge-value, [data-counter]');

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                }
            });
        }, { threshold: CONFIG.scrollRevealThreshold });

        counterElements.forEach(el => observer.observe(el));
    }

    // ============================================================================
    // SCROLL REVEAL ANIMATIONS
    // ============================================================================

    /**
     * Initialize scroll-triggered reveal animations
     */
    function initScrollReveal() {
        const revealElements = document.querySelectorAll(
            '.metric-card, .chart-card, .alert-item, .info-card, ' +
            '.table-section, .chatbot-section, .system-health, ' +
            '[data-reveal], .animate-on-scroll'
        );

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                    // Add staggered animation for grid items
                    const parent = entry.target.parentElement;
                    if (parent && parent.classList.contains('metrics-grid')) {
                        const siblings = Array.from(parent.children);
                        const index = siblings.indexOf(entry.target);
                        entry.target.style.animationDelay = `${index * 0.1}s`;
                    }
                    entry.target.classList.add('animate-fadeInUp');
                }
            });
        }, {
            threshold: CONFIG.scrollRevealThreshold,
            rootMargin: '0px 0px -50px 0px'
        });

        revealElements.forEach(el => {
            el.style.opacity = '0';
            observer.observe(el);
        });
    }

    // ============================================================================
    // NAVBAR SCROLL EFFECTS
    // ============================================================================

    /**
     * Add blur/glass effect to navbar on scroll
     */
    function initNavbarScrollEffect() {
        const navbar = document.querySelector('.navbar');
        if (!navbar) return;

        let ticking = false;

        function updateNavbar() {
            const scrollY = window.scrollY;

            if (scrollY > 50) {
                navbar.classList.add('navbar-scrolled');
                navbar.style.backdropFilter = 'blur(10px)';
                navbar.style.webkitBackdropFilter = 'blur(10px)';
            } else {
                navbar.classList.remove('navbar-scrolled');
                navbar.style.backdropFilter = '';
                navbar.style.webkitBackdropFilter = '';
            }

            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateNavbar);
                ticking = true;
            }
        });
    }

    // ============================================================================
    // SCROLL TO TOP BUTTON
    // ============================================================================

    /**
     * Create and manage scroll-to-top button
     */
    function initScrollTopButton() {
        // Check if button already exists
        if (document.querySelector('.scroll-top-btn')) return;

        const button = document.createElement('button');
        button.className = 'scroll-top-btn';
        button.innerHTML = '<i class="fas fa-arrow-up"></i>';
        button.setAttribute('aria-label', 'Scroll to top');
        document.body.appendChild(button);

        let ticking = false;

        function updateButton() {
            if (window.scrollY > CONFIG.scrollTopThreshold) {
                button.classList.add('visible');
            } else {
                button.classList.remove('visible');
            }
            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateButton);
                ticking = true;
            }
        });

        button.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // ============================================================================
    // SCROLL PROGRESS INDICATOR
    // ============================================================================

    /**
     * Create scroll progress bar at top of page
     */
    function initScrollProgress() {
        // Check if progress bar already exists
        if (document.querySelector('.scroll-progress')) return;

        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        progressBar.style.width = '0%';
        document.body.appendChild(progressBar);

        let ticking = false;

        function updateProgress() {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
            progressBar.style.width = progress + '%';
            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateProgress);
                ticking = true;
            }
        });
    }

    // ============================================================================
    // RIPPLE EFFECT FOR BUTTONS
    // ============================================================================

    /**
     * Add ripple effect to buttons on click
     */
    function initRippleEffect() {
        document.addEventListener('click', (e) => {
            const button = e.target.closest('.btn, .btn-ripple');
            if (!button) return;

            // Ensure button has ripple class
            if (!button.classList.contains('btn-ripple')) {
                button.classList.add('btn-ripple');
            }

            // Create ripple element
            const ripple = document.createElement('span');
            ripple.className = 'ripple-effect';

            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out forwards;
                pointer-events: none;
            `;

            button.style.position = 'relative';
            button.style.overflow = 'hidden';
            button.appendChild(ripple);

            ripple.addEventListener('animationend', () => {
                ripple.remove();
            });
        });
    }

    // ============================================================================
    // CARD HOVER TILT EFFECT
    // ============================================================================

    /**
     * Add subtle 3D tilt effect to cards on hover
     */
    function initCardTilt() {
        const cards = document.querySelectorAll('.metric-card, .chart-card, .info-card, [data-tilt]');

        cards.forEach(card => {
            card.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
            card.style.transformStyle = 'preserve-3d';

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
                card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
            });
        });
    }

    // ============================================================================
    // SMOOTH ANCHOR SCROLLING
    // ============================================================================

    /**
     * Enable smooth scrolling for anchor links
     */
    function initSmoothAnchors() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                const href = anchor.getAttribute('href');
                if (href === '#') return;

                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // ============================================================================
    // LOADING STATE ENHANCEMENTS
    // ============================================================================

    /**
     * Add loading shimmer to elements while data loads
     */
    function showLoadingState(element) {
        element.classList.add('skeleton', 'animate-shimmer');
    }

    function hideLoadingState(element) {
        element.classList.remove('skeleton', 'animate-shimmer');
    }

    // Expose for external use
    window.netraEffects = {
        showLoading: showLoadingState,
        hideLoading: hideLoadingState
    };

    // ============================================================================
    // TOOLTIP INITIALIZATION
    // ============================================================================

    /**
     * Initialize animated tooltips
     */
    function initTooltips() {
        document.querySelectorAll('[data-tooltip]').forEach(el => {
            el.classList.add('tooltip-animated');
        });
    }

    // ============================================================================
    // TABLE ROW HOVER ENHANCEMENT
    // ============================================================================

    /**
     * Add enhanced hover effects to table rows
     */
    function initTableEnhancements() {
        const tables = document.querySelectorAll('.data-table');

        tables.forEach(table => {
            table.classList.add('table-row-slide');

            // Add staggered animation on first load
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach((row, index) => {
                row.style.opacity = '0';
                row.style.animation = `fadeInUp 0.4s ease forwards`;
                row.style.animationDelay = `${index * 0.05}s`;
            });
        });
    }

    // ============================================================================
    // STAGGERED GRID ANIMATION
    // ============================================================================

    /**
     * Add staggered entrance animation to grid items
     */
    function initStaggeredGrids() {
        const grids = document.querySelectorAll('.metrics-grid, .cards-grid, .alerts-grid, .charts-row');

        grids.forEach(grid => {
            const children = grid.children;
            Array.from(children).forEach((child, index) => {
                child.style.opacity = '0';
                child.style.animation = 'fadeInUp 0.5s ease forwards';
                child.style.animationDelay = `${0.1 + index * 0.08}s`;
            });
        });
    }

    // ============================================================================
    // PAGE TRANSITION EFFECTS
    // ============================================================================

    /**
     * Add fade effect when navigating between pages
     */
    function initPageTransitions() {
        // Fade in on page load
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.3s ease';

        window.addEventListener('load', () => {
            document.body.style.opacity = '1';
        });

        // Fade out before navigation (for same-domain links)
        document.querySelectorAll('a:not([target="_blank"]):not([href^="#"]):not([href^="javascript"])').forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href && !href.startsWith('#') && !href.startsWith('javascript')) {
                    e.preventDefault();
                    document.body.style.opacity = '0';
                    setTimeout(() => {
                        window.location.href = href;
                    }, 200);
                }
            });
        });
    }

    // ============================================================================
    // CHATBOT TYPING INDICATOR
    // ============================================================================

    /**
     * Create typing indicator element
     */
    function createTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        return indicator;
    }

    // Expose for chatbot use
    window.netraEffects.createTypingIndicator = createTypingIndicator;

    // ============================================================================
    // NOTIFICATION TOAST ENHANCEMENTS
    // ============================================================================

    /**
     * Show enhanced toast notification
     */
    function showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type} animate-slideInRight`;
        toast.innerHTML = `
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('animate-fadeOutRight');
            setTimeout(() => toast.remove(), 300);
        }, duration);

        return toast;
    }

    // Expose for external use
    window.netraEffects.showToast = showToast;

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    /**
     * Initialize all effects when DOM is ready
     */
    function init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAll);
        } else {
            initAll();
        }
    }

    function initAll() {
        // Core animations
        initCounterAnimations();
        initScrollReveal();
        initNavbarScrollEffect();
        initScrollTopButton();
        initScrollProgress();

        // Interactions
        initRippleEffect();
        initCardTilt();
        initSmoothAnchors();

        // Enhancements
        initTooltips();
        initTableEnhancements();
        initStaggeredGrids();
        initPageTransitions();

        console.log('✨ NETRA TAX Premium Effects initialized');
    }

    // Start initialization
    init();

})();
