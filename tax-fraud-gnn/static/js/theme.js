// ============================================================================
// Theme Management - Shared across all pages
// ============================================================================

(function() {
    'use strict';
    
    // Initialize theme on page load
    function initTheme() {
        const body = document.body;
        const savedTheme = localStorage.getItem('theme') || 'light';
        body.setAttribute('data-theme', savedTheme);
    }
    
    // Update Plotly charts when theme changes
    function updateChartsTheme() {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#F1F6F4' : '#172B36';
        const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
        
        const themeLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor },
            xaxis: { gridcolor: gridColor, tickfont: { color: textColor } },
            yaxis: { gridcolor: gridColor, tickfont: { color: textColor } }
        };
        
        // Update all Plotly charts on the page
        const chartIds = ['fraudDistributionChart', 'riskDistributionChart', 'riskByLocationChart', 'turnoverVsRiskChart'];
        chartIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.data) {
                Plotly.relayout(id, themeLayout);
            }
        });
    }
    
    // Setup theme toggle button
    function setupThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        if (!themeToggle) return;
        
        const body = document.body;
        
        themeToggle.addEventListener('click', function() {
            const currentTheme = body.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update charts after theme change
            setTimeout(updateChartsTheme, 100);
        });
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initTheme();
            setupThemeToggle();
        });
    } else {
        initTheme();
        setupThemeToggle();
    }
})();

