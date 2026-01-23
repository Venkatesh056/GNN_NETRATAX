// ============================================================================
// Chart.js Configuration - Navy Color Palette
// ============================================================================

const NAVY_COLORS = {
    navy: '#2F4156',
    navyLight: '#3d5470',
    navyDark: '#1f2b3a',
    crimsonDepth: '#700034',
    crimsonLight: '#8f0044',
    warmSand: '#C3A582',
    warmSandLight: '#d4b89a',
    softPearl: '#F7F2F0',
    obsidianBlack: '#1B1616',
    
    // Semantic colors
    success: '#27ae60',
    warning: '#f39c12',
    danger: '#e74c3c',
    info: '#3498db',
    
    // Gradients
    navyGradient: ['#2F4156', '#700034'],
    crimsonGradient: ['#700034', '#8f0044'],
    warmGradient: ['#C3A582', '#d4b89a'],
    
    // Chart colors array
    chartColors: [
        '#2F4156', // Navy
        '#700034', // Crimson Depth
        '#C3A582', // Warm Sand
        '#3d5470', // Navy Light
        '#8f0044', // Crimson Light
        '#d4b89a', // Warm Sand Light
        '#27ae60', // Success
        '#f39c12', // Warning
        '#e74c3c', // Danger
        '#3498db'  // Info
    ]
};

// Get theme-aware colors
function getThemeColors() {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    return {
        textColor: isDark ? '#F7F2F0' : '#1B1616',
        gridColor: isDark ? 'rgba(247, 242, 240, 0.1)' : 'rgba(27, 22, 22, 0.1)',
        backgroundColor: isDark ? 'rgba(27, 22, 22, 0.8)' : 'rgba(247, 242, 240, 0.8)',
        borderColor: isDark ? 'rgba(247, 242, 240, 0.2)' : 'rgba(27, 22, 22, 0.2)'
    };
}

// Default Chart.js configuration
const defaultChartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                color: getThemeColors().textColor,
                font: {
                    family: "'Inter', 'Segoe UI', sans-serif",
                    size: 12,
                    weight: '500'
                },
                padding: 15,
                usePointStyle: true,
                pointStyle: 'circle'
            }
        },
        tooltip: {
            backgroundColor: getThemeColors().backgroundColor,
            titleColor: getThemeColors().textColor,
            bodyColor: getThemeColors().textColor,
            borderColor: getThemeColors().borderColor,
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            displayColors: true,
            titleFont: {
                size: 14,
                weight: 'bold'
            },
            bodyFont: {
                size: 13
            }
        }
    },
    scales: {
        x: {
            grid: {
                color: getThemeColors().gridColor,
                drawBorder: false
            },
            ticks: {
                color: getThemeColors().textColor,
                font: {
                    size: 11
                }
            }
        },
        y: {
            grid: {
                color: getThemeColors().gridColor,
                drawBorder: false
            },
            ticks: {
                color: getThemeColors().textColor,
                font: {
                    size: 11
                }
            }
        }
    }
};

// Create gradient for chart backgrounds
function createGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

// Create radial gradient for pie/doughnut charts
function createRadialGradient(ctx, color1, color2) {
    const gradient = ctx.createRadialGradient(150, 150, 0, 150, 150, 150);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

// Update chart theme when theme changes
function updateChartTheme(chart) {
    const themeColors = getThemeColors();
    
    if (chart.options.plugins.legend) {
        chart.options.plugins.legend.labels.color = themeColors.textColor;
    }
    
    if (chart.options.plugins.tooltip) {
        chart.options.plugins.tooltip.backgroundColor = themeColors.backgroundColor;
        chart.options.plugins.tooltip.titleColor = themeColors.textColor;
        chart.options.plugins.tooltip.bodyColor = themeColors.textColor;
        chart.options.plugins.tooltip.borderColor = themeColors.borderColor;
    }
    
    if (chart.options.scales) {
        Object.keys(chart.options.scales).forEach(scaleKey => {
            const scale = chart.options.scales[scaleKey];
            if (scale.grid) {
                scale.grid.color = themeColors.gridColor;
            }
            if (scale.ticks) {
                scale.ticks.color = themeColors.textColor;
            }
        });
    }
    
    chart.update();
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NAVY_COLORS,
        getThemeColors,
        defaultChartConfig,
        createGradient,
        createRadialGradient,
        updateChartTheme
    };
}
