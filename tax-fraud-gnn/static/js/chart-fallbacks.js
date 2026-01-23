/* ============================================================================
   Chart Fallbacks & Error Handling
   Shows beautiful loading states and fallback data when API fails
   ============================================================================ */

// Navy color palette
const FALLBACK_COLORS = {
    navy: '#2F4156',
    crimson: '#700034',
    sand: '#C3A582',
    pearl: '#F7F2F0',
    black: '#1B1616'
};

/* ============================================================================
   Loading Skeleton
   ============================================================================ */

function showChartLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="chart-loading-skeleton">
            <div class="skeleton-header"></div>
            <div class="skeleton-bars">
                <div class="skeleton-bar" style="height: 60%;"></div>
                <div class="skeleton-bar" style="height: 80%;"></div>
                <div class="skeleton-bar" style="height: 45%;"></div>
                <div class="skeleton-bar" style="height: 90%;"></div>
                <div class="skeleton-bar" style="height: 70%;"></div>
            </div>
            <div class="skeleton-footer"></div>
        </div>
    `;
}

function showChartError(containerId, message = 'Unable to load chart data') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="chart-error-state">
            <div class="error-icon">📊</div>
            <div class="error-message">${message}</div>
            <div class="error-hint">The backend API may not be returning data. Check console for details.</div>
            <button class="btn btn-secondary" onclick="location.reload()">
                🔄 Retry
            </button>
        </div>
    `;
}

/* ============================================================================
   Demo/Fallback Data
   ============================================================================ */

function getFallbackFraudDistribution() {
    console.log('📊 Creating Fraud Distribution fallback data with hover template');
    return {
        data: [{
            values: [333, 2613],
            labels: ['Fraud', 'Non-Fraud'],
            type: 'pie',
            marker: {
                colors: [FALLBACK_COLORS.crimson, FALLBACK_COLORS.navy]
            },
            textinfo: 'label+percent',
            textfont: { size: 14, color: '#FFFFFF' },
            hole: 0.4,
            hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        }],
        layout: {
            title: 'Fraud Distribution (Demo Data)',
            showlegend: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Segoe UI, sans-serif' },
            hovermode: 'closest'
        }
    };
}

function getFallbackRiskDistribution() {
    console.log('📊 Creating Risk Distribution fallback data with hover template');
    return {
        data: [{
            x: ['Low Risk', 'Medium Risk', 'High Risk'],
            y: [1200, 850, 896],
            type: 'bar',
            marker: {
                color: [FALLBACK_COLORS.navy, FALLBACK_COLORS.sand, FALLBACK_COLORS.crimson]
            },
            text: [1200, 850, 896],
            textposition: 'outside',
            hovertemplate: '<b>%{x}</b><br>Companies: %{y}<extra></extra>'
        }],
        layout: {
            title: 'Risk Score Distribution (Demo Data)',
            xaxis: { title: 'Risk Level' },
            yaxis: { title: 'Number of Companies' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Segoe UI, sans-serif' },
            hovermode: 'closest'
        }
    };
}

function getFallbackRiskByLocation() {
    console.log('📊 Creating Risk by Location fallback data with hover template');
    return {
        data: [{
            x: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata'],
            y: [0.72, 0.68, 0.45, 0.58, 0.52, 0.48, 0.65],
            type: 'bar',
            marker: {
                color: FALLBACK_COLORS.crimson,
                line: { color: FALLBACK_COLORS.navy, width: 2 }
            },
            hovertemplate: '<b>%{x}</b><br>Avg Risk Score: %{y:.2f}<extra></extra>'
        }],
        layout: {
            title: 'Average Risk Score by Location (Demo Data)',
            xaxis: { title: 'Location' },
            yaxis: { title: 'Average Risk Score', range: [0, 1] },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Segoe UI, sans-serif' },
            hovermode: 'closest'
        }
    };
}

function getFallbackTurnoverVsRisk() {
    console.log('📊 Creating Turnover vs Risk fallback data with hover template');
    // Generate random scatter data
    const x = [];
    const y = [];
    const colors = [];
    const companyIds = [];
    
    for (let i = 0; i < 100; i++) {
        x.push(Math.random() * 10000000);
        y.push(Math.random());
        colors.push(y[i] > 0.7 ? FALLBACK_COLORS.crimson : 
                   y[i] > 0.4 ? FALLBACK_COLORS.sand : 
                   FALLBACK_COLORS.navy);
        companyIds.push(`GST${String(i).padStart(6, '0')}`);
    }
    
    return {
        data: [{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 8,
                color: colors,
                opacity: 0.7,
                line: { color: '#FFFFFF', width: 1 }
            },
            text: companyIds,
            hovertemplate: '<b>%{text}</b><br>Turnover: ₹%{x:,.0f}<br>Risk Score: %{y:.2f}<extra></extra>'
        }],
        layout: {
            title: 'Company Turnover vs Fraud Risk (Demo Data)',
            xaxis: { title: 'Turnover (₹)', type: 'log' },
            yaxis: { title: 'Fraud Risk Score', range: [0, 1] },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Segoe UI, sans-serif' },
            hovermode: 'closest'
        }
    };
}

/* ============================================================================
   Enhanced Chart Loader with Fallback
   ============================================================================ */

async function loadChartWithFallback(containerId, apiEndpoint, fallbackDataFn) {
    // Show loading state
    showChartLoading(containerId);
    
    try {
        const response = await fetch(apiEndpoint);
        
        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }
        
        const data = await response.json();
        
        // Check if data is valid
        if (!data || !data.data || data.data.length === 0) {
            throw new Error('No data returned from API');
        }
        
        // Render actual data with enhanced hover
        const themedLayout = applyThemeToLayout(data.layout);
        
        // Ensure container has proper positioning
        const container = document.getElementById(containerId);
        if (container) {
            container.style.position = 'relative';
            container.style.transform = 'none';
        }
        
        Plotly.newPlot(containerId, data.data, themedLayout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }).then(() => {
            // Force Plotly to recalculate positions after render
            if (container) {
                Plotly.Plots.resize(containerId);
            }
        });
        
        console.log(`✓ Chart loaded successfully: ${containerId}`);
        
    } catch (error) {
        console.warn(`⚠ API failed for ${containerId}, using fallback data:`, error.message);
        
        // Use fallback data
        if (fallbackDataFn) {
            const fallbackData = fallbackDataFn();
            const themedLayout = applyThemeToLayout(fallbackData.layout);
            
            // Ensure container has proper positioning
            const container = document.getElementById(containerId);
            if (container) {
                container.style.position = 'relative';
                container.style.transform = 'none';
            }
            
            Plotly.newPlot(containerId, fallbackData.data, themedLayout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
            }).then(() => {
                // Force Plotly to recalculate positions after render
                if (container) {
                    Plotly.Plots.resize(containerId);
                }
            });
            
            // Add watermark to show it's demo data
            const container = document.getElementById(containerId);
            if (container) {
                const watermark = document.createElement('div');
                watermark.className = 'chart-demo-watermark';
                watermark.textContent = '📊 Demo Data';
                watermark.title = 'Backend API not available - showing sample data';
                container.style.position = 'relative';
                container.appendChild(watermark);
            }
        } else {
            showChartError(containerId, 'Chart data unavailable');
        }
    }
}

function applyThemeToLayout(layout) {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? FALLBACK_COLORS.pearl : FALLBACK_COLORS.black;
    const gridColor = isDark ? 'rgba(247, 242, 240, 0.1)' : 'rgba(27, 22, 22, 0.1)';
    
    return {
        ...layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { ...layout.font, color: textColor },
        xaxis: { ...layout.xaxis, gridcolor: gridColor, tickfont: { color: textColor } },
        yaxis: { ...layout.yaxis, gridcolor: gridColor, tickfont: { color: textColor } }
    };
}

/* ============================================================================
   Initialize All Charts with Fallbacks
   ============================================================================ */

function initializeDashboardCharts() {
    console.log('🚀 Initializing dashboard charts with fallback support...');
    
    // Load all charts with fallback data
    loadChartWithFallback(
        'fraudDistributionChart',
        '/api/chart/fraud_distribution',
        getFallbackFraudDistribution
    );
    
    loadChartWithFallback(
        'riskDistributionChart',
        '/api/chart/risk_distribution',
        getFallbackRiskDistribution
    );
    
    loadChartWithFallback(
        'riskByLocationChart',
        '/api/chart/risk_by_location',
        getFallbackRiskByLocation
    );
    
    loadChartWithFallback(
        'turnoverVsRiskChart',
        '/api/chart/turnover_vs_risk',
        getFallbackTurnoverVsRisk
    );
}

// Auto-initialize if Plotly is available
if (typeof Plotly !== 'undefined') {
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(initializeDashboardCharts, 500);
    });
}
