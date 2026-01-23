/**
 * NETRA TAX - Dashboard JavaScript
 * Main dashboard functionality and data loading
 */

// Configuration
const REFRESH_INTERVAL = 30000; // Refresh every 30 seconds
const CHART_COLORS = {
    primary: '#0F4C5C',      // Midnight Teal
    secondary: '#FFB703',    // Signal Amber
    danger: '#D62828',       // Alert Crimson
    warning: '#F77F00',      // Deep Orange
    success: '#2A9D8F',      // Secure Green
    info: '#3A86FF',         // Cyber Blue
};

// Global state
let dashboardState = {
    fraudSummary: null,
    systemHealth: null,
    autoRefresh: true,
    refreshInterval: null,
};

/**
 * Initialize dashboard on page load
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Dashboard initializing...');

    // Wait for API detection to complete
    if (api._readyPromise) {
        await api._readyPromise;
    }

    // Update user display
    const userDisplay = document.getElementById('userDisplay');
    if (userDisplay) {
        const user = getCurrentUser();
        userDisplay.textContent = user.full_name || user.username || 'Admin';
    }

    try {
        // Load dashboard data
        await loadDashboardData();

        // Setup event listeners
        setupEventListeners();

        // Setup auto-refresh
        if (dashboardState.autoRefresh) {
            dashboardState.refreshInterval = setInterval(loadDashboardData, REFRESH_INTERVAL);
        }

        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Dashboard initialization error:', error);
        showNotification('Failed to load dashboard - Backend may be offline', 'error');
    }
});

/**
 * Load all dashboard data
 */
async function loadDashboardData() {
    try {
        // Load fraud summary / statistics
        await loadFraudSummary();

        // Load charts
        await loadCharts();

        // Load alerts
        await loadAlerts();

        // Load high-risk companies table
        await loadHighRiskCompanies();

        // Update last update time
        const lastUpdate = document.getElementById('lastUpdate');
        if (lastUpdate) {
            lastUpdate.textContent = 'Just now';
        }

        console.log('Dashboard data loaded successfully');
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Error refreshing data', 'error');
    }
}

/**
 * Load fraud summary and update metrics
 */
async function loadFraudSummary() {
    try {
        const summary = await api.getFraudSummary();
        dashboardState.fraudSummary = summary;

        // Update metric cards by ID
        const totalEntities = document.getElementById('totalEntities');
        const highRiskCount = document.getElementById('highRiskCount');
        const mediumRiskCount = document.getElementById('mediumRiskCount');
        const fraudRingCount = document.getElementById('fraudRingCount');

        if (totalEntities) totalEntities.textContent = formatNumber(summary.total_entities);
        if (highRiskCount) highRiskCount.textContent = formatNumber(summary.high_risk_count);
        if (mediumRiskCount) mediumRiskCount.textContent = formatNumber(summary.medium_risk_count);
        if (fraudRingCount) fraudRingCount.textContent = formatNumber(summary.fraud_rings_detected);

    } catch (error) {
        console.error('Error loading fraud summary:', error);
        // Set fallback values
        const totalEntities = document.getElementById('totalEntities');
        const highRiskCount = document.getElementById('highRiskCount');
        const mediumRiskCount = document.getElementById('mediumRiskCount');
        const fraudRingCount = document.getElementById('fraudRingCount');

        if (totalEntities) totalEntities.textContent = '0';
        if (highRiskCount) highRiskCount.textContent = '0';
        if (mediumRiskCount) mediumRiskCount.textContent = '0';
        if (fraudRingCount) fraudRingCount.textContent = '0';
    }
}

/**
 * Load and render charts
 */
async function loadCharts() {
    try {
        const summary = dashboardState.fraudSummary;
        if (!summary) return;

        // Risk distribution pie chart
        renderRiskDistributionChart(summary);

        // Fraud score distribution (from backend data, no synthetic generation)
        await renderFraudScoreDistributionFromAPI();

        // Trend chart / calendar heatmap
        await renderCalendarHeatmapFromAPI();

        // Fraud ring bubble chart
        await renderFraudRingBubbleFromAPI();

        // Centrality heatmap
        await renderCentralityHeatmapFromAPI();

        // Top Senders & Top Receivers charts
        await renderTopSendersChart();
        await renderTopReceiversChart();
    } catch (error) {
        console.error('Error loading charts:', error);
    }
}

/**
 * Render risk distribution pie chart using Plotly
 */
function renderRiskDistributionChart(summary) {
    const container = document.getElementById('riskDistributionChart');
    if (!container) return;

    const high = summary.high_risk_count || 0;
    const medium = summary.medium_risk_count || 0;
    const low = summary.low_risk_count || (summary.total_entities - high - medium);

    const data = [{
        labels: ['High Risk', 'Medium Risk', 'Low Risk'],
        values: [high, medium, Math.max(0, low)],
        type: 'pie',
        marker: { colors: [CHART_COLORS.danger, CHART_COLORS.warning, CHART_COLORS.success] },
        hoverinfo: 'label+value+percent',
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textposition: 'inside',
        textinfo: 'percent'
    }];

    const layout = {
        height: 300,
        margin: { l: 20, r: 20, t: 20, b: 20 },
        font: { family: 'Inter, sans-serif', color: '#1A1A1A' },
        showlegend: true,
        legend: { orientation: 'v', y: 0.5, x: 1.0 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        hoverlabel: {
            bgcolor: 'white',
            bordercolor: '#0F4C5C',
            font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
        }
    };

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
}

/**
 * Render fraud score distribution using Plotly
 */
async function renderFraudScoreDistributionFromAPI() {
    const container = document.getElementById('fraudScoreChart');
    if (!container) return;
    try {
        const fig = await api.getRiskDistribution();

        // Use layout from API but ensure proper hover settings
        const layout = {
            ...fig.layout,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(0,0,0,0.02)',
            hovermode: 'closest',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#0F4C5C',
                font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
            }
        };

        Plotly.newPlot(container, fig.data, layout, { responsive: true, displayModeBar: false });
    } catch (e) {
        console.error('Risk distribution chart error:', e);
        container.innerHTML = '<div class="empty-state"><p>No data for risk distribution</p></div>';
    }
}

/**
 * Render trend chart using Plotly
 */
async function renderCalendarHeatmapFromAPI() {
    const container = document.getElementById('calendarHeatmap');
    if (!container) return;
    try {
        const data = await api.getTimelineData();
        const x = data.map(d => d.label || d.date);
        const y = data.map(d => d.value || 0);
        const trace = {
            x, y, type: 'bar',
            marker: { color: CHART_COLORS.info },
            hovertemplate: '<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>'
        };
        const layout = {
            height: 300,
            margin: { l: 40, r: 20, t: 20, b: 60 },
            xaxis: { type: 'category' },
            hovermode: 'closest',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#3A86FF',
                font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
            }
        };
        Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
    } catch (e) {
        console.error('Timeline chart error:', e);
        container.innerHTML = '<div class="empty-state"><p>No timeline data</p></div>';
    }
}

async function renderFraudRingBubbleFromAPI() {
    const container = document.getElementById('fraudRingBubbleChart');
    if (!container) return;
    try {
        const data = await api.getFraudRingSizes();
        const trace = {
            x: data.sizes,
            y: data.counts,
            mode: 'markers',
            marker: { size: data.sizes.map(s => s * 5), color: CHART_COLORS.danger, opacity: 0.7 },
            hovertemplate: '<b>Ring Size: %{x}</b><br>Count: %{y}<br>Companies in ring: %{marker.size}<extra></extra>'
        };
        const layout = {
            height: 300,
            xaxis: { title: 'Ring Size' },
            yaxis: { title: 'Count' },
            hovermode: 'closest',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#D62828',
                font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
            }
        };
        Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
    } catch (e) {
        console.error('Fraud ring chart error:', e);
        container.innerHTML = '<div class="empty-state"><p>No fraud ring data</p></div>';
    }
}

async function renderCentralityHeatmapFromAPI() {
    const container = document.getElementById('centralityHeatmap');
    if (!container) return;
    try {
        const res = await api.getCentralityHeatmap();
        const layout = {
            height: 300,
            margin: { l: 60, r: 20, t: 20, b: 80 },
            hovermode: 'closest',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#0F4C5C',
                font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
            }
        };
        // Merge with any existing layout from API
        const finalLayout = { ...layout, ...(res.layout || {}) };
        finalLayout.hoverlabel = layout.hoverlabel;
        finalLayout.hovermode = 'closest';
        Plotly.newPlot(container, res.data, finalLayout, { responsive: true, displayModeBar: false });
    } catch (e) {
        console.error('Centrality heatmap error:', e);
        container.innerHTML = '<div class="empty-state"><p>No centrality data</p></div>';
    }
}

/**
 * Render Top Invoice Senders Chart
 */
async function renderTopSendersChart() {
    const container = document.getElementById('topSendersChart');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/top_senders`);
        const chartData = await response.json();

        if (chartData.error) throw new Error(chartData.error);

        if (chartData.data && chartData.data.length > 0) {
            // Use layout from backend, just enhance with some visual tweaks
            const layout = {
                ...chartData.layout,
                height: 350,
                margin: { l: 100, r: 30, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter, sans-serif', size: 11 },
                hovermode: 'closest',
                hoverlabel: {
                    bgcolor: 'white',
                    bordercolor: '#F77F00',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
                }
            };

            // Add gradient color to bars
            if (chartData.data[0]) {
                const yValues = chartData.data[0].x || chartData.data[0].y;
                chartData.data[0].marker = {
                    color: yValues,
                    colorscale: [[0, '#FFB703'], [0.5, '#F77F00'], [1, '#D62828']],
                    showscale: false
                };
            }

            Plotly.newPlot(container, chartData.data, layout, { responsive: true, displayModeBar: false });
        } else {
            container.innerHTML = '<div class="empty-state" style="height: 100%; display: flex; align-items: center; justify-content: center;"><p>No sender data available</p></div>';
        }
    } catch (error) {
        console.error('Error loading top senders chart:', error);
        container.innerHTML = '<div class="empty-state" style="height: 100%; display: flex; align-items: center; justify-content: center;"><p>Failed to load sender data</p></div>';
    }
}

/**
 * Render Top Invoice Receivers Chart
 */
async function renderTopReceiversChart() {
    const container = document.getElementById('topReceiversChart');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/top_receivers`);
        const chartData = await response.json();

        if (chartData.error) throw new Error(chartData.error);

        if (chartData.data && chartData.data.length > 0) {
            // Use layout from backend, just enhance with some visual tweaks
            const layout = {
                ...chartData.layout,
                height: 350,
                margin: { l: 100, r: 30, t: 40, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter, sans-serif', size: 11 },
                hovermode: 'closest',
                hoverlabel: {
                    bgcolor: 'white',
                    bordercolor: '#3A86FF',
                    font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
                }
            };

            // Add gradient color to bars
            if (chartData.data[0]) {
                const xValues = chartData.data[0].x;
                chartData.data[0].marker = {
                    color: xValues,
                    colorscale: [[0, '#0F4C5C'], [0.5, '#2A9D8F'], [1, '#3A86FF']],
                    showscale: false
                };
            }

            Plotly.newPlot(container, chartData.data, layout, { responsive: true, displayModeBar: false });
        } else {
            container.innerHTML = '<div class="empty-state" style="height: 100%; display: flex; align-items: center; justify-content: center;"><p>No receiver data available</p></div>';
        }
    } catch (error) {
        console.error('Error loading top receivers chart:', error);
        container.innerHTML = '<div class="empty-state" style="height: 100%; display: flex; align-items: center; justify-content: center;"><p>Failed to load receiver data</p></div>';
    }
}

/**
 * Load alerts
 */
async function loadAlerts() {
    const alertsList = document.getElementById('alertsList');
    if (!alertsList) return;

    try {
        const result = await api.getAlerts();
        const alerts = result.data || result.alerts || [];

        if (alerts.length === 0) {
            alertsList.innerHTML = '<div class="empty-state"><p>No active alerts</p></div>';
            return;
        }

        alertsList.innerHTML = alerts.slice(0, 6).map(alert => {
            const riskType = alert.title || 'Suspicious Activity';
            const message = alert.message || 'Details not available';
            const timeStr = formatTimeAgo(alert.timestamp || new Date());
            const isHighRisk = alert.risk === 'high' || alert.severity === 'high';
            const iconColor = isHighRisk ? '#ef4444' : '#f59e0b';
            const icon = isHighRisk ? 'fa-bolt' : 'fa-exclamation-triangle';

            return `
                <div class="alert-item" onclick="window.location.href='/companies'" style="cursor: pointer;">
                    <div class="alert-icon-wrap" style="color: ${iconColor};">
                        <i class="fas ${icon}"></i>
                    </div>
                    <div class="alert-info">
                        <h4>${riskType}</h4>
                        <p>${message}</p>
                        <div class="alert-time">${timeStr}</div>
                    </div>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error loading alerts:', error);
        alertsList.innerHTML = '<div class="error-state"><p>Could not load alerts</p></div>';
    }
}

/**
 * Format timestamp to relative time
 */
function formatTimeAgo(dateString) {
    try {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 60) return `${diffMins}m ago`;
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${Math.floor(diffHours / 24)}d ago`;
    } catch (e) {
        return 'Recently';
    }
}

/**
 * Load high-risk companies table
 */
async function loadHighRiskCompanies() {
    try {
        const tableBody = document.getElementById('topCompaniesTable');
        if (!tableBody) return;

        // Try to get high-risk companies from API
        let companies = [];
        try {
            const response = await api.getHighRiskCompanies();
            companies = response.companies || response || [];
        } catch (e) {
            // Fallback to summary data if available
            if (dashboardState.fraudSummary && dashboardState.fraudSummary.high_risk_companies) {
                companies = dashboardState.fraudSummary.high_risk_companies;
            }
        }

        if (companies.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center">No high-risk companies found. Upload data to start analysis.</td></tr>';
            return;
        }

        const rows = companies.slice(0, 10).map(company => {
            const riskScore = company.risk_score || company.fraud_probability || 0;
            const riskInfo = getRiskLevelBadge(riskScore);
            return `
                <tr>
                    <td><span class="invoice-link">${company.gstin || company.company_id || 'N/A'}</span></td>
                    <td>${company.name || company.company_name || 'Unknown'}</td>
                    <td>${(riskScore * 100).toFixed(1)}%</td>
                    <td><span class="status-badge ${riskInfo.class}">${riskInfo.level}</span></td>
                    <td>${company.invoice_count || company.total_invoices || 0}</td>
                    <td>
                        <div class="action-icons">
                            <button class="action-icon" onclick="viewCompanyDetails('${company.gstin || company.company_id}')">👁️</button>
                            <button class="action-icon" onclick="flagCompany('${company.gstin || company.company_id}')">🚩</button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = rows;
    } catch (error) {
        console.error('Error loading high-risk companies:', error);
        const tableBody = document.getElementById('topCompaniesTable');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center">Error loading data. Please check backend connection.</td></tr>';
        }
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtns = document.querySelectorAll('.btn-refresh, [onclick="refreshData()"]');
    refreshBtns.forEach(btn => {
        btn.onclick = () => {
            showNotification('Refreshing data...', 'info', 1500);
            loadDashboardData();
        };
    });

    // Chart filter dropdowns
    const chartFilters = document.querySelectorAll('.chart-filters select');
    chartFilters.forEach(filter => {
        filter.addEventListener('change', (e) => {
            console.log(`Filter changed: ${e.target.id} = ${e.target.value}`);
            loadCharts();
        });
    });
}

/**
 * Refresh data - called from HTML
 */
function refreshData() {
    showNotification('Refreshing data...', 'info', 1500);
    loadDashboardData();
}

/**
 * View company details
 */
function viewCompanyDetails(companyId) {
    window.location.href = `company-explorer.html?gstin=${companyId}`;
}

/**
 * Flag company for investigation
 */
function flagCompany(companyId) {
    showNotification(`Company ${companyId} flagged for investigation`, 'warning');
}

/**
 * Investigate alert
 */
function investigateAlert(button) {
    const alertItem = button.closest('.alert-item');
    const title = alertItem.querySelector('.alert-title').textContent;
    showNotification(`Investigating: ${title}`, 'info');
}

/**
 * Toggle chart options (placeholder)
 */
function toggleChartOptions(button) {
    showNotification('Chart options coming soon', 'info', 2000);
}

/**
 * User profile functions
 */
function viewProfile() {
    showNotification('Profile page coming soon', 'info');
}

function changePassword() {
    showNotification('Password change coming soon', 'info');
}

function logout() {
    logoutUser();
}

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', () => {
    if (dashboardState.refreshInterval) {
        clearInterval(dashboardState.refreshInterval);
    }
});

// Make functions globally available
window.refreshData = refreshData;
window.viewCompanyDetails = viewCompanyDetails;
window.flagCompany = flagCompany;
window.investigateAlert = investigateAlert;
window.toggleChartOptions = toggleChartOptions;
window.viewProfile = viewProfile;
window.changePassword = changePassword;
window.logout = logout;
// ============================================================================
// CHATBOT FUNCTIONS
// ============================================================================

/**
 * Send a chat message to the backend
 */
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    // Clear input
    input.value = '';

    // Add user message to chat
    addChatMessage(message, 'user');

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch(`${api.baseURL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        // Remove typing indicator
        removeTypingIndicator();

        // Add bot response
        addChatMessage(data.response, 'bot', data.type);

    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addChatMessage('Sorry, I encountered an error. Please try again.', 'bot', 'error');
    }
}

/**
 * Add a message to the chat container
 */
function addChatMessage(text, sender, type = 'info') {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;

    const avatar = sender === 'bot' ? '🤖' : '👤';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Convert markdown-like formatting to HTML
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content ${type}">
            <p>${formattedText}</p>
            <span class="message-time">${time}</span>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;

    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
    const typing = document.getElementById('typingIndicator');
    if (typing) {
        typing.remove();
    }
}

/**
 * Handle Enter key in chat input
 */
function handleChatKeypress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

/**
 * Quick chat buttons
 */
function quickChat(message) {
    const input = document.getElementById('chatInput');
    if (input) {
        input.value = message;
        sendChatMessage();
    }
}

// ============================================================================
// NEW ENHANCED FEATURES
// ============================================================================

/**
 * Load Fraud Risk Gauge
 */
async function loadFraudGauge() {
    const container = document.getElementById('fraudGauge');
    const valueDisplay = document.getElementById('gaugeValue');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/fraud-gauge`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        const gaugeValue = data.gauge_value || 0;
        const color = data.color || '#ef4444';

        // Update value display
        if (valueDisplay) {
            valueDisplay.textContent = `${gaugeValue}%`;
            valueDisplay.style.color = color;
        }

        // Create gauge using Plotly
        const gaugeData = [{
            type: "indicator",
            mode: "gauge+number",
            value: gaugeValue,
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: "#334155" },
                bar: { color: color },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "#e2e8f0",
                steps: [
                    { range: [0, 30], color: "rgba(34, 197, 94, 0.2)" },
                    { range: [30, 60], color: "rgba(245, 158, 11, 0.2)" },
                    { range: [60, 100], color: "rgba(239, 68, 68, 0.2)" }
                ],
                threshold: {
                    line: { color: "#1e293b", width: 4 },
                    thickness: 0.75,
                    value: gaugeValue
                }
            },
            number: { suffix: "%", font: { size: 16 } }
        }];

        const layout = {
            width: 150,
            height: 100,
            margin: { t: 10, r: 10, l: 10, b: 10 },
            paper_bgcolor: "transparent",
            font: { family: "Inter, sans-serif" }
        };

        Plotly.newPlot(container, gaugeData, layout, { displayModeBar: false, responsive: true });
    } catch (error) {
        console.error('Error loading fraud gauge:', error);
        if (valueDisplay) valueDisplay.textContent = 'N/A';
    }
}

/**
 * Load Data Quality Banner
 */
async function loadDataQuality() {
    try {
        const response = await fetch(`${api.baseURL}/api/data-quality`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Update UI elements
        const qualityStatus = document.getElementById('dataQualityStatus');
        const companyCount = document.getElementById('companyCount');
        const invoiceCount = document.getElementById('invoiceCount');
        const qualityBar = document.getElementById('qualityBar');
        const qualityPercent = document.getElementById('qualityPercent');

        if (qualityStatus) qualityStatus.textContent = `Status: ${data.status?.toUpperCase() || 'Unknown'} • ${data.issues?.length || 0} issues detected`;
        if (companyCount) companyCount.textContent = formatNumber(data.metrics?.companies?.total_records || 0);
        if (invoiceCount) invoiceCount.textContent = formatNumber(data.metrics?.invoices?.total_records || 0);
        if (qualityBar) qualityBar.style.width = `${data.completeness_score || 0}%`;
        if (qualityPercent) qualityPercent.textContent = `${data.completeness_score || 0}%`;

        // Update banner color based on status
        const banner = document.getElementById('dataQualityBanner');
        if (banner && data.color) {
            if (data.status === 'excellent') {
                banner.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';
            } else if (data.status === 'good') {
                banner.style.background = 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)';
            } else if (data.status === 'fair') {
                banner.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
            } else {
                banner.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
            }
        }
    } catch (error) {
        console.error('Error loading data quality:', error);
    }
}

/**
 * Load Geographic Risk Map
 */
let geoMap = null;
async function loadGeographicMap() {
    const container = document.getElementById('geographicMap');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/geographic-risk`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Initialize map if not already done
        if (!geoMap) {
            geoMap = L.map('geographicMap').setView([20.5937, 78.9629], 5);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(geoMap);
        }

        // Clear existing markers
        geoMap.eachLayer(layer => {
            if (layer instanceof L.CircleMarker) {
                geoMap.removeLayer(layer);
            }
        });

        // Add markers for each location
        if (data.locations && data.locations.length > 0) {
            data.locations.forEach(loc => {
                const color = loc.risk_level === 'high' ? '#ef4444' :
                    loc.risk_level === 'medium' ? '#f59e0b' : '#22c55e';
                const radius = Math.min(20, Math.max(8, loc.company_count / 5));

                L.circleMarker([loc.lat, loc.lng], {
                    radius: radius,
                    fillColor: color,
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                }).addTo(geoMap)
                    .bindPopup(`
                      <strong>${loc.location}</strong><br>
                      Companies: ${loc.company_count}<br>
                      High Risk: ${loc.high_risk_count}<br>
                      Avg Risk: ${(loc.avg_risk * 100).toFixed(1)}%
                  `);
            });
        }
    } catch (error) {
        console.error('Error loading geographic map:', error);
        container.innerHTML = '<div class="empty-state"><p>Geographic data unavailable</p></div>';
    }
}

/**
 * Load Fraud Pattern Word Cloud
 */
async function loadWordCloud() {
    const container = document.getElementById('fraudWordCloud');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/fraud-patterns`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        if (!data.patterns || data.patterns.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No pattern data</p></div>';
            return;
        }

        // Clear container
        container.innerHTML = '';

        const width = container.offsetWidth || 400;
        const height = container.offsetHeight || 300;

        // Scale font sizes
        const maxValue = Math.max(...data.patterns.map(d => d.value));
        const fontScale = d3.scaleLinear()
            .domain([0, maxValue])
            .range([14, 45]);

        const colorScale = d3.scaleOrdinal()
            .domain(data.patterns.map(d => d.text))
            .range(['#ef4444', '#f59e0b', '#3b82f6', '#22c55e', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']);

        // Create word cloud layout
        const layout = d3.layout.cloud()
            .size([width, height])
            .words(data.patterns.map(d => ({ text: d.text, size: fontScale(d.value), value: d.value })))
            .padding(5)
            .rotate(() => (~~(Math.random() * 2) * 90))
            .font("Inter")
            .fontSize(d => d.size)
            .on("end", draw);

        layout.start();

        function draw(words) {
            const svg = d3.select(container)
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .append("g")
                .attr("transform", `translate(${width / 2},${height / 2})`);

            svg.selectAll("text")
                .data(words)
                .enter()
                .append("text")
                .style("font-size", d => `${d.size}px`)
                .style("font-family", "Inter, sans-serif")
                .style("font-weight", "600")
                .style("fill", d => colorScale(d.text))
                .style("cursor", "pointer")
                .attr("text-anchor", "middle")
                .attr("transform", d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
                .text(d => d.text)
                .on("click", function (event, d) {
                    // Filter companies by pattern when clicked
                    showNotification(`Filtering by: ${d.text}`, 'info');
                });
        }
    } catch (error) {
        console.error('Error loading word cloud:', error);
        container.innerHTML = '<div class="empty-state"><p>Pattern analysis unavailable</p></div>';
    }
}

/**
 * Load Sankey Diagram
 */
async function loadSankeyDiagram() {
    const container = document.getElementById('sankeyDiagram');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/sankey-flow`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No transaction flow data</p></div>';
            return;
        }

        const sankeyData = {
            type: "sankey",
            orientation: "h",
            node: {
                pad: 15,
                thickness: 20,
                line: { color: "black", width: 0.5 },
                label: data.nodes.map(n => n.name),
                color: data.nodes.map(n => n.color),
                hovertemplate: '<b>%{label}</b><br>Risk: %{value:.1%}<extra></extra>'
            },
            link: {
                source: data.links.map(l => l.source),
                target: data.links.map(l => l.target),
                value: data.links.map(l => l.value),
                color: data.links.map(l => l.color),
                hovertemplate: '<b>Transaction</b><br>From: %{source.label}<br>To: %{target.label}<br>Amount: ₹%{value:,.0f}<extra></extra>'
            }
        };

        const layout = {
            title: "",
            font: { size: 10, family: "Inter, sans-serif" },
            height: 350,
            margin: { l: 10, r: 10, t: 10, b: 10 },
            paper_bgcolor: "transparent",
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#F77F00',
                font: { family: 'Inter, sans-serif', size: 12, color: '#1A1A1A' }
            }
        };

        Plotly.newPlot(container, [sankeyData], layout, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Error loading Sankey diagram:', error);
        container.innerHTML = '<div class="empty-state"><p>Transaction flow unavailable</p></div>';
    }
}

/**
 * Load Fraud Trend Chart
 */
async function loadFraudTrends() {
    const container = document.getElementById('fraudTrendChart');
    if (!container) return;

    try {
        const response = await fetch(`${api.baseURL}/api/fraud-trends`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        if (!data.dates || data.dates.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No trend data available</p></div>';
            return;
        }

        const traces = [
            {
                x: data.dates,
                y: data.fraud_counts,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Fraud Cases',
                line: { color: '#ef4444', width: 3 },
                marker: { size: 8 },
                hovertemplate: '<b>%{x}</b><br>Fraud Cases: %{y:,}<extra></extra>'
            },
            {
                x: data.dates,
                y: data.fraud_rate,
                type: 'scatter',
                mode: 'lines',
                name: 'Fraud Rate (%)',
                line: { color: '#3b82f6', width: 2, dash: 'dot' },
                yaxis: 'y2',
                hovertemplate: '<b>%{x}</b><br>Fraud Rate: %{y:.1f}%<extra></extra>'
            }
        ];

        const layout = {
            height: 350,
            margin: { l: 50, r: 50, t: 20, b: 60 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter, sans-serif' },
            xaxis: { title: 'Period', tickangle: -45 },
            yaxis: { title: 'Fraud Cases', side: 'left' },
            yaxis2: { title: 'Rate (%)', side: 'right', overlaying: 'y' },
            legend: { x: 0, y: 1.15, orientation: 'h' },
            showlegend: true,
            hovermode: 'x unified',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#3b82f6',
                font: { family: 'Inter, sans-serif', size: 13, color: '#1A1A1A' }
            }
        };

        Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
    } catch (error) {
        console.error('Error loading fraud trends:', error);
        container.innerHTML = '<div class="empty-state"><p>Trend analysis unavailable</p></div>';
    }
}

/**
 * Advanced Search Function
 */
async function performAdvancedSearch() {
    const searchData = {
        gstin: document.getElementById('searchGstin')?.value || '',
        company_name: document.getElementById('searchCompanyName')?.value || '',
        risk_level: document.getElementById('searchRiskLevel')?.value || '',
        location: document.getElementById('searchLocation')?.value || '',
        min_fraud_score: document.getElementById('searchMinScore')?.value ? parseFloat(document.getElementById('searchMinScore').value) : null,
        max_fraud_score: document.getElementById('searchMaxScore')?.value ? parseFloat(document.getElementById('searchMaxScore').value) : null,
        limit: 50
    };

    // Remove null/empty values
    Object.keys(searchData).forEach(key => {
        if (searchData[key] === '' || searchData[key] === null) {
            delete searchData[key];
        }
    });

    try {
        const response = await fetch(`${api.baseURL}/api/advanced-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchData)
        });

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Show results
        const resultsDiv = document.getElementById('searchResults');
        const countSpan = document.getElementById('searchResultCount');
        const tableDiv = document.getElementById('searchResultsTable');

        if (resultsDiv) resultsDiv.style.display = 'block';
        if (countSpan) countSpan.textContent = data.total || 0;

        if (data.results && data.results.length > 0) {
            let tableHTML = `
                <table class="data-table" style="width: 100%;">
                    <thead>
                        <tr>
                            <th>GSTIN</th>
                            <th>Company Name</th>
                            <th>Fraud Score</th>
                            <th>Risk Level</th>
                            <th>Location</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            data.results.forEach(company => {
                const score = company.fraud_probability || 0;
                const riskInfo = getRiskLevelBadge(score);
                tableHTML += `
                    <tr>
                        <td>${company.gstin || company.company_id || 'N/A'}</td>
                        <td>${company.name || company.company_name || 'Unknown'}</td>
                        <td>${(score * 100).toFixed(1)}%</td>
                        <td><span class="status-badge ${riskInfo.class}">${riskInfo.level}</span></td>
                        <td>${company.location || company.city || 'N/A'}</td>
                        <td>
                            <button class="action-icon" onclick="viewCompanyDetails('${company.gstin || company.company_id}')">👁️</button>
                        </td>
                    </tr>
                `;
            });

            tableHTML += '</tbody></table>';
            if (tableDiv) tableDiv.innerHTML = tableHTML;
        } else {
            if (tableDiv) tableDiv.innerHTML = '<p style="text-align: center; padding: 20px; color: #64748b;">No companies found matching your criteria.</p>';
        }

        showNotification(`Found ${data.total} results`, 'success');
    } catch (error) {
        console.error('Error performing search:', error);
        showNotification('Search failed: ' + error.message, 'error');
    }
}

// Update loadDashboardData to include new features
const originalLoadDashboardData = loadDashboardData;
loadDashboardData = async function () {
    await originalLoadDashboardData.call(this);

    // Load new enhanced features
    await Promise.all([
        loadFraudGauge(),
        loadDataQuality(),
        loadGeographicMap(),
        loadWordCloud(),
        loadSankeyDiagram(),
        loadFraudTrends()
    ]);
};

// Make new functions globally available
window.performAdvancedSearch = performAdvancedSearch;
window.loadFraudGauge = loadFraudGauge;
window.loadDataQuality = loadDataQuality;
window.loadGeographicMap = loadGeographicMap;
window.loadWordCloud = loadWordCloud;
window.loadSankeyDiagram = loadSankeyDiagram;
window.loadFraudTrends = loadFraudTrends;

// Make chatbot functions globally available
window.sendChatMessage = sendChatMessage;
window.handleChatKeypress = handleChatKeypress;
window.quickChat = quickChat;