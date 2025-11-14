// ============================================================================
// Tax Fraud Detection Dashboard - JavaScript
// ============================================================================

// Load charts on page load
document.addEventListener('DOMContentLoaded', function() {
    loadFraudDistributionChart();
    loadRiskDistributionChart();
    loadRiskByLocationChart();
    loadTurnoverVsRiskChart();
});

// ============================================================================
// Chart Functions
// ============================================================================

function loadFraudDistributionChart() {
    fetch('/api/chart/fraud_distribution')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('fraudDistributionChart');
            if (container) {
                Plotly.newPlot(container, data.data, data.layout, {responsive: true});
            }
        })
        .catch(error => console.error('Error loading fraud distribution chart:', error));
}

function loadRiskDistributionChart() {
    fetch('/api/chart/risk_distribution')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('riskDistributionChart');
            if (container) {
                Plotly.newPlot(container, data.data, data.layout, {responsive: true});
            }
        })
        .catch(error => console.error('Error loading risk distribution chart:', error));
}

function loadRiskByLocationChart() {
    fetch('/api/chart/risk_by_location')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('riskByLocationChart');
            if (container) {
                Plotly.newPlot(container, data.data, data.layout, {responsive: true});
            }
        })
        .catch(error => console.error('Error loading risk by location chart:', error));
}

function loadTurnoverVsRiskChart() {
    fetch('/api/chart/turnover_vs_risk')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('turnoverVsRiskChart');
            if (container) {
                Plotly.newPlot(container, data.data, data.layout, {responsive: true});
            }
        })
        .catch(error => console.error('Error loading turnover vs risk chart:', error));
}
