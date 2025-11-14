// ============================================================================
// Tax Fraud Detection Analytics Page - JavaScript
// ============================================================================

// Load data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStatistics();
    loadTopSenders();
    loadTopReceivers();
});

// ============================================================================
// Load Statistics
// ============================================================================

function loadStatistics() {
    fetch('/api/statistics')
        .then(response => response.json())
        .then(stats => {
            document.getElementById('totalNodes').textContent = stats.total_companies;
            document.getElementById('totalEdges').textContent = stats.total_edges;
            document.getElementById('networkDensity').textContent = 
                (stats.total_edges > 0 ? (2 * stats.total_edges / (stats.total_companies * (stats.total_companies - 1))).toFixed(4) : '0.0000');
            document.getElementById('avgFraudProb').textContent = 
                (stats.average_fraud_probability * 100).toFixed(1) + '%';

            // Update risk distribution
            document.getElementById('highRiskValue').textContent = stats.high_risk_count;
            document.getElementById('mediumRiskValue').textContent = stats.medium_risk_count;
            document.getElementById('lowRiskValue').textContent = stats.low_risk_count;
        })
        .catch(error => console.error('Error loading statistics:', error));
}

// ============================================================================
// Load Top Senders
// ============================================================================

function loadTopSenders() {
    fetch('/api/top_senders')
        .then(response => response.json())
        .then(data => {
            if (data.length > 0) {
                const companies = data.map(d => `Company #${d.company_id}`);
                const counts = data.map(d => d.invoice_count);

                const trace = {
                    x: companies,
                    y: counts,
                    type: 'bar',
                    marker: {color: '#3498db'}
                };

                const layout = {
                    title: 'Top 10 Invoice Senders',
                    xaxis: {title: 'Company ID'},
                    yaxis: {title: 'Invoice Count'},
                    margin: {b: 100}
                };

                Plotly.newPlot('topSendersChart', [trace], layout);
            }
        })
        .catch(error => console.error('Error loading top senders:', error));
}

// ============================================================================
// Load Top Receivers
// ============================================================================

function loadTopReceivers() {
    fetch('/api/top_receivers')
        .then(response => response.json())
        .then(data => {
            if (data.length > 0) {
                const companies = data.map(d => `Company #${d.company_id}`);
                const counts = data.map(d => d.invoice_count);

                const trace = {
                    x: companies,
                    y: counts,
                    type: 'bar',
                    marker: {color: '#e74c3c'}
                };

                const layout = {
                    title: 'Top 10 Invoice Receivers',
                    xaxis: {title: 'Company ID'},
                    yaxis: {title: 'Invoice Count'},
                    margin: {b: 100}
                };

                Plotly.newPlot('topReceiversChart', [trace], layout);
            }
        })
        .catch(error => console.error('Error loading top receivers:', error));
}
