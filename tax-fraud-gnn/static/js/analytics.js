// ============================================================================
// Analytics Page - JavaScript
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Analytics page loaded');
    loadStatistics();
    loadTopSendersChart();
    loadTopReceiversChart();
});

function loadStatistics() {
    console.log('Loading statistics...');
    fetch('/api/statistics')
        .then(response => {
            console.log('Statistics response:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Statistics data:', data);
            
            // Check for errors
            if (data.error) {
                console.error('API returned error:', data.error);
                showError('Error loading statistics: ' + data.error);
                return;
            }
            
            // Update stat cards
            if (document.getElementById('totalNodes')) {
                document.getElementById('totalNodes').textContent = data.total_companies || '-';
            }
            if (document.getElementById('totalEdges')) {
                document.getElementById('totalEdges').textContent = data.total_edges || '-';
            }
            if (document.getElementById('networkDensity')) {
                // Calculate network density (simplified)
                const density = data.total_edges && data.total_companies > 1 
                    ? (data.total_edges / (data.total_companies * (data.total_companies - 1) / 2)).toFixed(4)
                    : '0';
                document.getElementById('networkDensity').textContent = density;
            }
            if (document.getElementById('avgFraudProb')) {
                const avgProb = data.average_fraud_probability 
                    ? (data.average_fraud_probability * 100).toFixed(2) + '%'
                    : '-';
                document.getElementById('avgFraudProb').textContent = avgProb;
            }
            
            // Update risk distribution
            if (document.getElementById('highRiskValue')) {
                document.getElementById('highRiskValue').textContent = data.high_risk_count || 0;
            }
            if (document.getElementById('mediumRiskValue')) {
                document.getElementById('mediumRiskValue').textContent = data.medium_risk_count || 0;
            }
            if (document.getElementById('lowRiskValue')) {
                document.getElementById('lowRiskValue').textContent = data.low_risk_count || 0;
            }
        })
        .catch(error => {
            console.error('Error loading statistics:', error);
            showError('Failed to load statistics. Check console for details.');
        });
}

function loadTopSendersChart() {
    console.log('Loading top senders chart...');
    fetch('/api/top_senders')
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Top senders chart data:', data);
            if (data.data && data.layout) {
                Plotly.newPlot('topSendersChart', data.data, data.layout, {responsive: true});
            } else {
                console.error('Invalid chart data format');
            }
        })
        .catch(error => {
            console.error('Error loading top senders chart:', error);
            document.getElementById('topSendersChart').innerHTML = '<p style="text-align: center; color: red;">Error loading chart</p>';
        });
}

function loadTopReceiversChart() {
    console.log('Loading top receivers chart...');
    fetch('/api/top_receivers')
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Top receivers chart data:', data);
            if (data.data && data.layout) {
                Plotly.newPlot('topReceiversChart', data.data, data.layout, {responsive: true});
            } else {
                console.error('Invalid chart data format');
            }
        })
        .catch(error => {
            console.error('Error loading top receivers chart:', error);
            document.getElementById('topReceiversChart').innerHTML = '<p style="text-align: center; color: red;">Error loading chart</p>';
        });
}

function showError(message) {
    // Show error in UI elements
    ['totalNodes', 'totalEdges', 'networkDensity', 'avgFraudProb', 'highRiskValue', 'mediumRiskValue', 'lowRiskValue'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = 'Error';
    });
    console.error(message);
}
