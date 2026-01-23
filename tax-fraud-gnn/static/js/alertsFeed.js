// ============================================================================
// Real-time Alerts Feed - Navy Color Palette
// ============================================================================

class AlertsFeed {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.maxAlerts = options.maxAlerts || 10;
        this.autoRefresh = options.autoRefresh !== false;
        this.refreshInterval = options.refreshInterval || 30000; // 30 seconds
        this.alerts = [];
        
        // Navy color palette
        this.colors = {
            navy: '#2F4156',
            crimson: '#700034',
            warmSand: '#C3A582',
            softPearl: '#F7F2F0',
            obsidian: '#1B1616',
            highRisk: '#700034',
            mediumRisk: '#C3A582',
            lowRisk: '#2F4156',
            success: '#27ae60',
            warning: '#f39c12',
            danger: '#e74c3c'
        };
        
        this.init();
    }
    
    init() {
        // Create alerts container
        this.container.innerHTML = `
            <div class="alerts-feed">
                <div class="alerts-header">
                    <h3>🚨 Real-time Fraud Alerts</h3>
                    <button class="btn-refresh" id="refreshAlerts">
                        <span class="refresh-icon">↻</span>
                    </button>
                </div>
                <div class="alerts-list" id="alertsList"></div>
            </div>
        `;
        
        // Add styles
        this.addStyles();
        
        // Setup refresh button
        document.getElementById('refreshAlerts').addEventListener('click', () => {
            this.refresh();
        });
        
        // Auto-refresh if enabled
        if (this.autoRefresh) {
            setInterval(() => this.refresh(), this.refreshInterval);
        }
        
        // Initial load
        this.refresh();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .alerts-feed {
                background: var(--bg-card);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px var(--shadow);
                border: 1px solid var(--border-color);
            }
            
            .alerts-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid var(--border-color);
            }
            
            .alerts-header h3 {
                margin: 0;
                color: var(--text-primary);
                font-size: 18px;
            }
            
            .btn-refresh {
                background: var(--navy);
                color: var(--soft-pearl);
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 18px;
            }
            
            .btn-refresh:hover {
                background: var(--navy-hover);
                transform: rotate(180deg);
            }
            
            .alerts-list {
                max-height: 500px;
                overflow-y: auto;
            }
            
            .alert-item {
                background: var(--bg-secondary);
                border-left: 4px solid var(--navy);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 12px;
                transition: all 0.3s ease;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .alert-item:hover {
                transform: translateX(5px);
                box-shadow: 0 4px 12px var(--shadow);
            }
            
            .alert-item.high-risk {
                border-left-color: ${this.colors.highRisk};
            }
            
            .alert-item.medium-risk {
                border-left-color: ${this.colors.mediumRisk};
            }
            
            .alert-item.low-risk {
                border-left-color: ${this.colors.lowRisk};
            }
            
            .alert-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .alert-title {
                font-weight: 600;
                color: var(--text-primary);
                font-size: 14px;
            }
            
            .alert-time {
                font-size: 11px;
                color: var(--text-secondary);
            }
            
            .alert-body {
                color: var(--text-primary);
                font-size: 13px;
                line-height: 1.5;
            }
            
            .alert-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                margin-top: 8px;
            }
            
            .alert-badge.high {
                background: rgba(112, 0, 52, 0.2);
                color: ${this.colors.highRisk};
            }
            
            .alert-badge.medium {
                background: rgba(195, 165, 130, 0.2);
                color: ${this.colors.mediumRisk};
            }
            
            .alert-badge.low {
                background: rgba(47, 65, 86, 0.2);
                color: ${this.colors.lowRisk};
            }
        `;
        document.head.appendChild(style);
    }
    
    async refresh() {
        try {
            // Fetch alerts from API
            const response = await fetch('/api/alerts');
            if (!response.ok) {
                throw new Error('Failed to fetch alerts');
            }
            
            const data = await response.json();
            this.alerts = data.alerts || [];
            this.render();
        } catch (error) {
            console.error('Error fetching alerts:', error);
            // Use mock data if API fails
            this.loadMockData();
        }
    }
    
    loadMockData() {
        // Mock data for demonstration
        this.alerts = [
            {
                id: 1,
                title: 'High-Risk Transaction Detected',
                message: 'Company GST123456 made suspicious transaction of ₹5,00,000',
                risk: 'high',
                timestamp: new Date(Date.now() - 5 * 60000)
            },
            {
                id: 2,
                title: 'Circular Trading Pattern',
                message: 'Detected circular trading between 3 companies',
                risk: 'high',
                timestamp: new Date(Date.now() - 15 * 60000)
            },
            {
                id: 3,
                title: 'GST Compliance Issue',
                message: 'Company GST789012 has low compliance rate (45%)',
                risk: 'medium',
                timestamp: new Date(Date.now() - 30 * 60000)
            },
            {
                id: 4,
                title: 'Late Filing Detected',
                message: 'Company GST345678 filed returns 15 days late',
                risk: 'low',
                timestamp: new Date(Date.now() - 60 * 60000)
            },
            {
                id: 5,
                title: 'Round Amount Transaction',
                message: 'Multiple round amount transactions detected',
                risk: 'medium',
                timestamp: new Date(Date.now() - 90 * 60000)
            }
        ];
        
        this.render();
    }
    
    render() {
        const alertsList = document.getElementById('alertsList');
        
        if (this.alerts.length === 0) {
            alertsList.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    <p>No alerts at this time</p>
                    <p style="font-size: 12px; margin-top: 10px;">System is monitoring for fraud patterns</p>
                </div>
            `;
            return;
        }
        
        // Sort by timestamp (newest first)
        const sortedAlerts = [...this.alerts].sort((a, b) => 
            new Date(b.timestamp) - new Date(a.timestamp)
        );
        
        // Limit to maxAlerts
        const displayAlerts = sortedAlerts.slice(0, this.maxAlerts);
        
        alertsList.innerHTML = displayAlerts.map(alert => `
            <div class="alert-item ${alert.risk}-risk">
                <div class="alert-header">
                    <span class="alert-title">${alert.title}</span>
                    <span class="alert-time">${this.formatTime(alert.timestamp)}</span>
                </div>
                <div class="alert-body">
                    ${alert.message}
                </div>
                <span class="alert-badge ${alert.risk}">
                    ${alert.risk.toUpperCase()} RISK
                </span>
            </div>
        `).join('');
    }
    
    formatTime(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = Math.floor((now - time) / 1000); // seconds
        
        if (diff < 60) return 'Just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return `${Math.floor(diff / 86400)}d ago`;
    }
    
    addAlert(alert) {
        this.alerts.unshift(alert);
        if (this.alerts.length > this.maxAlerts) {
            this.alerts.pop();
        }
        this.render();
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AlertsFeed;
}
