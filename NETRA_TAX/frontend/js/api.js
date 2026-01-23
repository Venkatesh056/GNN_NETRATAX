/**
 * NETRA TAX - API Client Utility Module
 * - Provides APIClient class
 * - Exposes global `api` instance
 * - Auto-detects backend on localhost: tries port 5000 first (Flask), then 8000 (FastAPI)
 */

class APIClient {
    constructor(baseURL = 'http://localhost:5000') {
        this.baseURL = baseURL;
        this.token = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
        this._ready = false;
        this._readyPromise = null;
    }

    // Wait for API detection to complete
    async waitForReady() {
        if (this._ready) return;
        if (this._readyPromise) return this._readyPromise;
        return Promise.resolve();
    }

    setBaseURL(url) {
        this.baseURL = url.replace(/\/$/, ''); // remove trailing slash
    }

    setToken(token) {
        this.token = token;
        localStorage.setItem('access_token', token);
    }

    getHeaders(contentType = 'application/json') {
        const headers = { 'Content-Type': contentType };
        if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
        return headers;
    }

    async request(endpoint, options = {}) {
        const url = endpoint.match(/^https?:\/\//) ? endpoint : `${this.baseURL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
        const defaultOptions = { headers: this.getHeaders(), ...options };

        try {
            const resp = await fetch(url, defaultOptions);
            if (!resp.ok) {
                const errBody = await resp.text().catch(() => null);
                let err = errBody;
                try { err = JSON.parse(errBody); } catch (_) {}
                throw new Error(err && err.detail ? err.detail : `HTTP ${resp.status}: ${resp.statusText}`);
            }
            return await resp.json();
        } catch (e) {
            console.error('API request failed:', e);
            throw e;
        }
    }

    // --- Dashboard APIs (matching actual backend endpoints) ---
    async getStatistics() { 
        return this.request('/api/statistics', { method: 'GET' }); 
    }
    
    async getFraudSummary() { 
        // The backend returns statistics which includes fraud summary
        const stats = await this.request('/api/statistics', { method: 'GET' });
        return {
            total_entities: stats.total_companies || 0,
            high_risk_count: stats.high_risk_count || 0,
            medium_risk_count: stats.medium_risk_count || 0,
            low_risk_count: stats.low_risk_count || 0,
            fraud_rings_detected: stats.fraud_rings || 0,
            high_risk_companies: stats.high_risk_companies || []
        };
    }

    // --- Company APIs ---
    async getCompanies(params = {}) {
        const queryParams = new URLSearchParams();
        if (params.page) queryParams.append('page', params.page);
        if (params.limit) queryParams.append('limit', params.limit);
        if (params.sort) queryParams.append('sort', params.sort);
        if (params.risk_level) queryParams.append('risk_level', params.risk_level);
        if (params.location) queryParams.append('location', params.location);
        if (params.search) queryParams.append('search', params.search);
        if (params.min_score) queryParams.append('min_score', params.min_score);
        const query = queryParams.toString();
        return this.request(`/api/companies${query ? '?' + query : ''}`, { method: 'GET' });
    }

    async getCompanyById(companyId) {
        return this.request(`/api/company/${encodeURIComponent(companyId)}`, { method: 'GET' });
    }

    async getHighRiskCompanies() {
        return this.request('/api/high_risk_companies', { method: 'GET' });
    }

    // --- Chart Data APIs ---
    async getFraudDistribution() {
        return this.request('/api/chart/fraud_distribution', { method: 'GET' });
    }

    async getRiskDistribution() {
        return this.request('/api/chart/risk_distribution', { method: 'GET' });
    }

    async getRiskByLocation() {
        return this.request('/api/chart/risk_by_location', { method: 'GET' });
    }

    async getTurnoverVsRisk() {
        return this.request('/api/chart/turnover_vs_risk', { method: 'GET' });
    }

    // --- Network Graph ---
    async getNetworkGraph() {
        return this.request('/api/network-graph', { method: 'GET' });
    }

    async getNetworkGraphByCenter(center, depth = 2) {
        const params = new URLSearchParams({ center, depth: String(depth) });
        return this.request(`/api/network-graph?${params.toString()}`, { method: 'GET' });
    }

    // --- Company Search & Analysis (wired to backend)
    async searchCompanies(query, page = 0, limit = 20) {
        const params = new URLSearchParams({ search: query, risk_threshold: '0.0' });
        return this.request(`/api/companies?${params.toString()}`, { method: 'GET' });
    }

    async getNetworkAnalysis(gstin) {
        // Derive simple analysis from ego network
        const graph = await this.getNetworkGraphByCenter(gstin, 2);
        const nodes = graph.nodes || [];
        const links = graph.links || [];
        const connected = new Set();
        links.forEach(l => { connected.add(l.source); connected.add(l.target); });
        const centerDegree = links.filter(l => l.source === gstin || l.target === gstin).length;
        const centerNode = nodes.find(n => n.id === gstin) || { risk: 0 };
        return {
            connected_entities: connected.size,
            network_depth: 2,
            in_fraud_ring: false,
            gnn_risk_score: centerNode.risk,
            pattern_anomalies: Math.max(0, centerDegree - 3),
            network_centrality: centerDegree / Math.max(1, nodes.length - 1),
            historical_patterns: 'N/A'
        };
    }

    async generateReport(gstin) {
        // Placeholder: backend report endpoint not defined yet
        return { report_id: `RPT-${Date.now()}` };
    }

    // --- Advanced Analytics ---
    async getFraudRingSizes() {
        return this.request('/api/fraud-ring-sizes', { method: 'GET' });
    }

    async getCentralityHeatmap() {
        return this.request('/api/centrality-heatmap', { method: 'GET' });
    }

    async getTimelineData() {
        return this.request('/api/timeline-data', { method: 'GET' });
    }

    // --- Alerts ---
    async getAlerts() {
        return this.request('/api/alerts', { method: 'GET' });
    }

    // --- Upload APIs ---
    async uploadData(formData) {
        return fetch(`${this.baseURL}/api/upload_data`, {
            method: 'POST',
            body: formData
        }).then(r => r.json());
    }

    async getAccumulationStatus() {
        return this.request('/api/accumulation_status', { method: 'GET' });
    }

    // --- Locations ---
    async getLocations() {
        return this.request('/api/locations', { method: 'GET' });
    }

    // --- Top Senders/Receivers ---
    async getTopSenders() {
        return this.request('/api/top_senders', { method: 'GET' });
    }

    async getTopReceivers() {
        return this.request('/api/top_receivers', { method: 'GET' });
    }

    // --- Chatbot ---
    async sendChatMessage(message) {
        return this.request('/api/chatbot', {
            method: 'POST',
            body: JSON.stringify({ message })
        });
    }

    // --- Prediction ---
    async predict(companyId) {
        return this.request('/api/predict', {
            method: 'POST',
            body: JSON.stringify({ company_id: companyId })
        });
    }

    // --- Health check (simple ping) ---
    async getHealth() {
        try {
            const resp = await fetch(`${this.baseURL}/api/statistics`);
            return { status: resp.ok ? 'healthy' : 'unhealthy', timestamp: new Date().toISOString() };
        } catch (e) {
            return { status: 'unhealthy', timestamp: new Date().toISOString() };
        }
    }
}

// Create global API instance
const api = new APIClient();
window.api = api;

// Backwards compatibility helper for older inline code
if (typeof window.apiCall === 'undefined') {
    window.apiCall = async function(endpoint, options = {}) { return api.request(endpoint, options); };
}

// Auto-detect backend baseURL: try port 5000 first (Flask), then 8000 (FastAPI)
async function detectBackendBaseURL(timeoutMs = 2000) {
    // Prioritize port 5000 since Flask backend runs there
    const candidates = ['http://localhost:5000', 'http://127.0.0.1:5000', 'http://localhost:8000', 'http://127.0.0.1:8000'];
    for (const base of candidates) {
        try {
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), timeoutMs);
            const resp = await fetch(`${base}/api/statistics`, { signal: controller.signal });
            clearTimeout(id);
            if (resp.ok) {
                console.info('Backend detected at', base);
                return base;
            }
        } catch (e) {
            console.debug('Backend not available at', base);
            // ignore and try next
        }
    }
    console.warn('No backend detected, using default port 5000');
    return 'http://localhost:5000'; // fallback to Flask default
}

// Run detection on load and set api.baseURL accordingly
api._readyPromise = (async () => {
    try {
        const detected = await detectBackendBaseURL(1500);
        api.setBaseURL(detected);
        api._ready = true;
        console.info('API base URL set to', detected);
    } catch (e) {
        api._ready = true;
        console.warn('Backend auto-detect failed, using default', api.baseURL);
    }
})();

// --- Utility Functions ---
function formatNumber(num) { 
    if (num === null || num === undefined) return '0'; 
    return Math.round(num).toLocaleString('en-IN'); 
}

function formatCurrency(amount) { 
    return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(amount); 
}

function formatDate(date) { 
    if (!date) return '-'; 
    return new Date(date).toLocaleDateString('en-IN'); 
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Unknown';
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now - date) / 1000);
    
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
    return `${Math.floor(seconds / 86400)} days ago`;
}

function getRiskLevelBadge(score) {
    if (score >= 0.7) return { level: 'High', class: 'high', color: '#DC3545' };
    if (score >= 0.4) return { level: 'Medium', class: 'medium', color: '#E8A838' };
    return { level: 'Low', class: 'low', color: '#28A745' };
}

function validateGSTIN(gstin) {
    const pattern = /^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$/;
    return pattern.test(gstin);
}

function validateEmail(email) {
    const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return pattern.test(email);
}

function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `toast ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 3000;
        animation: slideIn 0.3s ease-in-out;
    `;

    const bgColors = {
        success: '#28A745',
        error: '#DC3545',
        warning: '#E8A838',
        info: '#2F4156',
    };

    notification.style.backgroundColor = bgColors[type] || bgColors.info;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease-in-out';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

function showSpinner(container) {
    if (!container) return;
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.innerHTML = `
        <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
            <div style="
                border: 4px solid #EEF1F6;
                border-top: 4px solid #2F4156;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            "></div>
        </div>
    `;
    container.innerHTML = '';
    container.appendChild(spinner);
}

function hideSpinner() {
    const spinner = document.querySelector('.spinner');
    if (spinner) spinner.remove();
}

function isAuthenticated() {
    return !!localStorage.getItem('access_token');
}

function requireAuth() {
    // For now, allow access without auth for demo purposes
    // if (!isAuthenticated()) {
    //     window.location.href = '/login.html';
    // }
}

function getCurrentUser() {
    const userJSON = localStorage.getItem('current_user');
    return userJSON ? JSON.parse(userJSON) : { username: 'Admin', role: 'admin', full_name: 'Administrator' };
}

function setCurrentUser(user) {
    localStorage.setItem('current_user', JSON.stringify(user));
}

function logoutUser() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('current_user');
    window.location.href = '/login.html';
}

// Export functions for global use
window.api = api;
window.showNotification = showNotification;
window.showSpinner = showSpinner;
window.hideSpinner = hideSpinner;
window.formatNumber = formatNumber;
window.formatCurrency = formatCurrency;
window.formatDate = formatDate;
window.formatTimeAgo = formatTimeAgo;
window.getRiskLevelBadge = getRiskLevelBadge;
window.validateGSTIN = validateGSTIN;
window.validateEmail = validateEmail;
window.isAuthenticated = isAuthenticated;
window.requireAuth = requireAuth;
window.getCurrentUser = getCurrentUser;
window.setCurrentUser = setCurrentUser;
window.logoutUser = logoutUser;
