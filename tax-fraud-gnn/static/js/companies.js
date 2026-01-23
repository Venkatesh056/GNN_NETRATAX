// ============================================================================
// Tax Fraud Detection Companies Page - JavaScript
// Updated with Navy Color Palette
// ============================================================================

// Navy Color Palette
const NAVY_COLORS = {
    navy: '#2F4156',
    crimsonDepth: '#700034',
    warmSand: '#C3A582',
    softPearl: '#F7F2F0',
    obsidianBlack: '#1B1616'
};

let currentRiskThreshold = 0.5;
let currentLocations = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadLocations();
    loadCompanies();
    setupEventListeners();
});

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Risk threshold slider
    const riskSlider = document.getElementById('riskThreshold');
    const thresholdValue = document.getElementById('thresholdValue');

    if (riskSlider && thresholdValue) {
        riskSlider.addEventListener('input', function() {
            const value = parseFloat(this.value);
            currentRiskThreshold = value;
            thresholdValue.textContent = value.toFixed(2);
            // Update the slider value display immediately
            thresholdValue.style.color = 'var(--forsythia)';
        });
        
        // Initialize display value
        thresholdValue.textContent = riskSlider.value.toFixed(2);
    }

    // Location filter
    const locationFilter = document.getElementById('locationFilter');
    if (locationFilter) {
        locationFilter.addEventListener('change', function() {
            currentLocations = Array.from(this.selectedOptions).map(option => option.value).filter(v => v);
        });
    }

    // Apply filters button
    const applyButton = document.getElementById('applyFilters');
    if (applyButton) {
        applyButton.addEventListener('click', loadCompanies);
    }

    // Reset filters button
    const resetButton = document.getElementById('resetFilters');
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            document.getElementById('riskThreshold').value = 0.5;
            document.getElementById('thresholdValue').textContent = '0.50';
            document.getElementById('locationFilter').value = '';
            document.getElementById('companySearch').value = '';
            currentRiskThreshold = 0.5;
            currentLocations = [];
            loadCompanies();
        });
    }

    // Company search
    const searchInput = document.getElementById('companySearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                loadCompanies();
            }
        });
    }

    // Modal close - use event delegation for better reliability
    const modal = document.getElementById('companyModal');
    
    // Function to close modal
    function closeModal() {
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    // Close button click handler using event delegation
    document.addEventListener('click', function(event) {
        const target = event.target;
        if (target && (target.classList.contains('close') || target.textContent === '×' || target.textContent === '✕')) {
            event.preventDefault();
            event.stopPropagation();
            closeModal();
        }
    });

    // Close modal when clicking outside
    if (modal) {
        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                closeModal();
            }
        });
    }
    
    // Also add direct event listener as backup when modal is shown
    // This will be called when modal content is loaded
    function attachCloseListener() {
        const closeBtn = document.querySelector('#companyModal .close');
        if (closeBtn) {
            // Remove any existing listeners
            const newCloseBtn = closeBtn.cloneNode(true);
            closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
            
            // Add fresh event listener
            newCloseBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                closeModal();
            });
        }
    }
    
    // Attach listener initially
    attachCloseListener();
    
    // Make closeModal available globally for use in showCompanyDetail
    window.closeModal = closeModal;
}

// ============================================================================
// Load Functions
// ============================================================================

function loadLocations() {
    fetch('/api/locations')
        .then(response => response.json())
        .then(locations => {
            const select = document.getElementById('locationFilter');
            if (select) {
                locations.forEach(location => {
                    const option = document.createElement('option');
                    option.value = location;
                    option.textContent = location;
                    select.appendChild(option);
                });
            }
        })
        .catch(error => console.error('Error loading locations:', error));
}

function loadCompanies() {
    const searchId = document.getElementById('companySearch')?.value || '';
    const locationParam = currentLocations.length > 0 ? currentLocations.join(',') : '';

    const params = new URLSearchParams({
        risk_threshold: currentRiskThreshold,
        location: locationParam,
        search: searchId
    });

    fetch(`/api/companies?${params}`)
        .then(response => response.json())
        .then(companies => {
            displayCompanies(companies);
            loadStatistics();
        })
        .catch(error => console.error('Error loading companies:', error));
}

function loadStatistics() {
    fetch('/api/statistics')
        .then(response => response.json())
        .then(stats => {
            document.getElementById('totalCompanies').textContent = stats.total_companies;
            document.getElementById('highRiskCount').textContent = stats.high_risk_count;
            document.getElementById('mediumRiskCount').textContent = stats.medium_risk_count;
            document.getElementById('lowRiskCount').textContent = stats.low_risk_count;
        })
        .catch(error => console.error('Error loading statistics:', error));
}

// ============================================================================
// Display Functions
// ============================================================================

function displayCompanies(companies) {
    const tbody = document.getElementById('companiesTableBody');
    
    if (!tbody) return;

    if (companies.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No companies found</td></tr>';
        return;
    }

    tbody.innerHTML = companies.map(company => {
        // Escape company_id for use in JavaScript (handle string IDs)
        const companyId = typeof company.company_id === 'string' 
            ? `"${company.company_id.replace(/"/g, '\\"')}"` 
            : company.company_id;
        
        // Handle missing or unknown data with better formatting
        const companyName = company.company_name && company.company_name !== 'Unknown' 
            ? company.company_name 
            : `<span style="color: #C3A582; font-style: italic;">Company ${company.company_id}</span>`;
        
        const location = company.location && company.location !== 'Unknown' 
            ? company.location 
            : '<span style="color: #C3A582;">—</span>';
        
        const turnover = company.turnover && company.turnover !== 'Unknown' && typeof company.turnover === 'number'
            ? '₹' + company.turnover.toLocaleString('en-IN', {maximumFractionDigits: 0})
            : '<span style="color: #C3A582;">—</span>';
        
        const fraudProb = company.fraud_probability && typeof company.fraud_probability === 'number'
            ? (company.fraud_probability * 100).toFixed(1) + '%'
            : company.fraud_probability === 'Unknown' || !company.fraud_probability
            ? '<span style="color: #C3A582;">—</span>'
            : company.fraud_probability;
        
        const invoiceCount = company.sent_invoice_count || company.received_invoice_count || 0;
        const invoiceDisplay = invoiceCount > 0 
            ? invoiceCount 
            : '<span style="color: #C3A582;">0</span>';
        
        // Create tooltip content
        const tooltipContent = `
            <div class="company-tooltip">
                <div class="tooltip-header">
                    <strong>${company.company_id}</strong>
                    <span class="risk-badge ${getRiskClass(company.risk_level)}">${company.risk_level}</span>
                </div>
                <div class="tooltip-body">
                    <div class="tooltip-row">
                        <span class="tooltip-label">📍 Location:</span>
                        <span class="tooltip-value">${company.location || 'Unknown'}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">💰 Turnover:</span>
                        <span class="tooltip-value">${typeof company.turnover === 'number' ? '₹' + company.turnover.toLocaleString('en-IN') : 'Unknown'}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">⚠️ Fraud Risk:</span>
                        <span class="tooltip-value">${typeof company.fraud_probability === 'number' ? (company.fraud_probability * 100).toFixed(1) + '%' : 'Unknown'}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">📤 Sent Invoices:</span>
                        <span class="tooltip-value">${company.sent_invoice_count || 0}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">📥 Received Invoices:</span>
                        <span class="tooltip-value">${company.received_invoice_count || 0}</span>
                    </div>
                </div>
                <div class="tooltip-footer">
                    Click row to view full details
                </div>
            </div>
        `.replace(/\n/g, '').replace(/"/g, '&quot;');
        
        return `
        <tr class="company-row" 
            data-company-id="${company.company_id}"
            data-tooltip='${tooltipContent}'
            onclick="showCompanyDetail('${company.company_id.replace(/'/g, "\\'")}')">
            <td><strong>${company.company_id}</strong></td>
            <td>${companyName}</td>
            <td>${location}</td>
            <td>${turnover}</td>
            <td><strong>${fraudProb}</strong></td>
            <td><span class="risk-level ${getRiskClass(company.risk_level)}">${company.risk_level}</span></td>
            <td>${invoiceDisplay}</td>
            <td>
                <button class="btn btn-primary view-company-btn" 
                        style="padding: 6px 12px; font-size: 12px; cursor: pointer;"
                        data-company-id="${company.company_id}"
                        onclick="event.stopPropagation(); showCompanyDetail('${company.company_id.replace(/'/g, "\\'")}')">
                    👁️ View
                </button>
            </td>
        </tr>
    `;
    }).join('');
    
    // Initialize hover tooltips
    initCompanyTooltips();
    
    // Also add event listeners after rendering (backup method)
    tbody.querySelectorAll('.view-company-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const companyId = this.getAttribute('data-company-id');
            showCompanyDetail(companyId);
        });
    });
}

// ============================================================================
// Tooltip Functions
// ============================================================================

function initCompanyTooltips() {
    const rows = document.querySelectorAll('.company-row');
    let tooltipElement = document.getElementById('company-tooltip-container');
    
    // Create tooltip container if it doesn't exist
    if (!tooltipElement) {
        tooltipElement = document.createElement('div');
        tooltipElement.id = 'company-tooltip-container';
        tooltipElement.className = 'company-tooltip-container';
        document.body.appendChild(tooltipElement);
    }
    
    rows.forEach(row => {
        row.addEventListener('mouseenter', function(e) {
            const tooltipHTML = this.getAttribute('data-tooltip');
            tooltipElement.innerHTML = tooltipHTML;
            tooltipElement.style.display = 'block';
            tooltipElement.style.opacity = '0';
            
            // Position tooltip
            const rect = this.getBoundingClientRect();
            const tooltipRect = tooltipElement.getBoundingClientRect();
            
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
            let top = rect.top - tooltipRect.height - 10;
            
            // Adjust if tooltip goes off screen
            if (left < 10) left = 10;
            if (left + tooltipRect.width > window.innerWidth - 10) {
                left = window.innerWidth - tooltipRect.width - 10;
            }
            if (top < 10) {
                top = rect.bottom + 10;
                tooltipElement.classList.add('tooltip-below');
            } else {
                tooltipElement.classList.remove('tooltip-below');
            }
            
            tooltipElement.style.left = left + 'px';
            tooltipElement.style.top = top + window.scrollY + 'px';
            
            // Fade in
            setTimeout(() => {
                tooltipElement.style.opacity = '1';
            }, 10);
        });
        
        row.addEventListener('mouseleave', function() {
            tooltipElement.style.opacity = '0';
            setTimeout(() => {
                tooltipElement.style.display = 'none';
            }, 200);
        });
        
        row.addEventListener('mousemove', function(e) {
            // Optional: make tooltip follow cursor
            // Uncomment if you want this behavior
            /*
            const rect = this.getBoundingClientRect();
            const tooltipRect = tooltipElement.getBoundingClientRect();
            tooltipElement.style.left = (e.clientX - tooltipRect.width / 2) + 'px';
            tooltipElement.style.top = (e.clientY - tooltipRect.height - 20) + window.scrollY + 'px';
            */
        });
    });
}

function getRiskClass(riskLevel) {
    if (riskLevel.includes('HIGH')) return 'high';
    if (riskLevel.includes('MEDIUM')) return 'medium';
    return 'low';
}

// ============================================================================
// Modal Functions
// ============================================================================

function showCompanyDetail(companyId) {
    console.log('Loading company detail for:', companyId);
    
    // Ensure companyId is properly encoded
    const encodedId = encodeURIComponent(companyId);
    
    fetch(`/api/company/${encodedId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(company => {
            console.log('Company data received:', company);
            displayCompanyDetail(company);
            const modal = document.getElementById('companyModal');
            if (modal) {
                modal.style.display = 'block';
                // Re-attach close button listener after modal is shown
                setTimeout(function() {
                    const closeBtn = modal.querySelector('.close');
                    if (closeBtn) {
                        // Remove old listener and add new one
                        const newCloseBtn = closeBtn.cloneNode(true);
                        closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
                        newCloseBtn.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            if (window.closeModal) {
                                window.closeModal();
                            } else {
                                modal.style.display = 'none';
                            }
                        });
                    }
                }, 50);
            } else {
                console.error('Modal element not found');
            }
        })
        .catch(error => {
            console.error('Error loading company detail:', error);
            alert('Error loading company details: ' + error.message);
        });
}

function displayCompanyDetail(company) {
    const riskBadgeClass = company.risk_level.toLowerCase();
    const fraudStatus = company.predicted_fraud === 1 ? 'FRAUD' : 'NORMAL';

    const html = `
        <h2>Company #${company.company_id} - ${fraudStatus}</h2>
        
        <div class="modal-detail-row">
            <div class="modal-detail-item">
                <div class="modal-detail-label">Location</div>
                <div class="modal-detail-value">${company.location}</div>
            </div>
            <div class="modal-detail-item">
                <div class="modal-detail-label">Fraud Probability</div>
                <div class="modal-detail-value">
                    <span class="risk-badge ${riskBadgeClass}">${(company.fraud_probability * 100).toFixed(1)}%</span>
                </div>
            </div>
        </div>

        <div class="modal-detail-row">
            <div class="modal-detail-item">
                <div class="modal-detail-label">Risk Level</div>
                <div class="modal-detail-value">${company.risk_level}</div>
            </div>
            <div class="modal-detail-item">
                <div class="modal-detail-label">Turnover</div>
                <div class="modal-detail-value">₹${company.turnover.toLocaleString('en-IN')}</div>
            </div>
        </div>

        <div class="modal-detail-row">
            <div class="modal-detail-item">
                <div class="modal-detail-label">Invoices Sent</div>
                <div class="modal-detail-value">${company.sent_invoice_count}</div>
            </div>
            <div class="modal-detail-item">
                <div class="modal-detail-label">Invoices Received</div>
                <div class="modal-detail-value">${company.received_invoice_count}</div>
            </div>
        </div>

        <div class="modal-detail-row">
            <div class="modal-detail-item">
                <div class="modal-detail-label">Total Sent Amount</div>
                <div class="modal-detail-value">₹${company.total_sent_amount.toLocaleString('en-IN', {maximumFractionDigits: 0})}</div>
            </div>
            <div class="modal-detail-item">
                <div class="modal-detail-label">Total Received Amount</div>
                <div class="modal-detail-value">₹${company.total_received_amount.toLocaleString('en-IN', {maximumFractionDigits: 0})}</div>
            </div>
        </div>

        <div class="modal-detail-row">
            <div class="modal-detail-item">
                <div class="modal-detail-label">Outgoing Invoices</div>
                <div class="modal-detail-value">${company.outgoing_invoices}</div>
            </div>
            <div class="modal-detail-item">
                <div class="modal-detail-label">Incoming Invoices</div>
                <div class="modal-detail-value">${company.incoming_invoices}</div>
            </div>
        </div>
    `;

    document.getElementById('companyDetail').innerHTML = html;
}
