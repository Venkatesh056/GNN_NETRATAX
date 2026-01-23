// ============================================================================
// D3.js Timeline Visualization - Navy Color Palette
// ============================================================================

class Timeline {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.margin = options.margin || { top: 50, right: 50, bottom: 50, left: 80 };
        this.width = (options.width || this.container.clientWidth) - this.margin.left - this.margin.right;
        this.height = (options.height || this.container.clientHeight || 400) - this.margin.top - this.margin.bottom;
        
        // Navy color palette
        this.colors = {
            navy: '#2F4156',
            crimson: '#700034',
            warmSand: '#C3A582',
            softPearl: '#F7F2F0',
            obsidian: '#1B1616',
            highRisk: '#700034',
            mediumRisk: '#C3A582',
            lowRisk: '#2F4156'
        };
        
        this.svg = null;
        this.data = [];
    }
    
    init() {
        // Clear existing SVG
        d3.select(`#${this.containerId}`).selectAll('*').remove();
        
        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    }
    
    loadData(data) {
        // Data format: [{ date: Date, value: number, risk: number, label: string }]
        this.data = data.sort((a, b) => a.date - b.date);
        this.render();
    }
    
    render() {
        // Create scales
        const x = d3.scaleTime()
            .domain(d3.extent(this.data, d => d.date))
            .range([0, this.width]);
        
        const y = d3.scaleLinear()
            .domain([0, d3.max(this.data, d => d.value)])
            .nice()
            .range([this.height, 0]);
        
        // Add X axis
        this.svg.append('g')
            .attr('transform', `translate(0,${this.height})`)
            .call(d3.axisBottom(x).ticks(6))
            .selectAll('text')
            .attr('fill', this.getThemeTextColor())
            .style('font-size', '11px');
        
        // Add Y axis
        this.svg.append('g')
            .call(d3.axisLeft(y))
            .selectAll('text')
            .attr('fill', this.getThemeTextColor())
            .style('font-size', '11px');
        
        // Add grid lines
        this.svg.append('g')
            .attr('class', 'grid')
            .attr('opacity', 0.1)
            .call(d3.axisLeft(y)
                .tickSize(-this.width)
                .tickFormat(''));
        
        // Create line generator
        const line = d3.line()
            .x(d => x(d.date))
            .y(d => y(d.value))
            .curve(d3.curveMonotoneX);
        
        // Add gradient for area
        const gradient = this.svg.append('defs')
            .append('linearGradient')
            .attr('id', 'timeline-gradient')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '0%')
            .attr('y2', '100%');
        
        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', this.colors.navy)
            .attr('stop-opacity', 0.6);
        
        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', this.colors.navy)
            .attr('stop-opacity', 0.1);
        
        // Add area
        const area = d3.area()
            .x(d => x(d.date))
            .y0(this.height)
            .y1(d => y(d.value))
            .curve(d3.curveMonotoneX);
        
        this.svg.append('path')
            .datum(this.data)
            .attr('fill', 'url(#timeline-gradient)')
            .attr('d', area);
        
        // Add line
        this.svg.append('path')
            .datum(this.data)
            .attr('fill', 'none')
            .attr('stroke', this.colors.navy)
            .attr('stroke-width', 3)
            .attr('d', line);
        
        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'timeline-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background-color', this.getThemeBackgroundColor())
            .style('color', this.getThemeTextColor())
            .style('border', `2px solid ${this.colors.navy}`)
            .style('border-radius', '8px')
            .style('padding', '12px')
            .style('font-size', '13px')
            .style('pointer-events', 'none')
            .style('z-index', '1000')
            .style('box-shadow', '0 4px 12px rgba(0,0,0,0.2)');
        
        // Add data points
        this.svg.selectAll('.dot')
            .data(this.data)
            .enter().append('circle')
            .attr('class', 'dot')
            .attr('cx', d => x(d.date))
            .attr('cy', d => y(d.value))
            .attr('r', 6)
            .attr('fill', d => this.getRiskColor(d.risk))
            .attr('stroke', this.colors.obsidian)
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 10);
                
                tooltip
                    .style('visibility', 'visible')
                    .html(`
                        <strong>${d.label || 'Event'}</strong><br/>
                        Date: ${d.date.toLocaleDateString()}<br/>
                        Value: ${d.value.toLocaleString()}<br/>
                        Risk: ${(d.risk * 100).toFixed(1)}%
                    `);
            })
            .on('mousemove', function(event) {
                tooltip
                    .style('top', (event.pageY - 10) + 'px')
                    .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 6);
                
                tooltip.style('visibility', 'hidden');
            });
        
        // Add title
        this.svg.append('text')
            .attr('x', this.width / 2)
            .attr('y', -20)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .attr('fill', this.getThemeTextColor())
            .text('Fraud Detection Timeline');
        
        // Add Y axis label
        this.svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -60)
            .attr('x', -this.height / 2)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .attr('fill', this.getThemeTextColor())
            .text('Transaction Volume');
    }
    
    getRiskColor(risk) {
        if (risk > 0.7) return this.colors.highRisk;
        if (risk > 0.4) return this.colors.mediumRisk;
        return this.colors.lowRisk;
    }
    
    getThemeTextColor() {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        return isDark ? this.colors.softPearl : this.colors.obsidian;
    }
    
    getThemeBackgroundColor() {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        return isDark ? this.colors.obsidian : this.colors.softPearl;
    }
    
    updateTheme() {
        // Update colors when theme changes
        this.svg.selectAll('text')
            .attr('fill', this.getThemeTextColor());
        
        d3.selectAll('.timeline-tooltip')
            .style('background-color', this.getThemeBackgroundColor())
            .style('color', this.getThemeTextColor());
    }
    
    resize() {
        this.width = this.container.clientWidth - this.margin.left - this.margin.right;
        this.height = (this.container.clientHeight || 400) - this.margin.top - this.margin.bottom;
        
        // Re-render with new dimensions
        this.init();
        if (this.data.length > 0) {
            this.render();
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Timeline;
}
