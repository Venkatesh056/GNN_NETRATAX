// ============================================================================
// D3.js Heatmap Visualization - Navy Color Palette
// ============================================================================

class Heatmap {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.margin = options.margin || { top: 50, right: 50, bottom: 100, left: 100 };
        this.width = (options.width || this.container.clientWidth) - this.margin.left - this.margin.right;
        this.height = (options.height || this.container.clientHeight || 500) - this.margin.top - this.margin.bottom;
        
        // Navy color palette
        this.colors = {
            navy: '#2F4156',
            crimson: '#700034',
            warmSand: '#C3A582',
            softPearl: '#F7F2F0',
            obsidian: '#1B1616'
        };
        
        this.svg = null;
        this.data = [];
        this.xLabels = [];
        this.yLabels = [];
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
    
    loadData(data, xLabels, yLabels) {
        this.data = data;
        this.xLabels = xLabels;
        this.yLabels = yLabels;
        this.render();
    }
    
    render() {
        // Build X scales and axis
        const x = d3.scaleBand()
            .range([0, this.width])
            .domain(this.xLabels)
            .padding(0.05);
        
        this.svg.append('g')
            .style('font-size', 12)
            .attr('transform', `translate(0,${this.height})`)
            .call(d3.axisBottom(x).tickSize(0))
            .select('.domain').remove();
        
        // Rotate x-axis labels
        this.svg.selectAll('.tick text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end')
            .attr('fill', this.getThemeTextColor());
        
        // Build Y scales and axis
        const y = d3.scaleBand()
            .range([this.height, 0])
            .domain(this.yLabels)
            .padding(0.05);
        
        this.svg.append('g')
            .style('font-size', 12)
            .call(d3.axisLeft(y).tickSize(0))
            .select('.domain').remove();
        
        this.svg.selectAll('.tick text')
            .attr('fill', this.getThemeTextColor());
        
        // Build color scale
        const colorScale = d3.scaleLinear()
            .range([this.colors.softPearl, this.colors.warmSand, this.colors.crimson])
            .domain([0, 0.5, 1]);
        
        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'heatmap-tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('background-color', this.getThemeBackgroundColor())
            .style('color', this.getThemeTextColor())
            .style('border', `1px solid ${this.colors.navy}`)
            .style('border-radius', '8px')
            .style('padding', '10px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', '1000');
        
        // Add the squares
        this.svg.selectAll()
            .data(this.data, d => `${d.x}:${d.y}`)
            .enter()
            .append('rect')
            .attr('x', d => x(d.x))
            .attr('y', d => y(d.y))
            .attr('rx', 4)
            .attr('ry', 4)
            .attr('width', x.bandwidth())
            .attr('height', y.bandwidth())
            .style('fill', d => colorScale(d.value))
            .style('stroke-width', 2)
            .style('stroke', 'none')
            .style('opacity', 0.8)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .style('stroke', this.colors.navy)
                    .style('opacity', 1);
                
                tooltip
                    .style('visibility', 'visible')
                    .html(`
                        <strong>${d.x} → ${d.y}</strong><br/>
                        Risk Score: ${(d.value * 100).toFixed(1)}%
                    `);
            }.bind(this))
            .on('mousemove', function(event) {
                tooltip
                    .style('top', (event.pageY - 10) + 'px')
                    .style('left', (event.pageX + 10) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this)
                    .style('stroke', 'none')
                    .style('opacity', 0.8);
                
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
            .text('Transaction Risk Heatmap');
        
        // Add legend
        this.addLegend(colorScale);
    }
    
    addLegend(colorScale) {
        const legendWidth = 200;
        const legendHeight = 20;
        
        const legend = this.svg.append('g')
            .attr('transform', `translate(${this.width - legendWidth}, ${-40})`);
        
        // Create gradient for legend
        const defs = this.svg.append('defs');
        const linearGradient = defs.append('linearGradient')
            .attr('id', 'legend-gradient');
        
        linearGradient.selectAll('stop')
            .data([
                { offset: '0%', color: this.colors.softPearl },
                { offset: '50%', color: this.colors.warmSand },
                { offset: '100%', color: this.colors.crimson }
            ])
            .enter().append('stop')
            .attr('offset', d => d.offset)
            .attr('stop-color', d => d.color);
        
        // Draw legend rectangle
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#legend-gradient)')
            .attr('rx', 4);
        
        // Add legend labels
        legend.append('text')
            .attr('x', 0)
            .attr('y', legendHeight + 15)
            .style('font-size', '11px')
            .attr('fill', this.getThemeTextColor())
            .text('Low Risk');
        
        legend.append('text')
            .attr('x', legendWidth)
            .attr('y', legendHeight + 15)
            .attr('text-anchor', 'end')
            .style('font-size', '11px')
            .attr('fill', this.getThemeTextColor())
            .text('High Risk');
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
        
        d3.selectAll('.heatmap-tooltip')
            .style('background-color', this.getThemeBackgroundColor())
            .style('color', this.getThemeTextColor());
    }
    
    resize() {
        this.width = this.container.clientWidth - this.margin.left - this.margin.right;
        this.height = (this.container.clientHeight || 500) - this.margin.top - this.margin.bottom;
        
        // Re-render with new dimensions
        this.init();
        if (this.data.length > 0) {
            this.render();
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Heatmap;
}
