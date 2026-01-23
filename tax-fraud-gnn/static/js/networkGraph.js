// ============================================================================
// D3.js Network Graph Visualization - Navy Color Palette
// ============================================================================

class NetworkGraph {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.width = options.width || this.container.clientWidth;
        this.height = options.height || this.container.clientHeight || 600;
        
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
            link: 'rgba(47, 65, 86, 0.3)'
        };
        
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
    }
    
    init() {
        // Clear existing SVG
        d3.select(`#${this.containerId}`).selectAll('*').remove();
        
        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height]);
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('g').attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group
        this.mainGroup = this.svg.append('g');
        
        // Add arrow markers for directed edges
        this.svg.append('defs').selectAll('marker')
            .data(['end'])
            .enter().append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', this.colors.link);
    }
    
    loadData(graphData) {
        this.nodes = graphData.nodes || [];
        this.links = graphData.links || [];
        
        // Initialize force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links)
                .id(d => d.id)
                .distance(100))
            .force('charge', d3.forceManyBody()
                .strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        this.render();
    }
    
    render() {
        // Draw links
        const link = this.mainGroup.append('g')
            .selectAll('line')
            .data(this.links)
            .enter().append('line')
            .attr('stroke', this.colors.link)
            .attr('stroke-width', d => Math.sqrt(d.value || 1))
            .attr('marker-end', 'url(#arrow)');
        
        // Draw nodes
        const node = this.mainGroup.append('g')
            .selectAll('circle')
            .data(this.nodes)
            .enter().append('circle')
            .attr('r', d => this.getNodeSize(d))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', this.colors.obsidian)
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .call(this.drag(this.simulation));
        
        // Add node labels
        const label = this.mainGroup.append('g')
            .selectAll('text')
            .data(this.nodes)
            .enter().append('text')
            .text(d => d.label || d.id)
            .attr('font-size', 10)
            .attr('dx', 15)
            .attr('dy', 4)
            .attr('fill', this.getThemeTextColor())
            .style('pointer-events', 'none');
        
        // Add tooltips
        node.append('title')
            .text(d => `${d.label || d.id}\nRisk: ${(d.risk * 100).toFixed(1)}%`);
        
        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
        
        // Add hover effects
        node.on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', d => this.getNodeSize(d) * 1.5)
                .attr('stroke-width', 4);
        }.bind(this))
        .on('mouseout', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', d => this.getNodeSize(d))
                .attr('stroke-width', 2);
        }.bind(this));
    }
    
    getNodeSize(node) {
        // Size based on degree or importance
        const baseSize = 8;
        const scaleFactor = node.degree || 1;
        return baseSize + Math.sqrt(scaleFactor) * 2;
    }
    
    getNodeColor(node) {
        // Color based on risk level
        const risk = node.risk || 0;
        if (risk > 0.7) return this.colors.highRisk;
        if (risk > 0.4) return this.colors.mediumRisk;
        return this.colors.lowRisk;
    }
    
    getThemeTextColor() {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        return isDark ? this.colors.softPearl : this.colors.obsidian;
    }
    
    drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
    
    updateTheme() {
        // Update colors when theme changes
        this.mainGroup.selectAll('text')
            .attr('fill', this.getThemeTextColor());
    }
    
    resize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight || 600;
        
        this.svg
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height]);
        
        if (this.simulation) {
            this.simulation
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .alpha(0.3)
                .restart();
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetworkGraph;
}
