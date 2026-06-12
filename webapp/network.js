// Network graph visualization using D3.js force simulation

let simulation = null;
let svg = null;
let g = null;

function initializeNetworkGraph() {
    const container = document.getElementById('network-graph');
    container.innerHTML = ''; // Clear existing

    const width = container.clientWidth;
    const height = 600;

    // Create SVG
    svg = d3.select('#network-graph')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height]);

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Container for graph elements
    g = svg.append('g');

    // Create legend
    createLegend();

    // Initial render
    updateNetworkGraph();
}

function updateNetworkGraph() {
    if (!filteredData || !g) return;

    // Process data into nodes and links
    const { nodes, links } = processNetworkData(filteredData);

    // Clear existing elements
    g.selectAll('*').remove();

    // Create force simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(svg.attr('width') / 2, svg.attr('height') / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 5));

    // Create links
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => Math.sqrt(d.value) * 0.5);

    // Create nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('r', d => d.radius)
        .attr('fill', d => d.color)
        .attr('stroke', '#30363d')
        .attr('stroke-width', 2)
        .call(drag(simulation))
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke', '#58a6ff')
                .attr('stroke-width', 3);

            showTooltip(createNodeTooltip(d), event.pageX, event.pageY);
        })
        .on('mouseout', function() {
            d3.select(this)
                .attr('stroke', '#30363d')
                .attr('stroke-width', 2);

            hideTooltip();
        })
        .on('mousemove', function(event) {
            const tooltip = document.getElementById('tooltip');
            tooltip.style.left = event.pageX + 10 + 'px';
            tooltip.style.top = event.pageY + 10 + 'px';
        });

    // Add labels
    const labels = g.append('g')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .text(d => d.label)
        .attr('font-size', 10)
        .attr('fill', '#8b949e')
        .attr('text-anchor', 'middle')
        .attr('dy', d => d.radius + 12)
        .style('pointer-events', 'none');

    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        labels
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
}

function processNetworkData(data) {
    const nodes = [];
    const links = [];
    const nodeMap = new Map();

    // Create manifold nodes
    const manifolds = [...new Set(data.rows.map(r => r.manifold_name))];
    manifolds.forEach(manifold => {
        const id = `manifold_${manifold}`;
        nodes.push({
            id,
            label: manifold,
            type: 'manifold',
            color: manifoldColor,
            radius: 15,
            data: { manifold }
        });
        nodeMap.set(id, nodes.length - 1);
    });

    // Create feature nodes and links
    const featureGroups = d3.group(data.rows, d => `${d.sae_type}_${d.feature_idx}`);

    featureGroups.forEach((rows, featureId) => {
        const firstRow = rows[0];
        const id = `feature_${featureId}`;

        // Calculate average activation for this feature
        const avgActivation = d3.mean(rows, r => r.activation_value);
        const totalActivations = rows.length;

        nodes.push({
            id,
            label: `F${firstRow.feature_idx}`,
            type: 'feature',
            color: saeColors[firstRow.sae_type] || '#8b949e',
            radius: Math.min(20, 8 + Math.sqrt(totalActivations)),
            data: {
                sae_type: firstRow.sae_type,
                feature_idx: firstRow.feature_idx,
                avg_activation: avgActivation,
                total_activations: totalActivations,
                top_prompts: rows
                    .sort((a, b) => b.activation_value - a.activation_value)
                    .slice(0, 5)
            }
        });
        nodeMap.set(id, nodes.length - 1);

        // Create links to manifolds
        const manifoldGroups = d3.group(rows, r => r.manifold_name);
        manifoldGroups.forEach((manifoldRows, manifold) => {
            const totalActivation = d3.sum(manifoldRows, r => r.activation_value);
            links.push({
                source: id,
                target: `manifold_${manifold}`,
                value: totalActivation,
                count: manifoldRows.length
            });
        });
    });

    return { nodes, links };
}

function createNodeTooltip(node) {
    if (node.type === 'manifold') {
        return `
            <h3>Manifold: ${node.data.manifold}</h3>
            <p>Type: Manifold</p>
        `;
    } else {
        const prompts = node.data.top_prompts.map(p =>
            `<p style="margin: 0.5rem 0; padding: 0.5rem; background: #0d1117; border-radius: 4px;">
                <strong>${p.activation_value.toFixed(3)}</strong> - ${p.prompt.substring(0, 60)}${p.prompt.length > 60 ? '...' : ''}
            </p>`
        ).join('');

        return `
            <h3>Feature ${node.data.feature_idx} (${node.data.sae_type})</h3>
            <p>Avg Activation: ${node.data.avg_activation.toFixed(3)}</p>
            <p>Total Activations: ${node.data.total_activations}</p>
            <p style="margin-top: 0.5rem; font-weight: bold;">Top Prompts:</p>
            ${prompts}
        `;
    }
}

function createLegend() {
    const legend = document.getElementById('network-legend');
    legend.innerHTML = '';

    // Manifold legend
    const manifoldItem = document.createElement('div');
    manifoldItem.className = 'legend-item';
    manifoldItem.innerHTML = `
        <div class="legend-color" style="background: ${manifoldColor};"></div>
        <span>Manifolds</span>
    `;
    legend.appendChild(manifoldItem);

    // SAE type legends
    Object.entries(saeColors).forEach(([type, color]) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="legend-color" style="background: ${color};"></div>
            <span>${type}</span>
        `;
        legend.appendChild(item);
    });
}

function drag(simulation) {
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
