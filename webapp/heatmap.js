// Heatmap visualization using Plotly.js

function initializeHeatmap() {
    // Add mode change listener
    const modeSelect = document.getElementById('heatmap-mode');
    if (modeSelect) {
        modeSelect.addEventListener('change', updateHeatmap);
    }
    updateHeatmap();
}

function updateHeatmap() {
    if (!filteredData) return;

    const container = document.getElementById('heatmap');
    container.innerHTML = '<div class="loading">Loading heatmap...</div>';

    // Get heatmap mode
    const mode = document.getElementById('heatmap-mode')?.value || 'features';

    // Process data into matrix format based on mode
    const processedData = mode === 'prompts'
        ? processPromptHeatmapData(filteredData)
        : processHeatmapData(filteredData);

    if (!processedData) {
        container.innerHTML = '<div class="loading">No data available for this mode. Try selecting a single manifold for prompt mode.</div>';
        return;
    }

    const { matrix, features, xLabels, hoverText } = processedData;

    // Create plotly heatmap
    const trace = {
        z: matrix,
        x: xLabels,
        y: features,
        type: 'heatmap',
        colorscale: [
            [0, '#0d1117'],
            [0.2, '#1c2128'],
            [0.4, '#30363d'],
            [0.6, '#58a6ff'],
            [0.8, '#79c0ff'],
            [1.0, '#a5d6ff']
        ],
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        colorbar: {
            title: 'Activation',
            titleside: 'right',
            tickmode: 'linear',
            tick0: 0,
            dtick: 1,
            bgcolor: '#161b22',
            bordercolor: '#30363d',
            tickfont: { color: '#e6edf3' },
            titlefont: { color: '#e6edf3' }
        }
    };

    const xAxisTitle = mode === 'prompts' ? 'Prompts' : 'Manifolds';
    const heatmapTitle = mode === 'prompts'
        ? 'Feature × Prompt Activation Heatmap'
        : 'Feature × Manifold Activation Heatmap';

    const layout = {
        title: {
            text: heatmapTitle,
            font: { color: '#e6edf3', size: 16 }
        },
        xaxis: {
            title: xAxisTitle,
            tickangle: -45,
            side: 'bottom',
            color: '#e6edf3',
            gridcolor: '#30363d',
            tickfont: { size: mode === 'prompts' ? 8 : 10 }
        },
        yaxis: {
            title: 'Features (SAE_Type_Index)',
            color: '#e6edf3',
            gridcolor: '#30363d',
            autorange: 'reversed'
        },
        plot_bgcolor: '#0d1117',
        paper_bgcolor: '#161b22',
        font: { color: '#e6edf3' },
        margin: { l: 150, r: 50, t: 80, b: 100 },
        height: 800
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        displaylogo: false
    };

    container.innerHTML = '';
    Plotly.newPlot('heatmap', [trace], layout, config);

    // Add click handler
    document.getElementById('heatmap').on('plotly_click', function(data) {
        const point = data.points[0];
        const feature = point.y;
        const manifold = point.x;

        // Show detailed information
        showHeatmapDetail(feature, manifold);
    });
}

function processHeatmapData(data) {
    // Group by feature and manifold
    const featureManifoldMap = new Map();
    const manifoldsSet = new Set();
    const featuresSet = new Set();

    data.rows.forEach(row => {
        const featureId = `${row.sae_type}_${row.feature_idx}`;
        const manifold = row.manifold_name;

        manifoldsSet.add(manifold);
        featuresSet.add(featureId);

        const key = `${featureId}|${manifold}`;
        if (!featureManifoldMap.has(key)) {
            featureManifoldMap.set(key, {
                total: 0,
                count: 0,
                max: 0,
                prompts: []
            });
        }

        const entry = featureManifoldMap.get(key);
        entry.total += row.activation_value;
        entry.count += 1;
        entry.max = Math.max(entry.max, row.activation_value);
        entry.prompts.push({
            prompt: row.prompt,
            activation: row.activation_value
        });
    });

    // Sort features and manifolds
    const features = Array.from(featuresSet).sort();
    const manifolds = Array.from(manifoldsSet).sort();

    // Build matrix
    const matrix = [];
    const hoverText = [];

    features.forEach(feature => {
        const row = [];
        const hoverRow = [];

        manifolds.forEach(manifold => {
            const key = `${feature}|${manifold}`;
            const entry = featureManifoldMap.get(key);

            if (entry) {
                const avgActivation = entry.total / entry.count;
                row.push(avgActivation);

                // Create hover text
                const topPrompts = entry.prompts
                    .sort((a, b) => b.activation - a.activation)
                    .slice(0, 3)
                    .map(p => `  • ${p.prompt.substring(0, 50)}... (${p.activation.toFixed(3)})`)
                    .join('<br>');

                hoverRow.push(
                    `<b>${feature}</b> on <b>${manifold}</b><br>` +
                    `Avg: ${avgActivation.toFixed(3)}<br>` +
                    `Max: ${entry.max.toFixed(3)}<br>` +
                    `Count: ${entry.count}<br>` +
                    `<br>Top prompts:<br>${topPrompts}`
                );
            } else {
                row.push(0);
                hoverRow.push(`<b>${feature}</b> on <b>${manifold}</b><br>No activations`);
            }
        });

        matrix.push(row);
        hoverText.push(hoverRow);
    });

    return { matrix, features, xLabels: manifolds, hoverText };
}

function processPromptHeatmapData(data) {
    // For prompt mode, we need a specific manifold selected
    const manifoldFilter = document.getElementById('manifold-filter')?.value;

    if (!manifoldFilter || manifoldFilter === 'all') {
        return null; // Require a specific manifold for prompt mode
    }

    // Group by feature and prompt
    const featurePromptMap = new Map();
    const promptsSet = new Set();
    const featuresSet = new Set();

    data.rows.forEach(row => {
        if (row.manifold_name !== manifoldFilter) return;

        const featureId = `${row.sae_type}_${row.feature_idx}`;
        const prompt = row.prompt;

        promptsSet.add(prompt);
        featuresSet.add(featureId);

        const key = `${featureId}|${prompt}`;
        if (!featurePromptMap.has(key)) {
            featurePromptMap.set(key, {
                activation: row.activation_value,
                sae_type: row.sae_type,
                feature_idx: row.feature_idx
            });
        } else {
            // Keep max activation if multiple entries
            const existing = featurePromptMap.get(key);
            if (row.activation_value > existing.activation) {
                existing.activation = row.activation_value;
            }
        }
    });

    // Sort features and prompts
    const features = Array.from(featuresSet).sort();
    const prompts = Array.from(promptsSet);

    // Truncate prompt labels for display
    const promptLabels = prompts.map(p => {
        const maxLen = 40;
        return p.length > maxLen ? p.substring(0, maxLen) + '...' : p;
    });

    // Build matrix
    const matrix = [];
    const hoverText = [];

    features.forEach(feature => {
        const row = [];
        const hoverRow = [];

        prompts.forEach(prompt => {
            const key = `${feature}|${prompt}`;
            const entry = featurePromptMap.get(key);

            if (entry) {
                row.push(entry.activation);
                hoverRow.push(
                    `<b>${feature}</b><br>` +
                    `Activation: ${entry.activation.toFixed(3)}<br>` +
                    `<br>Prompt:<br>${prompt}`
                );
            } else {
                row.push(0);
                hoverRow.push(`<b>${feature}</b><br>No activation<br><br>${prompt}`);
            }
        });

        matrix.push(row);
        hoverText.push(hoverRow);
    });

    return { matrix, features, xLabels: promptLabels, hoverText };
}

function showHeatmapDetail(feature, manifold) {
    // Filter data for this specific feature-manifold combination
    const [saeType, featureIdx] = feature.split('_');
    const rows = filteredData.rows.filter(r =>
        r.sae_type === saeType &&
        r.feature_idx === parseInt(featureIdx) &&
        r.manifold_name === manifold
    );

    if (rows.length === 0) return;

    // Sort by activation
    rows.sort((a, b) => b.activation_value - a.activation_value);

    // Create modal or detailed view
    const content = `
        <h3>Feature ${featureIdx} (${saeType}) on ${manifold}</h3>
        <p><strong>Total Activations:</strong> ${rows.length}</p>
        <p><strong>Avg Activation:</strong> ${(rows.reduce((s, r) => s + r.activation_value, 0) / rows.length).toFixed(3)}</p>
        <p><strong>Max Activation:</strong> ${rows[0].activation_value.toFixed(3)}</p>
        <br>
        <p><strong>Top Prompts:</strong></p>
        ${rows.slice(0, 10).map(r =>
            `<p style="margin: 0.5rem 0; padding: 0.5rem; background: #0d1117; border-radius: 4px;">
                <strong>${r.activation_value.toFixed(3)}</strong> - ${r.prompt}
            </p>`
        ).join('')}
    `;

    // Create a temporary modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 2rem;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        z-index: 10000;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    `;
    modal.innerHTML = content;

    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: none;
        border: none;
        color: #8b949e;
        font-size: 2rem;
        cursor: pointer;
        padding: 0;
        width: 2rem;
        height: 2rem;
        line-height: 1;
    `;
    closeBtn.onclick = () => {
        modal.remove();
        backdrop.remove();
    };
    modal.appendChild(closeBtn);

    // Add backdrop
    const backdrop = document.createElement('div');
    backdrop.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 9999;
    `;
    backdrop.onclick = () => {
        modal.remove();
        backdrop.remove();
    };

    document.body.appendChild(backdrop);
    document.body.appendChild(modal);
}
