// 3D Concept Space — one subview per manifold, plotted in top-3 PCA components

let pcaData = null;          // full /pca_data payload
let currentConcept = null;   // name of the manifold currently shown

// ── Colour helpers ──────────────────────────────────────────────────────────

// For continuous labels: interpolate between two hex colours
function lerpColor(hex1, hex2, t) {
    const parse = h => [
        parseInt(h.slice(1, 3), 16),
        parseInt(h.slice(3, 5), 16),
        parseInt(h.slice(5, 7), 16),
    ];
    const [r1, g1, b1] = parse(hex1);
    const [r2, g2, b2] = parse(hex2);
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `rgb(${r},${g},${b})`;
}

// Pick the most informative label key for colouring points
function pickColorKey(points) {
    if (!points.length || !points[0].label) return null;
    const keys = Object.keys(points[0].label);
    if (!keys.length) return null;

    // Prefer numeric keys (continuous colour scale looks nicer)
    const numericKeys = keys.filter(k => typeof points[0].label[k] === 'number');
    if (numericKeys.length) return numericKeys[0];
    return keys[0];
}

// Build a Plotly colour array + colorbar config from labels
function buildColorConfig(points, colorKey) {
    if (!colorKey) {
        return { marker_color: '#58a6ff', colorbar: null, colorscale: null };
    }

    const values = points.map(p => p.label[colorKey]);
    const isNumeric = typeof values[0] === 'number';

    if (isNumeric) {
        return {
            marker_color: values,
            colorscale: 'Plasma',
            colorbar: {
                title: colorKey.replace(/_/g, ' '),
                thickness: 14,
                bgcolor: '#161b22',
                bordercolor: '#30363d',
                tickfont: { color: '#e6edf3', size: 10 },
                titlefont: { color: '#e6edf3', size: 11 },
            },
            showscale: true,
        };
    }

    // Categorical — assign a fixed colour per unique value
    const unique = [...new Set(values)].sort();
    const palette = [
        '#58a6ff', '#f778ba', '#79c0ff', '#ffa657', '#7ee787',
        '#a5d6ff', '#d2a8ff', '#ff7b72', '#ffa198', '#56d364',
    ];
    const colorMap = Object.fromEntries(unique.map((v, i) => [v, palette[i % palette.length]]));
    return {
        marker_color: values.map(v => colorMap[v]),
        colorscale: null,
        colorbar: null,
        showscale: false,
        legendMap: colorMap,
    };
}

// ── Initialise the view ─────────────────────────────────────────────────────

async function initializePCAView() {
    try {
        const resp = await fetch('/pca_data');
        if (!resp.ok) {
            document.getElementById('pca-plot').innerHTML =
                `<div class="loading">PCA data not available.<br>Run: <code>python generate_pca_data.py</code></div>`;
            return;
        }
        pcaData = await resp.json();
    } catch (e) {
        document.getElementById('pca-plot').innerHTML =
            `<div class="loading">Could not load PCA data: ${e.message}</div>`;
        return;
    }

    // Build concept-tab buttons
    const tabBar = document.getElementById('pca-concept-tabs');
    tabBar.innerHTML = '';
    const concepts = Object.keys(pcaData).sort();

    concepts.forEach((name, idx) => {
        const btn = document.createElement('button');
        btn.className = 'concept-tab-btn' + (idx === 0 ? ' active' : '');
        btn.textContent = name;
        btn.dataset.concept = name;
        btn.addEventListener('click', () => {
            document.querySelectorAll('.concept-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderConcept(name);
        });
        tabBar.appendChild(btn);
    });

    // Render the first concept
    if (concepts.length) renderConcept(concepts[0]);
}

// ── Render one concept ───────────────────────────────────────────────────────

function renderConcept(name) {
    currentConcept = name;
    const info = pcaData[name];
    if (!info) return;

    const { points, variance_explained } = info;

    // Stats bar
    const statsEl = document.getElementById('pca-stats');
    statsEl.style.display = '';
    document.getElementById('pca-concept-title').textContent = `Concept: ${name}`;

    document.getElementById('pca-stats-grid').innerHTML = [
        { label: 'Samples', value: points.length },
        { label: 'PC1 var explained', value: (variance_explained[0] * 100).toFixed(1) + '%' },
        { label: 'PC2 var explained', value: (variance_explained[1] * 100).toFixed(1) + '%' },
        { label: 'PC3 var explained', value: variance_explained[2] !== undefined ? (variance_explained[2] * 100).toFixed(1) + '%' : 'n/a' },
        { label: 'Total (PC1–3)', value: (variance_explained.slice(0, 3).reduce((a, b) => a + b, 0) * 100).toFixed(1) + '%' },
    ].map(s => `
        <div class="stat-item">
            <div class="stat-label">${s.label}</div>
            <div class="stat-value" style="font-size:1.1rem;">${s.value}</div>
        </div>
    `).join('');

    // Data
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const zs = points.map(p => p.z);
    const texts = points.map(p => buildHoverText(p, name));

    const colorKey = pickColorKey(points);
    const colorCfg = buildColorConfig(points, colorKey);

    const trace = {
        type: 'scatter3d',
        mode: 'markers',
        x: xs,
        y: ys,
        z: zs,
        text: texts,
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            size: 4,
            opacity: 0.85,
            color: colorCfg.marker_color,
            ...(colorCfg.colorscale ? { colorscale: colorCfg.colorscale } : {}),
            ...(colorCfg.colorbar ? { colorbar: colorCfg.colorbar } : {}),
            ...(colorCfg.showscale !== undefined ? { showscale: colorCfg.showscale } : {}),
            line: { width: 0 },
        },
    };

    // For categorical labels, add a legend via separate traces
    let traces = [trace];
    if (colorCfg.legendMap) {
        const colorKey2 = colorKey;
        const groups = {};
        points.forEach((p, i) => {
            const val = p.label[colorKey2];
            if (!groups[val]) groups[val] = { xs: [], ys: [], zs: [], texts: [] };
            groups[val].xs.push(p.x);
            groups[val].ys.push(p.y);
            groups[val].zs.push(p.z);
            groups[val].texts.push(texts[i]);
        });

        traces = Object.entries(groups).map(([val, g]) => ({
            type: 'scatter3d',
            mode: 'markers',
            name: String(val),
            x: g.xs,
            y: g.ys,
            z: g.zs,
            text: g.texts,
            hovertemplate: '%{text}<extra></extra>',
            marker: {
                size: 4,
                opacity: 0.85,
                color: colorCfg.legendMap[val],
                line: { width: 0 },
            },
        }));
    }

    const layout = {
        scene: {
            xaxis: {
                title: `PC1 (${(variance_explained[0] * 100).toFixed(1)}%)`,
                color: '#8b949e',
                gridcolor: '#30363d',
                backgroundcolor: '#0d1117',
                showbackground: true,
            },
            yaxis: {
                title: `PC2 (${(variance_explained[1] * 100).toFixed(1)}%)`,
                color: '#8b949e',
                gridcolor: '#30363d',
                backgroundcolor: '#0d1117',
                showbackground: true,
            },
            zaxis: {
                title: variance_explained[2] !== undefined
                    ? `PC3 (${(variance_explained[2] * 100).toFixed(1)}%)`
                    : 'PC3',
                color: '#8b949e',
                gridcolor: '#30363d',
                backgroundcolor: '#0d1117',
                showbackground: true,
            },
            bgcolor: '#0d1117',
            camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
        },
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        font: { color: '#e6edf3', size: 12 },
        margin: { l: 0, r: 0, t: 40, b: 0 },
        height: 600,
        legend: {
            bgcolor: '#1c2128',
            bordercolor: '#30363d',
            font: { color: '#e6edf3', size: 10 },
        },
        title: {
            text: `${name} — prompts in activation PCA space`,
            font: { color: '#e6edf3', size: 14 },
            x: 0.5,
        },
    };

    Plotly.react('pca-plot', traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage'],
    });
}

function buildHoverText(point, concept) {
    const labelLines = Object.entries(point.label)
        .map(([k, v]) => {
            const val = typeof v === 'number' ? v.toFixed(3) : v;
            return `<b>${k}:</b> ${val}`;
        })
        .join('<br>');

    const promptLine = point.prompt.length > 80
        ? point.prompt.substring(0, 80) + '…'
        : point.prompt;

    return `<i>${promptLine}</i>${labelLines ? '<br>' + labelLines : ''}`;
}
