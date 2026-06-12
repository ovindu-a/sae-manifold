// Main application logic
let data = null;
let filteredData = null;

// Color schemes
const saeColors = {
    'batchtopk': '#58a6ff',
    'gated': '#f778ba',
    'jumprelu': '#79c0ff',
    'matryoshka': '#ffa657',
    'subspace': '#a5d6ff'
};

const manifoldColor = '#7ee787';

// Load data
async function loadData() {
    try {
        const response = await fetch('/data');
        data = await response.json();

        if (!data || data.error) {
            showError(data?.error || 'Failed to load data');
            return;
        }

        filteredData = data;

        // Populate filters
        populateFilters();

        // Initialize all views
        initializeNetworkGraph();
        initializeHeatmap();
        initializePromptExplorer();

        // Update statistics
        updateStatistics();

    } catch (error) {
        showError('Error loading data: ' + error.message);
    }
}

function populateFilters() {
    // Default values
    const DEFAULT_SAE = 'subspace';
    const DEFAULT_MANIFOLD = 'days';

    // SAE types
    const saeFilter = document.getElementById('sae-filter');
    const saeTypes = [...new Set(data.rows.map(r => r.sae_type))].sort();
    saeTypes.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        saeFilter.appendChild(option);
    });

    // Set default SAE if available
    if (saeTypes.includes(DEFAULT_SAE)) {
        saeFilter.value = DEFAULT_SAE;
    }

    // Manifolds
    const manifoldFilter = document.getElementById('manifold-filter');
    const manifolds = [...new Set(data.rows.map(r => r.manifold_name))].sort();
    manifolds.forEach(manifold => {
        const option = document.createElement('option');
        option.value = manifold;
        option.textContent = manifold;
        manifoldFilter.appendChild(option);
    });

    // Set default manifold if available
    if (manifolds.includes(DEFAULT_MANIFOLD)) {
        manifoldFilter.value = DEFAULT_MANIFOLD;
    }

    // Apply the default filters
    applyFilters();
}

function applyFilters() {
    const saeType = document.getElementById('sae-filter').value;
    const threshold = parseFloat(document.getElementById('threshold-filter').value);
    const manifold = document.getElementById('manifold-filter').value;

    filteredData = {
        ...data,
        rows: data.rows.filter(row => {
            if (saeType !== 'all' && row.sae_type !== saeType) return false;
            if (row.activation_value < threshold) return false;
            if (manifold !== 'all' && row.manifold_name !== manifold) return false;
            return true;
        })
    };

    // Update all views
    updateNetworkGraph();
    updateHeatmap();
    updatePromptExplorer();
    updateStatistics();
}

function updateStatistics() {
    if (!filteredData) return;

    const features = new Set(filteredData.rows.map(r => `${r.sae_type}_${r.feature_idx}`));
    const manifolds = new Set(filteredData.rows.map(r => r.manifold_name));
    const avgActivation = filteredData.rows.reduce((sum, r) => sum + r.activation_value, 0) / filteredData.rows.length;

    document.getElementById('stat-features').textContent = features.size;
    document.getElementById('stat-manifolds').textContent = manifolds.size;
    document.getElementById('stat-connections').textContent = filteredData.rows.length;
    document.getElementById('stat-avg-activation').textContent = avgActivation.toFixed(3);
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;

        // Update active tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update active content
        document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tabName}-content`).classList.add('active');

        // Trigger resize for plotly charts
        if (tabName === 'heatmap') {
            Plotly.Plots.resize('heatmap');
        }
    });
});

// Filter change listeners
document.getElementById('sae-filter').addEventListener('change', applyFilters);
document.getElementById('threshold-filter').addEventListener('input', applyFilters);
document.getElementById('manifold-filter').addEventListener('change', applyFilters);

function showError(message) {
    const error = document.createElement('div');
    error.className = 'error';
    error.textContent = message;
    document.body.insertBefore(error, document.body.firstChild);
}

// Tooltip functions
function showTooltip(content, x, y) {
    const tooltip = document.getElementById('tooltip');
    tooltip.innerHTML = content;
    tooltip.style.left = x + 10 + 'px';
    tooltip.style.top = y + 10 + 'px';
    tooltip.classList.add('show');
}

function hideTooltip() {
    document.getElementById('tooltip').classList.remove('show');
}

// Initialize on load
loadData();
