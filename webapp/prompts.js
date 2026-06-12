// Prompt explorer visualization - shows individual prompts and their activating features

let currentPromptFilters = {
    saeType: 'all',
    manifold: 'all',
    search: ''
};

function initializePromptExplorer() {
    // Add prompt-specific controls
    const promptContent = document.getElementById('prompt-content');

    // Check if controls already exist
    if (!document.getElementById('prompt-sae-filter')) {
        const controls = document.createElement('div');
        controls.style.cssText = 'display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;';
        controls.innerHTML = `
            <div class="control-group">
                <label for="prompt-sae-filter">SAE Type:</label>
                <select id="prompt-sae-filter">
                    <option value="all">All</option>
                </select>
            </div>
            <div class="control-group">
                <label for="prompt-manifold-filter">Manifold:</label>
                <select id="prompt-manifold-filter">
                    <option value="all">All</option>
                </select>
            </div>
            <div class="control-group">
                <label for="prompt-sort">Sort by:</label>
                <select id="prompt-sort">
                    <option value="activation">Max Activation</option>
                    <option value="features">Feature Count</option>
                    <option value="manifold">Manifold</option>
                </select>
            </div>
        `;

        promptContent.insertBefore(controls, promptContent.firstChild);

        // Populate SAE types
        if (data) {
            const DEFAULT_SAE = 'subspace';
            const DEFAULT_MANIFOLD = 'days';

            const saeTypes = [...new Set(data.rows.map(r => r.sae_type))].sort();
            const saeFilter = document.getElementById('prompt-sae-filter');
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

            // Populate manifolds
            const manifolds = [...new Set(data.rows.map(r => r.manifold_name))].sort();
            const manifoldFilter = document.getElementById('prompt-manifold-filter');
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
        }

        // Add event listeners
        document.getElementById('prompt-sae-filter').addEventListener('change', updatePromptExplorer);
        document.getElementById('prompt-manifold-filter').addEventListener('change', updatePromptExplorer);
        document.getElementById('prompt-sort').addEventListener('change', updatePromptExplorer);
        document.getElementById('prompt-search').addEventListener('input', updatePromptExplorer);
    }

    updatePromptExplorer();
}

function updatePromptExplorer() {
    if (!data) return;

    // Get filter values
    currentPromptFilters.saeType = document.getElementById('prompt-sae-filter')?.value || 'all';
    currentPromptFilters.manifold = document.getElementById('prompt-manifold-filter')?.value || 'all';
    currentPromptFilters.search = document.getElementById('prompt-search')?.value.toLowerCase() || '';
    const sortBy = document.getElementById('prompt-sort')?.value || 'activation';

    // Process data - group by prompt
    // Use original data, not filteredData, for the prompt explorer
    const promptMap = new Map();

    data.rows.forEach(row => {
        // Apply prompt-specific filters
        if (currentPromptFilters.saeType !== 'all' && row.sae_type !== currentPromptFilters.saeType) {
            return;
        }
        if (currentPromptFilters.manifold !== 'all' && row.manifold_name !== currentPromptFilters.manifold) {
            return;
        }
        if (currentPromptFilters.search && !row.prompt.toLowerCase().includes(currentPromptFilters.search)) {
            return;
        }

        const key = `${row.manifold_name}|${row.prompt}`;

        if (!promptMap.has(key)) {
            promptMap.set(key, {
                prompt: row.prompt,
                manifold: row.manifold_name,
                features: []
            });
        }

        promptMap.get(key).features.push({
            sae_type: row.sae_type,
            feature_idx: row.feature_idx,
            activation: row.activation_value
        });
    });

    // Convert to array and sort
    let prompts = Array.from(promptMap.values());

    // Sort features within each prompt by activation
    prompts.forEach(p => {
        p.features.sort((a, b) => b.activation - a.activation);
        p.maxActivation = p.features[0]?.activation || 0;
        p.featureCount = p.features.length;
    });

    // Sort prompts
    if (sortBy === 'activation') {
        prompts.sort((a, b) => b.maxActivation - a.maxActivation);
    } else if (sortBy === 'features') {
        prompts.sort((a, b) => b.featureCount - a.featureCount);
    } else if (sortBy === 'manifold') {
        prompts.sort((a, b) => a.manifold.localeCompare(b.manifold));
    }

    // Render prompt cards
    renderPromptCards(prompts);
}

function renderPromptCards(prompts) {
    const grid = document.getElementById('prompt-grid');
    grid.innerHTML = '';

    if (prompts.length === 0) {
        grid.innerHTML = '<div class="loading">No prompts match the current filters</div>';
        return;
    }

    prompts.forEach(prompt => {
        const card = createPromptCard(prompt);
        grid.appendChild(card);
    });
}

function createPromptCard(promptData) {
    const card = document.createElement('div');
    card.className = 'prompt-card';

    // Group features by SAE type
    const featuresBySAE = d3.group(promptData.features, f => f.sae_type);

    const saeGroups = Array.from(featuresBySAE.entries()).map(([saeType, features]) => {
        const badges = features.slice(0, 10).map(f => {
            const color = saeColors[saeType] || '#8b949e';
            return `
                <div class="feature-badge"
                     style="border-color: ${color}; color: ${color};"
                     title="Feature ${f.feature_idx}: ${f.activation.toFixed(3)}">
                    F${f.feature_idx}
                </div>
            `;
        }).join('');

        const moreCount = features.length > 10 ? features.length - 10 : 0;
        const moreText = moreCount > 0 ? `<div class="feature-badge" style="opacity: 0.6;">+${moreCount} more</div>` : '';

        return `
            <div style="margin-bottom: 0.5rem;">
                <div style="font-size: 0.7rem; color: #8b949e; margin-bottom: 0.25rem;">
                    ${saeType} (${features.length})
                </div>
                <div class="feature-badges">
                    ${badges}
                    ${moreText}
                </div>
            </div>
        `;
    }).join('');

    card.innerHTML = `
        <div class="prompt-text">"${promptData.prompt}"</div>
        <div class="prompt-meta">
            <strong>Manifold:</strong> ${promptData.manifold} |
            <strong>Max Activation:</strong> ${promptData.maxActivation.toFixed(3)} |
            <strong>Total Features:</strong> ${promptData.featureCount}
        </div>
        <div style="margin-top: 0.75rem;">
            ${saeGroups}
        </div>
    `;

    // Add click handler to show detailed view
    card.addEventListener('click', () => {
        showPromptDetail(promptData);
    });

    return card;
}

function showPromptDetail(promptData) {
    // Group features by SAE type for detailed view
    const featuresBySAE = d3.group(promptData.features, f => f.sae_type);

    const saeDetails = Array.from(featuresBySAE.entries()).map(([saeType, features]) => {
        const color = saeColors[saeType] || '#8b949e';
        const featureList = features.map(f => `
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: #0d1117; border-radius: 4px; margin: 0.25rem 0;">
                <span style="color: ${color};">Feature ${f.feature_idx}</span>
                <span style="font-weight: bold;">${f.activation.toFixed(4)}</span>
            </div>
        `).join('');

        return `
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: ${color}; margin-bottom: 0.5rem;">${saeType} (${features.length} features)</h4>
                <div style="max-height: 300px; overflow-y: auto;">
                    ${featureList}
                </div>
            </div>
        `;
    }).join('');

    const content = `
        <h3 style="margin-bottom: 1rem;">Prompt Detail</h3>
        <div style="background: #0d1117; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; font-style: italic;">
            "${promptData.prompt}"
        </div>
        <div style="margin-bottom: 1rem;">
            <p><strong>Manifold:</strong> ${promptData.manifold}</p>
            <p><strong>Total Features Activated:</strong> ${promptData.featureCount}</p>
            <p><strong>Max Activation:</strong> ${promptData.maxActivation.toFixed(4)}</p>
        </div>
        <h4 style="margin-bottom: 0.75rem;">Activated Features by SAE Type:</h4>
        ${saeDetails}
    `;

    // Create modal
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
        max-width: 700px;
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
