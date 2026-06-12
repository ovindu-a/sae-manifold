# SAE Feature Activation Explorer

Interactive web application for visualizing SAE feature activations across manifold datasets.

## Features

### 1. **Network Graph View**
- Interactive force-directed graph showing connections between features and manifolds
- Node size represents activation frequency
- Edges weighted by activation strength
- Hover over nodes to see top activating prompts
- Drag nodes to rearrange the layout
- Color-coded by SAE type

### 2. **Heatmap View**
- Matrix visualization of Features × Manifolds
- Color intensity represents activation strength
- Click cells to see detailed prompt information
- Sortable and filterable
- Hoverable cells show statistics and top prompts

### 3. **Prompt Explorer View**
- Browse individual prompts and see which features they activate
- Filter by SAE type and manifold
- Search prompts by text
- Sort by max activation, feature count, or manifold
- Click prompts to see detailed feature breakdown
- Features grouped and color-coded by SAE type

## Quick Start

### 1. Generate activation data

First, run the analysis script to generate the CSV file:

```bash
cd /Users/ovindu/Desktop/Repos/fyp/sae-manifold
python analyze_sae_activations.py --output sae_feature_activations.csv
```

### 2. Start the web server

```bash
cd webapp
python server.py --csv ../sae_feature_activations.csv --port 8000
```

### 3. Open in browser

Navigate to: **http://localhost:8000**

## Usage

### Global Filters (Apply to all views)

- **SAE Type**: Filter by specific SAE architecture (batchtopk, gated, jumprelu, matryoshka, subspace)
- **Min Activation**: Set minimum activation threshold
- **Manifold**: Filter by specific manifold dataset

### View-Specific Controls

**Network Graph:**
- Drag nodes to rearrange
- Zoom and pan with mouse/trackpad
- Hover for detailed tooltips

**Heatmap:**
- Click cells for detailed prompt view
- Hover for quick statistics
- Color scale shows activation intensity

**Prompt Explorer:**
- Additional SAE type filter specific to this view
- Search box for text matching
- Sort by activation strength, feature count, or manifold
- Click prompt cards for detailed breakdown

## Server Options

```bash
python server.py --help

Options:
  --csv PATH       Path to CSV file (default: ../sae_feature_activations.csv)
  --port PORT      Port number (default: 8000)
  --host HOST      Host address (default: localhost)
```

## File Structure

```
webapp/
├── index.html       # Main HTML structure and styles
├── app.js          # Core application logic and data loading
├── network.js      # Network graph visualization (D3.js)
├── heatmap.js      # Heatmap visualization (Plotly)
├── prompts.js      # Prompt explorer view
├── server.py       # Python HTTP server
└── README.md       # This file
```

## Technology Stack

- **Frontend**: Vanilla JavaScript (no build step required)
- **Visualization**: D3.js (network graph), Plotly.js (heatmap)
- **Backend**: Python HTTP server (built-in `http.server`)
- **Styling**: Custom CSS with GitHub-inspired dark theme

## Performance Notes

- Large CSV files (>100MB) may take a few seconds to load
- Consider using `--top-n` in `analyze_sae_activations.py` to reduce file size
- The webapp filters data client-side for instant updates
- Network graph performance degrades with >1000 nodes (use filters to reduce)

## Keyboard Shortcuts

- **Tab**: Switch between views (when focused on tabs)
- **Escape**: Close modal dialogs
- **Ctrl/Cmd + F**: Focus search box (in Prompt Explorer)

## Color Scheme

Each SAE type has a distinct color:
- **batchtopk**: Blue (#58a6ff)
- **gated**: Pink (#f778ba)
- **jumprelu**: Light Blue (#79c0ff)
- **matryoshka**: Orange (#ffa657)
- **subspace**: Pale Blue (#a5d6ff)
- **manifolds**: Green (#7ee787)

## Troubleshooting

**Server won't start:**
- Check that the CSV file exists at the specified path
- Make sure port 8000 is not already in use
- Try a different port: `python server.py --port 8080`

**No data appears:**
- Check browser console for errors (F12)
- Verify CSV file format matches expected schema
- Make sure filters aren't excluding all data

**Performance issues:**
- Reduce data size with `--top-n` parameter
- Use filters to show fewer connections
- Close other browser tabs

**Visualization not rendering:**
- Check that D3.js and Plotly.js are loading from CDN
- Try refreshing the page (Ctrl/Cmd + Shift + R)
- Check browser console for JavaScript errors

## Examples

### Filter for high-activation features
Set "Min Activation" to 2.0 to see only strongly-activating features

### Explore a specific SAE
Select SAE type "batchtopk" to see only those features

### Find features for a specific concept
In Prompt Explorer, search for keywords like "color" or "age"

### Compare SAE architectures
Use Network Graph to see which SAE types have more connections to manifolds

## Citation

If you use this tool in your research, please cite the paper:

```
Do Sparse Autoencoders Capture Concept Manifolds?
[Paper details here]
```
