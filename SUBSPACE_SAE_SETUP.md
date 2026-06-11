# SubspaceSAE Default Directions Setup

This document explains how to set up default directions for SubspaceSAE training.

## Quick Start

### 1. Create Default Directions File

```bash
# Create default directions from multiple manifolds
python create_multi_directions.py \
    --manifolds years colors temperature \
    --n-components 5 5 3 \
    --set-as-default
```

This creates `cache/default_directions.npy` with:
- 5 directions from years manifold
- 5 directions from colors manifold
- 3 directions from temperature manifold
- **Total: 13 pinned features**

### 2. Train SubspaceSAE (No Extra Arguments Needed!)

```bash
# Now this works automatically - no --directions needed!
uv run train_sae.py --sae-type subspace

# Or train all SAE types including subspace
uv run train_sae.py
```

The training script will automatically find and use `cache/default_directions.npy`.

## How It Works

### Priority Order

When training a subspace SAE, `train_sae.py` looks for directions in this order:

1. **`--directions <file>`** - Explicit file path (highest priority)
2. **`--directions-manifold <name>`** - Compute PCA on-the-fly
3. **`cache/default_directions.npy`** - Default file (if exists)
4. **Error** - If none of the above exist

### Creating Default Directions

#### Option 1: From Multiple Manifolds (Recommended)

```bash
# Equal components from each
python create_multi_directions.py \
    --manifolds years colors temperature geography \
    --n-components 5 5 5 5 \
    --set-as-default

# Different components per manifold
python create_multi_directions.py \
    --manifolds years colors \
    --n-components 10 5 \
    --set-as-default
```

#### Option 2: From Single Manifold

```bash
# Use PCA from just one manifold
uv run train_sae.py \
    --sae-type subspace \
    --directions-manifold years \
    --directions-n-components 10

# Then save those directions for reuse
python -c "
from subspace_capture import get_manifold_pca_directions
import numpy as np
dirs, _ = get_manifold_pca_directions('years', n_components=10)
np.save('cache/default_directions.npy', dirs)
"
```

## Overriding Defaults

Even with a default file, you can override it:

```bash
# Use different directions for this run
uv run train_sae.py \
    --sae-type subspace \
    --directions my_custom_directions.npy

# Or compute fresh PCA
uv run train_sae.py \
    --sae-type subspace \
    --directions-manifold colors \
    --directions-n-components 8
```

## Checking Current Default

```bash
# Check if default directions file exists
ls -lh cache/default_directions.npy

# View shape and info
python -c "
import numpy as np
d = np.load('cache/default_directions.npy')
print(f'Shape: {d.shape}')
print(f'Pinned features: {d.shape[0]}')
print(f'Model dimension: {d.shape[1]}')
"
```

## Example Workflow

```bash
# 1. Extract manifold data
uv run data.py --manifold years colors temperature

# 2. Create default directions (one time setup)
python create_multi_directions.py \
    --manifolds years colors temperature \
    --n-components 5 5 3 \
    --set-as-default

# 3. Train all SAEs (including subspace) - no extra args!
uv run train_sae.py

# 4. Evaluate
uv run subspace_capture.py plot
```

## Benefits

1. **Convenience**: No need to specify directions every time
2. **Consistency**: All experiments use the same baseline directions
3. **Flexibility**: Easy to override when needed
4. **Reproducibility**: Default file is version-controlled

## File Location

- **Default file**: `cache/default_directions.npy`
- **Format**: NumPy array of shape `[n_dirs, d_in]` (e.g., `[13, 4096]`)
- **Git**: Add to `.gitignore` or commit for reproducibility
