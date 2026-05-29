# SAE Manifold Evaluation

Code accompanying **"Do Sparse Autoencoders Capture Concept Manifolds?"**
([arxiv.org/abs/2604.28119](https://arxiv.org/abs/2604.28119))
— *Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, Boppana, Kowal, Shyam,
Lewis, McGrath, Merullo, Geiger, Lubana*.

This repository releases two of the evaluations from the paper:

1. **Subspace capture** (`subspace_capture.py`) — measures whether an SAE's
   features span a fixed low-dimensional subspace that contains a given
   concept manifold, or whether they *shatter* into many disjoint atoms
   across the manifold.
2. **Unsupervised feature clustering** (`unsupervised_clustering.py`) —
   groups SAE features by their pairwise relationships (decoder cosine,
   co-activation, correlation, mutual information, inverse Ising couplings),
   then runs Leiden / spectral community detection.

The implementation targets Llama-3.1-8B by default, but the model
and layer can be overridden in `data.py`. SAE encoding is done with a
minimal `BatchTopKSAE` class in `saes.py`; if you trained with a different
architecture (TopK, JumpReLU, Matryoshka, …) you can swap in your own class
— the rest of the code only depends on three methods (`encode`, `decode`,
decoder weight access).

## Quickstart

### 1. Extract manifold activations

Each "manifold" is a set of prompts whose ground-truth structure is known.
We push them through the LM once and cache the last-token hidden state.

```bash
uv run data.py                    # extract all manifolds
uv run data.py --manifold colors  # or just one
```

Cached activations live in `cache/{manifold}.pt`.

### 2. Train SAEs

Train all four SAE architectures in one command. Checkpoints are saved to
`cache/saes/{type}.pt`.

```bash
uv run train_sae.py
```

This trains `batchtopk`, `gated`, `jumprelu`, and `matryoshka` sequentially
with default hyperparameters (k=64, 20 epochs, expansion factor 4×).

**Train a subset of types:**
```bash
uv run train_sae.py --sae-type batchtopk matryoshka
```

**Common overrides:**
```bash
uv run train_sae.py \
    --sae-type batchtopk gated jumprelu matryoshka \
    --k 64 \
    --epochs 20 \
    --output-dir cache/saes
```

| Flag | Default | Description |
|---|---|---|
| `--sae-type` | all four | Space-separated list of types to train |
| `--output-dir` | `cache/saes/` | Where to write `{type}.pt` checkpoints |
| `--k` | `64` | Target sparsity (top-k for BatchTopK/Matryoshka, target L0 for JumpReLU) |
| `--matryoshka-ks` | `[k//4, k//2, k]` | k levels for Matryoshka multi-resolution loss |
| `--l1-weight` | `1e-3` for gated/jumprelu, else `0` | L1 sparsity penalty |
| `--manifold` | all | Restrict training data to specific manifolds |
| `--epochs` | `20` | Training epochs per SAE |
| `--output-dir` | `cache/saes/` | Checkpoint directory |

### 3. Run subspace-capture experiments

#### Compare all SAEs (auto-discovers `cache/saes/*.pt`)

```bash
uv run subspace_capture.py plot
```

Produces one PDF per manifold in `cache/subspace_capture/`, with all four
SAEs overlaid. Each SAE gets a distinct colour; solid line = geometric greedy
(decoder directions), dashed line = statistical greedy (actual SAE codes).
PCA and random baselines are shown for reference.

#### Exclude specific SAEs

```bash
uv run subspace_capture.py plot --ignore gated jumprelu
```

#### Use a custom checkpoint folder or explicit paths

```bash
# Point at a different folder
uv run subspace_capture.py plot --sae-dir cache/experiment_1

# Explicit paths (original behaviour)
uv run subspace_capture.py plot \
    --sae cache/saes/batchtopk.pt cache/saes/matryoshka.pt

# Custom display labels
uv run subspace_capture.py plot \
    --sae cache/saes/batchtopk.pt cache/saes/gated.pt \
    --sae-labels "BatchTopK" "Gated"
```

#### Feature tuning curves

```bash
# Uses the first discovered SAE by default
uv run subspace_capture.py tuning --manifold years

# Or pick one explicitly
uv run subspace_capture.py tuning --sae cache/saes/batchtopk.pt --manifold years
```

### 4. Run unsupervised clustering

```bash
# (a) Extract background activations from a streaming text dataset.
uv run background.py --n-tokens 500000

# (b) Compute all five similarity matrices (cosine, coactivation,
#     correlation, MI, Ising).
uv run unsupervised_clustering.py matrices \
    --sae cache/saes/batchtopk.pt --n-tokens 500000

# (c) Cluster features. Ising is recommended (Sec. 5 of the paper).
uv run unsupervised_clustering.py cluster \
    --sae cache/saes/batchtopk.pt --matrix ising --method leiden

# Optional: refit Ising with a tuned alpha or on a feature subset.
uv run unsupervised_clustering.py ising \
    --sae cache/saes/batchtopk.pt --alpha 0.005

# Optional: heatmap of a specific feature subset across all matrices.
uv run unsupervised_clustering.py heatmap \
    --sae cache/saes/batchtopk.pt --features 100 200 300 400
```

### Full pipeline — minimal commands

```bash
uv run data.py                       # 1. extract manifold activations
uv run train_sae.py                  # 2. train all four SAE types → cache/saes/
uv run subspace_capture.py plot      # 3. comparison chart → cache/subspace_capture/
```

## Manifolds shipped

| Manifold | Structure | Source |
|---|---|---|
| `colors` | paraboloid (hue, lightness, saturation) | `burkelibbey/colors` |
| `temperature` | line | synthetic |
| `age` | line | synthetic |
| `geography` | hierarchical tree (lat/long + continent) | `jamescalam/world-cities-geo` |
| `days` | 2D grid (7 days × 8 times of day) | synthetic |
| `years` | helix (1800–1999) | synthetic |
| `formality` | line | `osyvokon/pavlick-formality-scores` |
| `sent_length` | line | `wikitext` |

## SAE format

`saes.load_sae(path, ...)` accepts either a raw `state_dict` (as saved by
`torch.save(sae.state_dict(), path)`) or a dict with `state_dict` and
`model_config` keys. It expects an encoder linear, a decoder linear, and
an optional per-feature `threshold` buffer for JumpReLU-style inference.

If you trained with a different architecture, replace `BatchTopKSAE` with
your class — `subspace_capture.py` and `unsupervised_clustering.py` only
use `sae.encode`, `sae.decode`, `get_decoder(sae)`, and `get_decoder_bias(sae)`.

## Citation

```bibtex
@article{bhalla2026sae,
  title   = {Do Sparse Autoencoders Capture Concept Manifolds?},
  author  = {Bhalla, Usha and Fel, Thomas and Rager, Can and Feucht, Sheridan
             and Haklay, Tal and Wurgaft, Daniel and Boppana, Siddharth
             and Kowal, Matthew and Shyam, Vasudev and Lewis, Owen
             and McGrath, Thomas and Merullo, Jack and Geiger, Atticus
             and Lubana, Ekdeep Singh},
  journal = {arXiv preprint arXiv:2604.28119},
  year    = {2026},
}
```

## License

MIT. See `LICENSE`.
