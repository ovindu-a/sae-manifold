"""
Subspace capture evaluation for SAE features on manifold datasets.

This implements the two reconstruction-curve experiments and the
feature-tuning visualization from "Do Sparse Autoencoders Capture Concept
Manifolds?":

  - **Geometric reconstruction** (``find_support_greedy``): greedy subspace
    pursuit over decoder directions. At each step, pick the decoder atom that
    best reduces the residual of the centered manifold activations.
  - **Statistical reconstruction** (``find_support_greedy_codes``): greedy
    selection using the actual SAE codes — at each step, pick the feature
    whose centered contribution ``(z_i - <z_i>) d_i`` most reduces the
    residual.

Compared against PCA (optimal linear) and random-baseline curves, these two
greedy methods produce Fig. 4 of the paper. The tuning-curve plot reproduces
Fig. 5 (years_recon_tuning).

Usage:
  uv run subspace_capture.py plot   --sae /path/to/sae.pt --k 64
  uv run subspace_capture.py tuning --sae /path/to/sae.pt --k 64 --manifold years
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

from data import (
    D_MODEL, DEVICE, CACHE_DIR,
    get_all_manifold_names, load_manifold_data,
)
from saes import load_sae, encode_sae, get_decoder

RESULTS_DIR = CACHE_DIR / "subspace_capture"


# ── Elbow detection (used by the greedy curves) ──────────────────────────────

def _detect_elbow(curve, min_k=1):
    """Maximum-distance-to-chord elbow detector on a monotone curve.

    Returns the index of the elbow point; used by the greedy methods below
    to suggest a cutoff when neither a variance threshold nor ``max_k`` is hit.
    """
    n = len(curve)
    if n <= min_k + 1:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(curve, dtype=float)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len
    dists = np.abs(np.cross(line_unit, p0 - np.column_stack([x, y])))
    dists[:min_k] = -1
    return int(np.argmax(dists))


# ── Geometric reconstruction (decoder directions) ────────────────────────────

def find_support_greedy(activations, decoder, max_k=100, var_threshold=0.95):
    """Greedy subspace pursuit over decoder directions.

    At each step adds the decoder atom whose direction captures the most
    remaining variance of the centered manifold activations. Returns the
    selected feature indices, the cumulative variance-explained curve, and
    the suggested elbow ``k``.
    """
    X = np.asarray(activations, dtype=np.float32)
    X = X - X.mean(0)
    total_ss = (X ** 2).sum()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    candidates = np.arange(decoder.shape[0])
    D_cand = decoder[candidates]
    d_norms_sq = (D_cand ** 2).sum(1)
    alive = d_norms_sq > 1e-10

    selected_local = []
    selected_global = []
    var_curve = []
    residual = X.copy()

    for _ in range(max_k):
        projections = residual @ D_cand.T
        scores = (projections ** 2).sum(0) / d_norms_sq.clip(1e-10)
        scores[~alive] = -np.inf
        for i in selected_local:
            scores[i] = -np.inf
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected_local.append(best)
        selected_global.append(int(candidates[best]))
        D_sel = decoder[selected_global]
        _, s, Vt = np.linalg.svd(D_sel, full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = X - (X @ basis.T) @ basis
        explained = 1.0 - (residual ** 2).sum() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected_global), var_curve, elbow_k


# ── Statistical reconstruction (actual SAE codes) ────────────────────────────

def find_support_greedy_codes(activations, sae, codes,
                              max_k=100, var_threshold=0.95):
    """Greedy selection by manifold variance explained from actual SAE codes.

    For each feature ``i``, its centered contribution to the reconstruction is
    ``(z_i - <z_i>) d_i``. At each step adds the feature whose contribution
    most reduces the residual.
    """
    X = np.asarray(activations, dtype=np.float32)
    X_c = X - X.mean(0)
    total_ss = (X_c ** 2).sum()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    Z = np.asarray(codes, dtype=np.float32)
    decoder = get_decoder(sae)

    candidates = np.where((Z > 0).any(0))[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([]), 0

    Z_c = Z[:, candidates] - Z[:, candidates].mean(0)
    D_cand = decoder[candidates]
    n_cand = len(candidates)
    z_c_sq = (Z_c ** 2).sum(0)
    d_sq = (D_cand ** 2).sum(1)
    contrib_ss = z_c_sq * d_sq

    selected, selected_local, var_curve = [], [], []
    residual = X_c.copy()
    alive = np.ones(n_cand, dtype=bool)

    for _ in range(max_k):
        cross = (residual @ D_cand.T) * Z_c
        scores = 2 * cross.sum(0) - contrib_ss
        scores[~alive] = -np.inf
        best_local = int(np.argmax(scores))
        if scores[best_local] <= 0:
            break
        selected_local.append(best_local)
        selected.append(int(candidates[best_local]))
        alive[best_local] = False
        S_local = np.array(selected_local)
        recon = Z_c[:, S_local] @ D_cand[S_local]
        residual = X_c - recon
        explained = 1.0 - (residual ** 2).sum() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected), var_curve, elbow_k


# ── Plotting: variance-explained curves ──────────────────────────────────────

# Distinct colors for up to 8 SAEs in the comparison plot.
_SAE_PALETTE = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
    '#bcbd22',  # yellow-green
]


def _infer_sae_label(sae_path):
    """Read sae_type from checkpoint config; fall back to the filename stem."""
    try:
        ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            t = ckpt.get("model_config", {}).get("sae_type", "")
            if t:
                return t
    except Exception:
        pass
    return Path(sae_path).stem


def plot_greedy_variance_curves(sae_paths, sae_labels=None, manifolds=None,
                                max_k=64, out_dir=None, **sae_kwargs):
    """Plot variance-explained curves for one or more SAEs on the same axes.

    For each SAE, two curves are drawn:
      - solid line: geometric greedy (decoder directions)
      - dashed line: statistical greedy (actual SAE codes)

    Baselines (PCA, random orthogonal, random overcomplete) are computed once
    per manifold and shown in neutral colors.

    Args:
        sae_paths: a single path string or a list of path strings
        sae_labels: optional list of display names (inferred from checkpoint if None)
        manifolds: list of manifold names to evaluate (default: all)
        max_k: maximum number of features to select
        out_dir: directory for output PDFs
        **sae_kwargs: forwarded to ``load_sae`` (d_in, d_sae, k, …)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(sae_paths, str):
        sae_paths = [sae_paths]
    if sae_labels is None:
        sae_labels = [_infer_sae_label(p) for p in sae_paths]

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    names = manifolds or get_all_manifold_names()

    def _avg_curves(curves):
        if not curves:
            return np.array([])
        ml = max(len(v) for v in curves)
        padded = np.array([
            np.pad(v, (0, ml - len(v)),
                   constant_values=v[-1] if len(v) > 0 else 0)
            for v in curves
        ])
        return padded.mean(0)

    for manifold in names:
        data = load_manifold_data(manifold)
        if data is None:
            print(f"  {manifold}: no cached data, skipping")
            continue
        acts = data['activations']
        acts_np = (acts.float().numpy()
                   if isinstance(acts, torch.Tensor) else acts)

        X_c = acts_np - acts_np.mean(0)
        d = X_c.shape[1]

        # ── Baselines (manifold-specific, SAE-independent) ─────────────────
        pca = PCA(n_components=min(max_k, *X_c.shape)).fit(X_c)
        vc_pca = np.cumsum(pca.explained_variance_ratio_)

        # Use the first SAE's atom count for the overcomplete random baseline.
        first_sae = load_sae(sae_paths[0], device=DEVICE, **sae_kwargs)
        n_atoms = get_decoder(first_sae).shape[0]
        k_max = max_k  # upper bound; tightened per-SAE below
        del first_sae

        n_seeds = 5
        print(f"  {manifold}: random orthogonal baseline...")
        vc_orth_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
            _, vc_r, _ = find_support_greedy(acts_np, Q, max_k=max_k, var_threshold=1.0)
            vc_orth_list.append(vc_r[:k_max])

        print(f"  {manifold}: random overcomplete baseline...")
        vc_over_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 100)
            R = rng.standard_normal((n_atoms, d)).astype(np.float32)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            _, vc_r, _ = find_support_greedy(acts_np, R, max_k=max_k, var_threshold=1.0)
            vc_over_list.append(vc_r[:k_max])

        vc_rand_orth = _avg_curves(vc_orth_list)
        vc_rand_over = _avg_curves(vc_over_list)

        eval_ks = sorted(set(
            [1, 2] + [2**i for i in range(2, 10) if 2**i <= k_max] + [k_max]
        ))

        def _subsample(curve, ks=eval_ks):
            if len(curve) == 0:
                return np.array([]), np.array([])
            valid = [k for k in ks if k <= len(curve)]
            return np.array(valid), np.array([curve[k - 1] for k in valid])

        sns.set_style('whitegrid', {'grid.color': '#dedede'})
        fig, ax = plt.subplots(figsize=(9, 4))

        # Baselines
        ks_pca, vs_pca = _subsample(vc_pca[:k_max])
        if len(ks_pca):
            ax.plot(ks_pca, vs_pca, lw=1.5, color='gray', alpha=0.6,
                    ls='--', label='PCA (optimal)', marker='o', ms=3)
        if len(vc_rand_orth):
            ks_ro, vs_ro = _subsample(vc_rand_orth)
            ax.plot(ks_ro, vs_ro, lw=1.5, color='orange', alpha=0.55,
                    ls=':', label='Random orthogonal', marker='o', ms=3)
        if len(vc_rand_over):
            ks_rov, vs_rov = _subsample(vc_rand_over)
            ax.plot(ks_rov, vs_rov, lw=1.5, color='saddlebrown', alpha=0.55,
                    ls=':', label='Random overcomplete', marker='o', ms=3)

        # ── One pair of curves per SAE ─────────────────────────────────────
        for idx, (sae_path, label) in enumerate(zip(sae_paths, sae_labels)):
            color = _SAE_PALETTE[idx % len(_SAE_PALETTE)]
            sae = load_sae(sae_path, device=DEVICE, **sae_kwargs)
            decoder = get_decoder(sae)
            codes = encode_sae(sae, acts)

            n_active = int((codes > 0).any(0).sum())
            sae_k_max = min(max_k, n_active)

            print(f"  {manifold} [{label}]: geometric reconstruction...")
            _, vc_dir, _ = find_support_greedy(
                acts_np, decoder, max_k=max_k, var_threshold=1.0)

            print(f"  {manifold} [{label}]: statistical reconstruction...")
            _, vc_codes, _ = find_support_greedy_codes(
                acts_np, sae, codes, max_k=max_k, var_threshold=1.0)

            if len(vc_dir):
                ks_d, vs_d = _subsample(vc_dir[:sae_k_max])
                ax.plot(ks_d, vs_d, lw=1.8, color=color, alpha=0.9,
                        ls='-', label=f'{label} geometric',
                        marker='o', ms=4)
            if len(vc_codes):
                vc = vc_codes[:sae_k_max]
                if len(vc) < sae_k_max:
                    vc = np.concatenate([vc, np.full(sae_k_max - len(vc), vc[-1])])
                ks_c, vs_c = _subsample(vc)
                ax.plot(ks_c, vs_c, lw=1.8, color=color, alpha=0.9,
                        ls='--', label=f'{label} statistical',
                        marker='s', ms=4)

        ax.set_xlabel('Number of SAE features')
        ax.set_ylabel('Variance explained')
        ax.set_title(f"{manifold.capitalize()} — manifold reconstruction")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0, frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()
        out_path = out_dir / f"{manifold}_greedy_ve.pdf"
        fig.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"  {manifold}: saved to {out_path}")


# ── Plotting: tuning curves ──────────────────────────────────────────────────

# Manifold -> list of continuous label keys to plot on the x-axis.
_TUNING_LABELS = {
    'age':          ['age'],
    'temperature':  ['fahrenheit'],
    'colors':       ['hue', 'lightness', 'saturation'],
    'geography':    ['latitude', 'longitude'],
    'formality':    ['formality'],
    'sent_length':  ['n_tokens'],
    'years':        ['year'],
}


def plot_tuning_curves(sae_path, manifolds=None, n_features=10, sigma=3,
                       out_dir=None, label_range=None, normalize=True,
                       **sae_kwargs):
    """Plot feature tuning curves — ground-truth label (x) vs activation (y).

    For each manifold with continuous labels, finds the top-``n_features``
    features via the statistical-reconstruction greedy and plots each
    feature's binned activation as a smoothed line over sorted ground-truth
    values. Reproduces Fig. 5 (years_recon_tuning) of the paper.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    import seaborn as sns
    from scipy.ndimage import gaussian_filter1d

    sae = load_sae(sae_path, device=DEVICE, **sae_kwargs)

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [m for m in (manifolds or list(_TUNING_LABELS.keys()))
             if m in _TUNING_LABELS]
    if not names:
        print("No manifolds with continuous labels to plot.")
        return

    plasma = mpl_cm.get_cmap('magma')

    for manifold in names:
        data = load_manifold_data(manifold)
        if data is None:
            print(f"  {manifold}: no cached data, skipping")
            continue

        acts = data['activations']
        mlabels = data['labels']
        acts_np = (acts.float().numpy()
                   if isinstance(acts, torch.Tensor) else acts)
        codes = encode_sae(sae, acts)

        label_keys = [k for k in _TUNING_LABELS[manifold]
                      if k in mlabels[0]]
        if not label_keys:
            print(f"  {manifold}: no matching label keys, skipping")
            continue

        if label_range is not None:
            primary_all = np.array([l[label_keys[0]] for l in mlabels],
                                   dtype=float)
            mask = (primary_all >= label_range[0]) & (primary_all <= label_range[1])
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print(f"  {manifold}: no samples in range {label_range}")
                continue
            acts_np = acts_np[idx]
            codes = codes[idx]
            mlabels = [mlabels[i] for i in idx]

        print(f"  {manifold}: finding top {n_features} features...")
        sel_codes, _, _ = find_support_greedy_codes(
            acts_np, sae, codes, max_k=n_features, var_threshold=1.0)
        top_feats = sel_codes[:n_features]
        n_top = len(top_feats)

        primary_vals = np.array([l[label_keys[0]] for l in mlabels], dtype=float)
        primary_order = np.argsort(primary_vals)
        primary_sorted = primary_vals[primary_order]
        _ux, _inv = np.unique(primary_sorted, return_inverse=True)
        peak_x = []
        for fi in top_feats:
            y = codes[primary_order, fi].astype(float)
            y_bin = np.array([y[_inv == bi].mean() for bi in range(len(_ux))])
            if sigma > 0:
                y_bin = gaussian_filter1d(y_bin, sigma=sigma)
            peak_x.append(_ux[np.argmax(y_bin)])
        top_feats = top_feats[np.argsort(peak_x)]

        sns.set_style('whitegrid', {'grid.color': '#dedede'})
        n_labels = len(label_keys)
        fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 4),
                                 squeeze=False)

        for li, lkey in enumerate(label_keys):
            ax = axes[0, li]
            vals = np.array([l[lkey] for l in mlabels], dtype=float)
            order = np.argsort(vals)
            x_sorted = vals[order]
            unique_x, inverse = np.unique(x_sorted, return_inverse=True)

            for fi_idx, fi in enumerate(top_feats):
                y_sorted = codes[order, fi].astype(float)
                y_binned = np.array([y_sorted[inverse == bi].mean()
                                     for bi in range(len(unique_x))])
                y_smooth = (gaussian_filter1d(y_binned, sigma=sigma)
                            if sigma > 0 else y_binned)
                if normalize:
                    y_min, y_max = y_smooth.min(), y_smooth.max()
                    if y_max - y_min > 1e-10:
                        y_smooth = (y_smooth - y_min) / (y_max - y_min)
                    else:
                        y_smooth = np.zeros_like(y_smooth)
                color = plasma(fi_idx / max(n_top - 1, 1))
                ax.plot(unique_x, y_smooth, linewidth=1.2, color=color,
                        alpha=0.85, label=f'feat {int(fi)}')
                ax.fill_between(unique_x, 0, y_smooth, color=color, alpha=0.15)

            ax.set_xlabel(lkey.replace('_', ' ').capitalize())
            ax.set_ylabel('Feature activation (normalized)' if normalize
                          else 'Feature activation')
            if n_labels > 1:
                ax.set_title(lkey.replace('_', ' ').capitalize())

        fig.suptitle(f'{manifold.capitalize()} — feature tuning curves (top {n_top})',
                     fontsize=13)
        handles, lbls = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='center left',
                   bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()
        out_path = out_dir / f"{manifold}_tuning_curves.pdf"
        fig.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"  {manifold}: saved to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

_DEFAULT_SAE_DIR = CACHE_DIR / "saes"


def _discover_saes(sae_dir, ignore=None):
    """Return sorted list of .pt paths in sae_dir, excluding ignored stems."""
    ignore = set(ignore or [])
    paths = sorted(Path(sae_dir).glob("*.pt"))
    kept = [str(p) for p in paths if p.stem not in ignore]
    if not kept:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {sae_dir}. "
            "Run: uv run train_sae.py"
        )
    return kept


def _resolve_sae_paths(explicit, sae_dir, ignore):
    """Resolve the final list of SAE paths from explicit args + discovery."""
    ignore = set(ignore or [])
    if explicit:
        return [p for p in explicit if Path(p).stem not in ignore]
    return _discover_saes(sae_dir, ignore)


def _add_sae_args(parser):
    parser.add_argument(
        '--sae', type=str, nargs='*', default=None,
        help='SAE checkpoint path(s). If omitted, all .pt files in --sae-dir are used.',
    )
    parser.add_argument(
        '--sae-dir', type=str, default=str(_DEFAULT_SAE_DIR),
        help='Directory to auto-discover checkpoints from (default: cache/saes/)',
    )
    parser.add_argument(
        '--ignore', type=str, nargs='*', default=None,
        metavar='NAME',
        help='SAE names (file stems) to exclude, e.g. --ignore gated jumprelu',
    )
    parser.add_argument('--d-in', type=int, default=D_MODEL)
    parser.add_argument('--d-sae', type=int, default=None,
                        help='SAE width; inferred from checkpoint if absent')
    parser.add_argument('--expansion-factor', type=int, default=None,
                        help='Alternative to --d-sae (d_sae = d_in * factor)')
    parser.add_argument('--k', type=int, default=None,
                        help='Sparsity k; inferred from checkpoint if absent')


def _sae_kwargs(a):
    return dict(d_in=a.d_in, d_sae=a.d_sae,
                expansion_factor=a.expansion_factor, k=a.k)


def main():
    p = argparse.ArgumentParser(
        description="Subspace capture evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command')

    plot_p = sub.add_parser('plot',
                            help='VE curves: geometric vs. statistical greedy + PCA/random baselines')
    _add_sae_args(plot_p)
    plot_p.add_argument('--sae-labels', nargs='*', default=None,
                        help='Display labels for each SAE (default: inferred from checkpoint)')
    plot_p.add_argument('--manifold', nargs='*', default=None)
    plot_p.add_argument('--max-k', type=int, default=64)
    plot_p.add_argument('--out-dir', type=str, default=None)

    tc_p = sub.add_parser('tuning', help='Feature tuning curves (label vs activation)')
    _add_sae_args(tc_p)
    tc_p.add_argument('--manifold', nargs='*', default=None)
    tc_p.add_argument('--n-features', type=int, default=10)
    tc_p.add_argument('--sigma', type=float, default=3,
                       help='Gaussian smoothing sigma (samples)')
    tc_p.add_argument('--out-dir', type=str, default=None)
    tc_p.add_argument('--label-range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'))
    tc_p.add_argument('--no-normalize', action='store_true')

    a = p.parse_args()
    if a.command == 'plot':
        sae_paths = _resolve_sae_paths(a.sae, a.sae_dir, a.ignore)
        print(f"SAEs to compare: {[Path(p).stem for p in sae_paths]}")
        plot_greedy_variance_curves(
            sae_paths, sae_labels=a.sae_labels,
            manifolds=a.manifold, max_k=a.max_k,
            out_dir=a.out_dir, **_sae_kwargs(a))
    elif a.command == 'tuning':
        sae_paths = _resolve_sae_paths(a.sae, a.sae_dir, a.ignore)
        sae_path = sae_paths[0]
        if len(sae_paths) > 1:
            print(f"Note: tuning uses the first SAE ({Path(sae_path).stem}). "
                  "Pass --sae explicitly to choose a different one.")
        plot_tuning_curves(
            sae_path, manifolds=a.manifold,
            n_features=a.n_features, sigma=a.sigma,
            out_dir=a.out_dir,
            label_range=tuple(a.label_range) if a.label_range else None,
            normalize=not a.no_normalize,
            **_sae_kwargs(a))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
