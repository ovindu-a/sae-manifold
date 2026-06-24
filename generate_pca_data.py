"""Pre-compute 3D PCA projections for all manifold activations.

Generates pca_data.json alongside the SAE activations CSV, which the webapp
server serves via the /pca_data endpoint.

Usage:
    python generate_pca_data.py
    python generate_pca_data.py --output pca_data.json --manifolds colors age days
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from data import CACHE_DIR, get_all_manifold_names, load_manifold_data


def generate_pca_data(manifolds=None, output_path="pca_data.json",
                      n_components=3, filter_outliers=True, n_std=3.0):
    names = manifolds or get_all_manifold_names()
    result = {}

    for name in names:
        print(f"Processing {name}...")
        data = load_manifold_data(name, filter_outliers=filter_outliers, n_std=n_std)
        if data is None:
            print(f"  Skipping {name}: no cache found")
            continue

        acts = data["activations"]
        prompts = data.get("prompts", [f"sample_{i}" for i in range(len(acts))])
        labels = data.get("labels", [{} for _ in range(len(acts))])

        acts_np = acts.float().numpy() if isinstance(acts, torch.Tensor) else np.asarray(acts, dtype=np.float32)
        n_samples = acts_np.shape[0]
        n_comp = min(n_components, n_samples, acts_np.shape[1])

        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(acts_np)  # [N, 3]
        var_exp = pca.explained_variance_ratio_.tolist()

        print(f"  {n_samples} samples | variance explained: {[f'{v:.3f}' for v in var_exp]}")

        points = []
        for i in range(n_samples):
            point = {
                "prompt": prompts[i],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]) if n_comp >= 3 else 0.0,
                "label": labels[i],
            }
            points.append(point)

        result[name] = {
            "points": points,
            "variance_explained": var_exp,
            "n_samples": n_samples,
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {out} ({size_mb:.2f} MB)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="pca_data.json")
    parser.add_argument("--manifolds", nargs="*", default=None)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--no-filter-outliers", action="store_true")
    parser.add_argument("--n-std", type=float, default=3.0)
    args = parser.parse_args()

    generate_pca_data(
        manifolds=args.manifolds,
        output_path=args.output,
        n_components=args.n_components,
        filter_outliers=not args.no_filter_outliers,
        n_std=args.n_std,
    )
