"""Analyze SAE feature activations across manifold datasets.

This script loads all trained SAEs and runs cached manifold activations through them,
tracking which prompts activate which features and by how much.

Output: CSV with columns [sae_type, feature_idx, activation_value, manifold_name, prompt]
sorted by activation strength for each feature.

Usage:
    python analyze_sae_activations.py --output sae_feature_activations.csv
    python analyze_sae_activations.py --sae-dir cache/saes --manifolds colors age temperature
    python analyze_sae_activations.py --top-n 10  # Only keep top 10 prompts per feature
"""
import argparse
from pathlib import Path
from collections import defaultdict
import csv

import torch
import numpy as np
from tqdm import tqdm

from data import CACHE_DIR, get_all_manifold_names, load_manifold_data
from saes import load_sae


def find_available_saes(sae_dir):
    """Find all .pt files in the SAE directory."""
    sae_path = Path(sae_dir)
    if not sae_path.exists():
        return []

    sae_files = list(sae_path.glob("*.pt"))
    # Extract SAE type from filename (e.g., "batchtopk.pt" -> "batchtopk")
    return [(f.stem, f) for f in sae_files]


def analyze_sae_on_manifolds(sae, sae_type, manifolds, filter_outliers=True, n_std=3.0):
    """Run a single SAE on all manifold activations and collect firing patterns.

    Returns:
        dict: {feature_idx: [(activation_value, manifold_name, prompt), ...]}
    """
    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype

    # Store all activations for each feature
    feature_activations = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"Analyzing {sae_type} SAE")
    print(f"{'='*60}")

    for manifold_name in manifolds:
        print(f"\nProcessing manifold: {manifold_name}")

        # Load manifold data
        data = load_manifold_data(manifold_name, filter_outliers=filter_outliers, n_std=n_std)
        if data is None:
            print(f"  Skipping {manifold_name}: no cache found")
            continue

        activations = data['activations']
        prompts = data.get('prompts', [f"sample_{i}" for i in range(len(activations))])

        print(f"  Loaded {len(activations)} samples")

        # Convert to tensor
        x = torch.as_tensor(activations, dtype=dtype, device=device)

        # Encode through SAE
        with torch.no_grad():
            z = sae.encode(x)  # [N, d_sae]

        # Convert to numpy for easier processing
        z_np = z.cpu().float().numpy()  # [N, d_sae]

        # For each sample, record which features fired
        n_samples, d_sae = z_np.shape
        print(f"  SAE output shape: {z_np.shape}")

        # Count non-zero activations
        n_active = (z_np > 0).sum()
        print(f"  Total active features: {n_active} ({n_active / (n_samples * d_sae) * 100:.2f}%)")

        # Collect activations for each feature
        for feature_idx in range(d_sae):
            feature_acts = z_np[:, feature_idx]  # [N]

            # Find samples where this feature fired
            active_mask = feature_acts > 0
            if not active_mask.any():
                continue

            active_indices = np.where(active_mask)[0]

            for sample_idx in active_indices:
                activation_value = float(feature_acts[sample_idx])
                prompt = prompts[sample_idx]
                feature_activations[feature_idx].append(
                    (activation_value, manifold_name, prompt)
                )

        # Print feature firing statistics for this manifold
        n_features_fired = sum(1 for acts in z_np.T if (acts > 0).any())
        print(f"  Features that fired at least once: {n_features_fired}/{d_sae}")

    return feature_activations


def write_results_to_csv(results, output_path, top_n=None):
    """Write feature activation results to CSV.

    Args:
        results: dict of {sae_type: {feature_idx: [(activation, manifold, prompt), ...]}}
        output_path: path to output CSV file
        top_n: if provided, only keep top N activations per feature
    """
    print(f"\nWriting results to {output_path}...")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sae_type', 'feature_idx', 'activation_value', 'manifold_name', 'prompt'])

        total_rows = 0

        for sae_type, feature_data in results.items():
            for feature_idx, activations in tqdm(feature_data.items(),
                                                  desc=f"Writing {sae_type}"):
                # Sort by activation value (descending)
                activations_sorted = sorted(activations, key=lambda x: x[0], reverse=True)

                # Optionally limit to top N
                if top_n is not None:
                    activations_sorted = activations_sorted[:top_n]

                # Write each activation
                for activation_value, manifold_name, prompt in activations_sorted:
                    writer.writerow([
                        sae_type,
                        feature_idx,
                        f"{activation_value:.6f}",
                        manifold_name,
                        prompt
                    ])
                    total_rows += 1

        print(f"Wrote {total_rows} rows")


def print_summary(results):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    for sae_type, feature_data in results.items():
        n_features = len(feature_data)
        total_activations = sum(len(acts) for acts in feature_data.values())

        # Find most active feature
        if feature_data:
            most_active_feature = max(feature_data.items(), key=lambda x: len(x[1]))
            most_active_idx, most_active_acts = most_active_feature

            print(f"\n{sae_type}:")
            print(f"  Features that fired: {n_features}")
            print(f"  Total activations: {total_activations}")
            print(f"  Avg activations per feature: {total_activations / n_features:.1f}")
            print(f"  Most active feature: {most_active_idx} ({len(most_active_acts)} activations)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sae-dir",
        type=str,
        default="cache/saes",
        help="Directory containing trained SAE checkpoints (default: cache/saes)",
    )
    parser.add_argument(
        "--manifolds",
        nargs="*",
        default=None,
        help="Specific manifolds to analyze (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sae_feature_activations.csv",
        help="Output CSV file path (default: sae_feature_activations.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only keep top N activations per feature (default: keep all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="Disable activation-norm outlier filtering",
    )
    parser.add_argument(
        "--n-std",
        type=float,
        default=3.0,
        help="Outlier cutoff in std devs (default: 3.0)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Find available SAEs
    available_saes = find_available_saes(args.sae_dir)
    if not available_saes:
        print(f"No SAE checkpoints found in {args.sae_dir}")
        print("Run: python train_sae.py")
        return

    print(f"Found {len(available_saes)} SAE(s):")
    for sae_type, sae_path in available_saes:
        print(f"  - {sae_type}: {sae_path}")

    # Get manifolds to analyze
    if args.manifolds is None or len(args.manifolds) == 0:
        manifolds = get_all_manifold_names()
    else:
        manifolds = args.manifolds

    print(f"\nManifolds to analyze: {', '.join(manifolds)}")

    # Check that manifold caches exist
    available_manifolds = []
    for manifold_name in manifolds:
        cache_path = CACHE_DIR / f"{manifold_name}.pt"
        if cache_path.exists():
            available_manifolds.append(manifold_name)
        else:
            print(f"Warning: {manifold_name} cache not found at {cache_path}")

    if not available_manifolds:
        print("\nNo manifold caches found. Run: python data.py")
        return

    manifolds = available_manifolds
    print(f"Available manifolds: {', '.join(manifolds)}")

    # Analyze each SAE
    all_results = {}

    for sae_type, sae_path in available_saes:
        print(f"\nLoading {sae_type} SAE from {sae_path}...")

        try:
            sae = load_sae(str(sae_path), device=device)
            print(f"  d_in={sae.d_in}, d_sae={sae.d_sae}")
            if hasattr(sae, 'k'):
                print(f"  k={sae.k}")

            # Analyze this SAE on all manifolds
            feature_activations = analyze_sae_on_manifolds(
                sae, sae_type, manifolds,
                filter_outliers=not args.no_filter_outliers,
                n_std=args.n_std
            )

            all_results[sae_type] = feature_activations

        except Exception as e:
            print(f"Error loading/analyzing {sae_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\nNo results to write.")
        return

    # Print summary
    print_summary(all_results)

    # Write results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_to_csv(all_results, output_path, top_n=args.top_n)

    print(f"\n✓ Analysis complete! Results written to {output_path}")
    print(f"\nTo explore the results:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{output_path}')")
    print(f"  df.head()")


if __name__ == "__main__":
    main()
