"""Create combined PCA directions from multiple manifolds for SubspaceSAE training.

Usage:
    # Create custom directions file
    python create_multi_directions.py \
        --manifolds years colors temperature \
        --n-components 5 5 3 \
        --output combined_directions.npy

    # Create default directions file (auto-used by train_sae.py)
    python create_multi_directions.py \
        --manifolds years colors \
        --n-components 10 10 \
        --set-as-default
"""
import argparse
import numpy as np
from pathlib import Path
from data import CACHE_DIR
from subspace_capture import get_manifold_pca_directions

DEFAULT_DIRECTIONS_FILE = CACHE_DIR / "default_directions.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Combine PCA directions from multiple manifolds"
    )
    parser.add_argument(
        "--manifolds",
        nargs="+",
        required=True,
        help="List of manifold names (e.g., years colors temperature)",
    )
    parser.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        required=True,
        help="Number of PCA components per manifold (must match number of manifolds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: multi_manifold_directions.npy, or cache/default_directions.npy if --set-as-default)",
    )
    parser.add_argument(
        "--set-as-default",
        action="store_true",
        help=f"Save as default directions file at {DEFAULT_DIRECTIONS_FILE} (will be auto-used by train_sae.py)",
    )
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        default=True,
        help="Filter activation norm outliers (default: True)",
    )
    parser.add_argument(
        "--n-std",
        type=float,
        default=3.0,
        help="Outlier cutoff in standard deviations (default: 3.0)",
    )

    args = parser.parse_args()

    if len(args.manifolds) != len(args.n_components):
        parser.error(
            f"Number of manifolds ({len(args.manifolds)}) must match "
            f"number of n-components ({len(args.n_components)})"
        )

    # Determine output path
    if args.set_as_default:
        output_path = DEFAULT_DIRECTIONS_FILE
    elif args.output:
        output_path = args.output
    else:
        output_path = "multi_manifold_directions.npy"

    all_directions = []
    total_components = 0

    print("Extracting PCA directions from manifolds:")
    print("=" * 60)

    for manifold, n_comp in zip(args.manifolds, args.n_components):
        print(f"\n{manifold}: extracting {n_comp} components...")

        try:
            directions, var_exp = get_manifold_pca_directions(
                manifold,
                n_components=n_comp,
                filter_outliers=args.filter_outliers,
                n_std=args.n_std,
            )

            all_directions.append(directions)
            total_components += n_comp

            print(f"  Shape: {directions.shape}")
            print(f"  Variance explained: {var_exp}")
            print(f"  Total variance: {var_exp.sum():.4f}")

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  Run: uv run data.py --manifold {manifold}")
            return
        except Exception as e:
            print(f"  ERROR: {e}")
            return

    # Stack all directions vertically
    combined = np.vstack(all_directions)

    print("\n" + "=" * 60)
    print(f"Combined directions shape: {combined.shape}")
    print(f"Total pinned features: {total_components}")
    print(f"Saving to: {output_path}")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, combined)

    if args.set_as_default:
        print(f"\n✓ Set as default directions file!")
        print(f"  train_sae.py will now use these directions automatically for subspace SAE")

    print("Done!")


if __name__ == "__main__":
    main()
