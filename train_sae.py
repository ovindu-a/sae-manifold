"""Train a BatchTopK SAE on cached manifold activations.

This script is intentionally small: it loads cached activations from
``cache/{manifold}.pt``, trains ``saes.BatchTopKSAE`` with a reconstruction
loss, and saves a checkpoint that ``saes.load_sae`` can read back later.

Typical workflow:

  1. Extract cached activations with ``uv run data.py``.
  2. Train an SAE on one or more cached manifolds.
  3. Reuse the saved checkpoint with ``subspace_capture.py`` or
     ``unsupervised_clustering.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import CACHE_DIR, get_all_manifold_names, load_manifold_data
from saes import BatchTopKSAE, GatedSAE, JumpReLUSAE, MatryoshkaSAE, SubspaceSAE


def _load_training_activations(manifolds, filter_outliers=True, n_std=3.0):
    """Load and concatenate cached activations from the requested manifolds."""
    if manifolds is None:
        manifolds = get_all_manifold_names()

    tensors = []
    selected = []
    for name in manifolds:
        data = load_manifold_data(name, filter_outliers=filter_outliers, n_std=n_std)
        if data is None:
            print(f"Skipping {name}: no cache found")
            continue
        acts = data["activations"]
        tensors.append(torch.as_tensor(acts, dtype=torch.float32))
        selected.append(name)
        print(f"Loaded {name}: {acts.shape[0]} samples")

    if not tensors:
        raise FileNotFoundError(
            f"No cached activations found in {CACHE_DIR}. Run: uv run data.py"
        )

    return selected, torch.cat(tensors, dim=0)


def _make_loaders(activations, batch_size, val_fraction, seed):
    dataset = TensorDataset(activations)
    if len(dataset) < 2 or val_fraction <= 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None

    n_val = int(round(len(dataset) * val_fraction))
    n_val = max(1, min(n_val, len(dataset) - 1))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _init_model(sae_type, d_in, d_sae, k, ks, device, dtype, directions=None):
    if sae_type == "gated":
        model = GatedSAE(d_in=d_in, d_sae=d_sae, device=device, dtype=dtype)
    elif sae_type == "jumprelu":
        model = JumpReLUSAE(d_in=d_in, d_sae=d_sae, device=device, dtype=dtype)
    elif sae_type == "matryoshka":
        if ks is None:
            ks = [max(1, k // 4), k // 2, k]
        model = MatryoshkaSAE(d_in=d_in, d_sae=d_sae, ks=ks, device=device, dtype=dtype)
    elif sae_type == "subspace":
        # directions is a [n_dirs, d_in] array whose rows become the fixed decoder
        # directions for the first n_dirs features.
        if directions is None:
            raise ValueError(
                "sae_type='subspace' requires directions. "
                "Pass --directions <file> or --directions-manifold <name>."
            )
        model = SubspaceSAE(d_in=d_in, d_sae=d_sae, k=k,
                            directions=directions, device=device, dtype=dtype)
    else:
        model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)
    # Zero all Linear biases at init; encoder bias is the only one for SubspaceSAE
    # (free_decoder is an nn.Parameter, not nn.Linear, so it is unaffected).
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    return model


def _run_epoch(model, loader, optimizer=None, device="cpu", l1_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    for (batch,) in loader:
        batch = batch.to(device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        if isinstance(model, MatryoshkaSAE):
            # Average reconstruction loss across all k levels.
            recons = model.forward_all_k(batch)
            loss = sum(F.mse_loss(r, batch) for r in recons) / len(recons)
        else:
            recon, z = model(batch)
            loss = F.mse_loss(recon, batch)
            if l1_weight > 0:
                loss = loss + l1_weight * z.abs().mean()

        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach()) * batch.shape[0]
        total_examples += batch.shape[0]

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _fit_jumprelu_thresholds(model, train_loader, target_k, device):
    """Set per-feature thresholds so average L0 ≈ target_k on training data."""
    model.eval()
    pres = []
    for (batch,) in train_loader:
        pre = F.relu(model.encoder(batch.to(device)))
        pres.append(pre.cpu().float())
    pre_all = torch.cat(pres, dim=0).numpy()  # [N, d_sae]

    # For each feature, find the threshold that keeps (target_k / d_sae) of
    # all samples active — i.e. the (1 - target_rate) quantile of pre-activations.
    target_rate = float(target_k) / model.d_sae
    thresholds = np.zeros(model.d_sae, dtype=np.float32)
    for i in range(model.d_sae):
        t = float(np.quantile(pre_all[:, i], 1.0 - target_rate))
        thresholds[i] = max(t, 0.0)  # pre is already ≥ 0 after ReLU
    model.threshold.copy_(torch.from_numpy(thresholds))

    total_fires = total_n = 0
    for (batch,) in train_loader:
        z = model.encode(batch.to(device))
        total_fires += (z > 0).float().sum().item()
        total_n += batch.shape[0]
    print(f"  JumpReLU thresholds set: avg L0={total_fires / max(total_n, 1):.1f} "
          f"(target {target_k})")


def train_sae(
    manifolds=None,
    output_path="cache/sae.pt",
    sae_type="batchtopk",
    d_sae=None,
    expansion_factor=4.0,
    k=64,
    matryoshka_ks=None,
    l1_weight=0.0,
    batch_size=256,
    epochs=20,
    lr=3e-4,
    weight_decay=0.0,
    val_fraction=0.1,
    seed=0,
    device=None,
    dtype=torch.float32,
    filter_outliers=True,
    n_std=3.0,
    directions=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_manifolds, activations = _load_training_activations(
        manifolds, filter_outliers=filter_outliers, n_std=n_std
    )
    d_in = activations.shape[1]
    if d_sae is None:
        d_sae = int(round(d_in * expansion_factor))

    train_loader, val_loader = _make_loaders(activations, batch_size, val_fraction, seed)
    model = _init_model(sae_type=sae_type, d_in=d_in, d_sae=d_sae, k=k,
                        ks=matryoshka_ks, device=device, dtype=dtype,
                        directions=directions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ks_actual = model.ks if isinstance(model, MatryoshkaSAE) else None
    # For SubspaceSAE, also show how many features are pinned vs free.
    n_dirs_str = (f", n_dirs={model.n_dirs}" if isinstance(model, SubspaceSAE) else "")
    print(f"Training on manifolds: {', '.join(selected_manifolds)}")
    print(f"Activations: {tuple(activations.shape)}")
    print(f"SAE type: {sae_type} | d_in={d_in}, d_sae={d_sae}, k={k}"
          + n_dirs_str
          + (f", ks={ks_actual}" if ks_actual else "")
          + (f", l1={l1_weight}" if l1_weight > 0 else "")
          + f" | device={device}, dtype={dtype}")

    history = []
    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer,
                                device=device, l1_weight=l1_weight)
        val_loss = (_run_epoch(model, val_loader, device=device)
                    if val_loader else None)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss is None:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {_k: v.detach().cpu().clone()
                              for _k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # JumpReLU: fit per-feature thresholds on training data after finding the
    # best weights.  Thresholds are stored in the buffer so they are part of
    # the saved state_dict.
    if sae_type == "jumprelu":
        print("Fitting JumpReLU thresholds...")
        _fit_jumprelu_thresholds(model, train_loader, target_k=k, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {
            "sae_type": sae_type,
            "d_in": d_in,
            "d_sae": d_sae,
            "k": k,
            "ks": ks_actual,
            "expansion_factor": float(d_sae) / float(d_in),
            "manifolds": selected_manifolds,
            "filter_outliers": filter_outliers,
            "n_std": n_std,
            # SubspaceSAE: store n_dirs so load_sae can reconstruct the model shape.
            # The actual direction vectors are already in state_dict["pinned_directions"]
            # (saved as a buffer), so this is just a convenience for load_sae.
            **({"n_dirs": model.n_dirs} if isinstance(model, SubspaceSAE) else {}),
        },
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "l1_weight": l1_weight,
            "weight_decay": weight_decay,
            "val_fraction": val_fraction,
            "seed": seed,
            "device": device,
            "dtype": str(dtype),
        },
        "history": history,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")
    return output_path


_ALL_SAE_TYPES = ["batchtopk", "gated", "jumprelu", "matryoshka", "subspace"]
_L1_DEFAULTS = {"gated": 1e-3, "jumprelu": 1e-3}
_DEFAULT_DIRECTIONS_FILE = CACHE_DIR / "default_directions.npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifold",
        nargs="*",
        default=None,
        help="Cached manifolds to train on (default: all shipped manifolds with caches)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cache/saes",
        help="Directory to write checkpoints; each type saved as {type}.pt (default: cache/saes/)",
    )
    parser.add_argument(
        "--sae-type",
        nargs="+",
        default=_ALL_SAE_TYPES,
        metavar="TYPE",
        help=(
            "One or more SAE types to train: "
            "batchtopk gated jumprelu matryoshka subspace. "
            "Default: all five."
        ),
    )
    parser.add_argument(
        "--directions",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a .npy or .pt file containing an [n_dirs, d_in] array of "
            "concept directions for the subspace SAE type. "
            "Mutually exclusive with --directions-manifold. "
            f"If not specified, will use default file at {_DEFAULT_DIRECTIONS_FILE} if it exists."
        ),
    )
    parser.add_argument(
        "--directions-manifold",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Compute PCA directions from this manifold's cached activations and "
            "use them as the subspace SAE's pinned directions. "
            "Mutually exclusive with --directions."
        ),
    )
    parser.add_argument(
        "--directions-n-components",
        type=int,
        default=5,
        help=(
            "Number of PCA components to extract when using --directions-manifold "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--d-sae",
        type=int,
        default=None,
        help="Number of SAE features (default: d_in * expansion_factor)",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=4.0,
        help="Feature expansion factor used when --d-sae is not set",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="Target sparsity: top-k for batchtopk/matryoshka, target L0 for jumprelu",
    )
    parser.add_argument(
        "--matryoshka-ks",
        type=int,
        nargs="+",
        default=None,
        help="k levels for matryoshka training (default: [k//4, k//2, k])",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=None,
        help=(
            "L1 sparsity penalty weight. "
            "Defaults to 1e-3 for gated/jumprelu, 0 otherwise."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to train on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="Disable activation-norm outlier filtering",
    )
    parser.add_argument("--n-std", type=float, default=3.0, help="Outlier cutoff in std devs")

    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    unknown = [t for t in args.sae_type if t not in _ALL_SAE_TYPES]
    if unknown:
        parser.error(f"Unknown SAE type(s): {unknown}. Choose from {_ALL_SAE_TYPES}")

    # ── Resolve directions for SubspaceSAE ───────────────────────────────────
    # directions is only needed (and validated) when subspace is in the type list.
    directions = None
    if "subspace" in args.sae_type:
        if args.directions is not None and args.directions_manifold is not None:
            parser.error("Pass --directions or --directions-manifold, not both.")
        elif args.directions is not None:
            # Load a pre-computed directions array from a file.
            p = Path(args.directions)
            if p.suffix == ".npy":
                directions = np.load(p)
            else:
                # Accept a raw tensor or a dict containing a "directions" key.
                obj = torch.load(p, map_location="cpu", weights_only=False)
                directions = (obj.numpy() if isinstance(obj, torch.Tensor)
                              else np.array(obj))
            print(f"Loaded {directions.shape[0]} directions from {p}")
        elif args.directions_manifold is not None:
            # Compute PCA directions on the fly from a cached manifold.
            from subspace_capture import get_manifold_pca_directions
            directions, var_exp = get_manifold_pca_directions(
                args.directions_manifold,
                n_components=args.directions_n_components,
            )
            print(
                f"PCA directions from '{args.directions_manifold}': "
                f"{directions.shape[0]} components, "
                f"variance explained = {[f'{v:.3f}' for v in var_exp]}"
            )
        elif _DEFAULT_DIRECTIONS_FILE.exists():
            # Fall back to default directions file if it exists.
            print(f"Using default directions file: {_DEFAULT_DIRECTIONS_FILE}")
            directions = np.load(_DEFAULT_DIRECTIONS_FILE)
            print(f"Loaded {directions.shape[0]} directions from default file")
        else:
            parser.error(
                "--sae-type subspace requires --directions <file>, "
                "--directions-manifold <name>, or a default directions file at "
                f"{_DEFAULT_DIRECTIONS_FILE}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sae_type in args.sae_type:
        output_path = output_dir / f"{sae_type}.pt"
        l1_weight = (args.l1_weight if args.l1_weight is not None
                     else _L1_DEFAULTS.get(sae_type, 0.0))
        print(f"\n{'='*60}\nTraining {sae_type}  →  {output_path}\n{'='*60}")
        train_sae(
            manifolds=args.manifold,
            output_path=str(output_path),
            sae_type=sae_type,
            d_sae=args.d_sae,
            expansion_factor=args.expansion_factor,
            k=args.k,
            matryoshka_ks=args.matryoshka_ks,
            l1_weight=l1_weight,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            dtype=dtype,
            filter_outliers=not args.no_filter_outliers,
            n_std=args.n_std,
            directions=directions,
        )


if __name__ == "__main__":
    main()