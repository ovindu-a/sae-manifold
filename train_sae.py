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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import CACHE_DIR, get_all_manifold_names, load_manifold_data
from saes import BatchTopKSAE


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


def _init_model(d_in, d_sae, k, device, dtype):
    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)
    if model.encoder.bias is not None:
        nn.init.zeros_(model.encoder.bias)
    if model.decoder.bias is not None:
        nn.init.zeros_(model.decoder.bias)
    return model


def _run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    for (batch,) in loader:
        batch = batch.to(device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        recon, _ = model(batch)
        loss = torch.nn.functional.mse_loss(recon, batch)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = batch.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_sae(
    manifolds=None,
    output_path="cache/sae.pt",
    d_sae=None,
    expansion_factor=4.0,
    k=64,
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
    model = _init_model(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Training on manifolds: {', '.join(selected_manifolds)}")
    print(f"Activations: {tuple(activations.shape)}")
    print(f"SAE: d_in={d_in}, d_sae={d_sae}, k={k}, device={device}, dtype={dtype}")

    history = []
    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss = _run_epoch(model, val_loader, optimizer=None, device=device) if val_loader else None
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss is None:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {
            "d_in": d_in,
            "d_sae": d_sae,
            "k": k,
            "expansion_factor": float(d_sae) / float(d_in),
            "manifolds": selected_manifolds,
            "filter_outliers": filter_outliers,
            "n_std": n_std,
        },
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifold",
        nargs="*",
        default=None,
        help="Cached manifolds to train on (default: all shipped manifolds with caches)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cache/sae.pt",
        help="Path to write the SAE checkpoint",
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
    parser.add_argument("--k", type=int, default=64, help="Inference-time sparsity")
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

    train_sae(
        manifolds=args.manifold,
        output_path=args.output,
        d_sae=args.d_sae,
        expansion_factor=args.expansion_factor,
        k=args.k,
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
    )


if __name__ == "__main__":
    main()