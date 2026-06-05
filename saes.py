"""
Reference inference-only implementation of a BatchTopK Sparse Autoencoder.

This module accompanies the paper "Do Sparse Autoencoders Capture Concept
Manifolds?" and exists solely to load a trained SAE checkpoint and run the
evaluations in this repository. It is *not* a training implementation:
the BatchTopK objective is described in Bussmann, Leask & Nanda (2024),
"BatchTopK Sparse Autoencoders" (arXiv:2412.06410), and any standard SAE
training stack can produce a checkpoint compatible with ``load_sae`` below.

At inference we take a per-sample top-k, which is consistent with how
BatchTopK SAEs are evaluated in practice and does not depend on batch size.
If the checkpoint stores a per-feature JumpReLU-style threshold buffer
(``threshold``), that is used instead.

If you trained with a different architecture (TopK, JumpReLU, Matryoshka,
Archetypal, ...), drop in your own ``nn.Module``: the rest of the repo only
depends on three things,

  - ``sae.encode(x) -> codes``           (sparse [N, d_sae])
  - ``sae.decode(z) -> reconstruction``  (dense [N, d_in])
  - ``get_decoder(sae) -> ndarray``      ([d_sae, d_in])

so adapting another class is a small change.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """Minimal BatchTopK SAE with per-sample top-k at inference.

    Args:
        d_in: input (model activation) dimension
        d_sae: number of SAE features
        k: number of features kept active per sample at inference
        device: optional device to place the model on
        dtype: parameter dtype (default float32)
    """
    def __init__(self, d_in, d_sae, k, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = int(k)
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        # Optional learned per-feature threshold for JumpReLU-style inference.
        # If set (any nonzero entry), encode() uses thresholding instead of top-k.
        self.register_buffer("threshold", torch.zeros(d_sae))
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode(self, x):
        """Return sparse codes ``[N, d_sae]`` for inputs ``x`` ``[N, d_in]``."""
        pre = F.relu(self.encoder(x))
        if torch.any(self.threshold > 0):
            return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        if self.k >= self.d_sae:
            return pre
        top = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, top.indices, top.values)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class GatedSAE(nn.Module):
    """Gated SAE: separate linear maps for binary gate and activation magnitude.

    The gate decides which features fire; the magnitude controls how strongly.
    During training a straight-through estimator (STE) keeps the gate
    differentiable. At inference the gate is a hard threshold at zero.
    """
    def __init__(self, d_in, d_sae, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_gate = nn.Linear(d_in, d_sae)
        self.W_mag = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode(self, x):
        gate = (self.W_gate(x) > 0).float()
        return gate * F.relu(self.W_mag(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        pre_gate = self.W_gate(x)
        mag = F.relu(self.W_mag(x))
        soft = torch.sigmoid(pre_gate)
        gate = (pre_gate > 0).float() + (soft - soft.detach())  # STE
        z = gate * mag
        return self.decoder(z), z


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE: ReLU encoder with a per-feature activation threshold.

    The threshold buffer starts at zero and is fitted post-training (by
    ``train_sae.py``) so average L0 matches a target sparsity. At inference
    features only fire when their pre-activation exceeds their threshold.
    """
    def __init__(self, d_in, d_sae, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        self.register_buffer("threshold", torch.zeros(d_sae))
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode(self, x):
        pre = F.relu(self.encoder(x))
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))  # won't activations that are always below threshold get pruned during training?
    

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        pre = F.relu(self.encoder(x))
        z = torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        return self.decoder(z), z


class MatryoshkaSAE(nn.Module):
    """Matryoshka SAE: shared decoder trained simultaneously at multiple k levels.

    The reconstruction loss is averaged over all k levels during training.
    At inference ``self.k`` (the largest k) is used; set ``sae.k`` before
    calling ``encode`` to evaluate at a coarser resolution.
    """
    def __init__(self, d_in, d_sae, ks, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.ks = sorted(ks)
        self.k = self.ks[-1]
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode(self, x):
        pre = F.relu(self.encoder(x))
        top = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, top.indices, top.values)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z

    def forward_all_k(self, x):
        """Return a list of reconstructions, one per k in ``self.ks``."""
        pre = F.relu(self.encoder(x))
        recons = []
        for k in self.ks:
            top = pre.topk(k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, top.indices, top.values)
            recons.append(self.decoder(z))
        return recons


class SubspaceSAE(nn.Module):
    """SAE with dedicated features whose decoder directions are fixed to provided concept vectors.

    The first ``n_dirs`` features are "pinned": their decoder directions are set to
    the provided directions at construction time and never updated by the optimiser.
    The remaining ``d_sae - n_dirs`` features are free and behave like standard
    BatchTopK features.

    All features compete in the same global top-k selection, so a pinned feature
    fires only when the input has meaningful signal along its fixed direction —
    it is not forced to be always active.

    Args:
        d_in: input (model activation) dimension
        d_sae: total number of SAE features; must be strictly greater than n_dirs
        k: number of features kept active per sample (top-k sparsity)
        directions: ``[n_dirs, d_in]`` array — the concept directions to pin
            (e.g. PCA axes). They will be L2-normalised internally.
        device: optional torch device
        dtype: parameter dtype (default float32)
    """
    def __init__(self, d_in, d_sae, k, directions, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = int(k)

        # Normalise the provided directions to unit norm so that each pinned
        # feature's decoder direction has the same scale as a standard SAE atom.
        dirs = F.normalize(
            torch.as_tensor(directions, dtype=dtype), dim=1
        )  # [n_dirs, d_in]
        n_dirs = dirs.shape[0]
        if not (0 < n_dirs < d_sae):
            raise ValueError(
                f"n_dirs={n_dirs} must satisfy 0 < n_dirs < d_sae={d_sae}"
            )
        self.n_dirs = n_dirs

        # ── Encoder ──────────────────────────────────────────────────────────
        # A single unconstrained linear map: all d_sae features (pinned and free)
        # are encoded from the full d_in activation. The pinned decoder directions
        # don't restrict what the encoder can learn, but warm-starting rows 0..n_dirs-1
        # to the concept directions gives those features a head-start: at
        # initialisation, feature i computes the projection of x onto direction i.
        self.encoder = nn.Linear(d_in, d_sae)
        with torch.no_grad():
            # Overwrite the first n_dirs encoder rows with the pinned directions.
            # The remaining rows keep the default kaiming_uniform_ initialisation.
            self.encoder.weight[:n_dirs] = dirs  # [n_dirs, d_in]

        # ── Decoder ──────────────────────────────────────────────────────────
        # Pinned directions are registered as a buffer so they are:
        #   - saved and restored by state_dict (checkpoint round-trips correctly)
        #   - NOT treated as nn.Parameter, so the optimiser never touches them
        self.register_buffer('pinned_directions', dirs)  # [n_dirs, d_in]

        # The free decoder weights for the remaining d_sae - n_dirs features are
        # a standard trainable Parameter of shape [d_sae - n_dirs, d_in].
        self.free_decoder = nn.Parameter(
            torch.empty(d_sae - n_dirs, d_in, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.free_decoder)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _full_decoder_weight(self):
        """Assemble the full ``[d_sae, d_in]`` decoder weight matrix on the fly.

        Row i is the decoder direction for feature i. Pinned rows (0..n_dirs-1)
        come first; free rows (n_dirs..d_sae-1) follow. The cat is cheap — it
        just creates a view, so there is no meaningful runtime overhead.
        """
        return torch.cat([
            self.pinned_directions,  # [n_dirs, d_in] — fixed buffer, never updated
            self.free_decoder,       # [d_sae - n_dirs, d_in] — trained parameter
        ], dim=0)

    def encode(self, x):
        """Sparse-encode ``x`` ``[N, d_in]`` → codes ``[N, d_sae]``."""
        # All d_sae features (pinned and free) compete for the k active slots.
        pre = F.relu(self.encoder(x))
        top = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, top.indices, top.values)
        return z

    def decode(self, z):
        """Reconstruct from sparse codes ``z`` ``[N, d_sae]`` → ``[N, d_in]``."""
        # Each active code z_i contributes z_i * (row i of decoder) to the output.
        # For pinned features this is z_i * fixed_direction_i, so pinned features
        # can only reconstruct signal along their assigned concept direction.
        return z @ self._full_decoder_weight()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def load_sae(path, d_in=4096, d_sae=None, k=None, expansion_factor=None,
             device=None, dtype=torch.float32):
    """Load an SAE checkpoint of any supported type.

    Accepts either a raw ``state_dict`` or a dict with ``state_dict`` and
    ``model_config`` keys. The config may carry ``sae_type`` (one of
    ``batchtopk``, ``gated``, ``jumprelu``, ``matryoshka``, ``subspace``) plus
    the usual architecture parameters.

    Args:
        path: path to the checkpoint
        d_in: model activation dim (default: 4096 for Llama-3.1-8B)
        d_sae: explicit SAE width (if not in checkpoint config)
        k: sparsity / target L0 (if not in checkpoint config)
        expansion_factor: alternative to ``d_sae`` (``d_sae = d_in * factor``)
        device: optional torch device
        dtype: parameter dtype
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    sae_type = "batchtopk"
    ks = None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        cfg = ckpt.get("model_config", ckpt.get("config", {}))
        d_in = cfg.get("d_in", d_in)
        d_sae = cfg.get("d_sae", d_sae)
        k = cfg.get("k", cfg.get("target_l0", k))
        sae_type = cfg.get("sae_type", "batchtopk")
        ks = cfg.get("ks", None)
        if d_sae is None and "expansion_factor" in cfg:
            expansion_factor = cfg["expansion_factor"]
    else:
        state_dict = ckpt

    if d_sae is None:
        if expansion_factor is not None:
            d_sae = int(d_in * expansion_factor)
        else:
            for key in ("encoder.weight", "encoder_linear.weight", "W_enc",
                        "W_gate.weight"):
                if key in state_dict:
                    d_sae = state_dict[key].shape[0]
                    break
    if d_sae is None:
        raise ValueError("Could not infer d_sae; pass d_sae=... or expansion_factor=...")

    if sae_type == "gated":
        sae = GatedSAE(d_in=d_in, d_sae=d_sae, device=device, dtype=dtype)
    elif sae_type == "jumprelu":
        sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae, device=device, dtype=dtype)
    elif sae_type == "matryoshka":
        if ks is None:
            ks = [max(1, k // 4), k // 2, k] if k else [16, 32, 64]
        sae = MatryoshkaSAE(d_in=d_in, d_sae=d_sae, ks=ks, device=device, dtype=dtype)
    elif sae_type == "subspace":
        if k is None:
            raise ValueError("Could not infer k; pass k=...")
        # n_dirs is stored in model_config. The actual direction vectors live in
        # state_dict["pinned_directions"] and will be restored by load_state_dict
        # below — so we only need the shape here to construct the model.
        n_dirs = cfg.get("n_dirs") if (isinstance(ckpt, dict) and "state_dict" in ckpt) else None
        if n_dirs is None and "pinned_directions" in state_dict:
            n_dirs = state_dict["pinned_directions"].shape[0]
        if n_dirs is None:
            raise ValueError(
                "Could not infer n_dirs for SubspaceSAE; "
                "checkpoint model_config must contain 'n_dirs'."
            )
        # Use placeholder directions of the right shape; load_state_dict will
        # overwrite pinned_directions with the saved values.
        dummy_dirs = torch.zeros(n_dirs, d_in)
        sae = SubspaceSAE(d_in=d_in, d_sae=d_sae, k=k,
                          directions=dummy_dirs, device=device, dtype=dtype)
    else:
        if k is None:
            raise ValueError("Could not infer k; pass k=...")
        sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)

    # Rename legacy keys ("encoder_linear" / "decoder_linear" → "encoder" / "decoder").
    renamed = {}
    for key, val in state_dict.items():
        new_key = key.replace("encoder_linear.", "encoder.").replace(
            "decoder_linear.", "decoder.")
        renamed[new_key] = val
    missing, unexpected = sae.load_state_dict(renamed, strict=False)
    if missing:
        print(f"load_sae: missing keys {missing}")
    if unexpected:
        print(f"load_sae: unexpected keys {unexpected}")

    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae


@torch.no_grad()
def encode_sae(sae, activations, device=None):
    """Encode ``[N, d_in]`` activations to ``[N, d_sae]`` codes (numpy)."""
    if device is None:
        device = next(sae.parameters()).device
    x = torch.as_tensor(activations).to(device=device, dtype=next(sae.parameters()).dtype)
    z = sae.encode(x)
    return z.detach().cpu().float().numpy()


def get_decoder(sae):
    """Return the decoder weight as a ``[d_sae, d_in]`` numpy array."""
    if isinstance(sae, SubspaceSAE):
        # SubspaceSAE has no single decoder Linear; assemble the full matrix
        # from the pinned buffer and the free parameter.
        return sae._full_decoder_weight().detach().float().cpu().numpy()
    W = sae.decoder.weight.detach().float().cpu()
    # nn.Linear stores [out, in] = [d_in, d_sae]. Transpose to [d_sae, d_in].
    if W.shape[0] == sae.d_sae:
        return W.numpy()
    return W.T.numpy()


def get_decoder_bias(sae):
    """Return the decoder bias ``[d_in]`` as a numpy array, or None."""
    b = getattr(sae.decoder, "bias", None)
    if b is None:
        return None
    return b.detach().float().cpu().numpy()
