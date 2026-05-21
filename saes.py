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


def load_sae(path, d_in=4096, d_sae=None, k=None, expansion_factor=None,
             device=None, dtype=torch.float32):
    """Load a BatchTopK SAE checkpoint.

    Accepts either a raw ``state_dict`` (as produced by
    ``torch.save(sae.state_dict(), ...)``) or a dict containing a
    ``state_dict`` key alongside a ``model_config`` dict with ``d_in``,
    ``d_sae`` (or ``expansion_factor``), and ``k``.

    Args:
        path: path to the checkpoint
        d_in: model activation dim (default: 4096 for Llama-3.1-8B)
        d_sae: explicit SAE width (if not in checkpoint config)
        k: sparsity (if not in checkpoint config)
        expansion_factor: alternative to ``d_sae`` (``d_sae = d_in * factor``)
        device: optional torch device
        dtype: parameter dtype
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Two checkpoint layouts: bare state_dict, or {"state_dict": ..., "model_config": ...}.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        cfg = ckpt.get("model_config", ckpt.get("config", {}))
        d_in = cfg.get("d_in", d_in)
        d_sae = cfg.get("d_sae", d_sae)
        k = cfg.get("k", cfg.get("target_l0", k))
        if d_sae is None and "expansion_factor" in cfg:
            expansion_factor = cfg["expansion_factor"]
    else:
        state_dict = ckpt

    if d_sae is None:
        if expansion_factor is None:
            # Infer from encoder weight shape.
            for key in ("encoder.weight", "encoder_linear.weight", "W_enc"):
                if key in state_dict:
                    d_sae = state_dict[key].shape[0]
                    break
        else:
            d_sae = d_in * expansion_factor
    if d_sae is None:
        raise ValueError("Could not infer d_sae; pass d_sae=... or expansion_factor=...")
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
