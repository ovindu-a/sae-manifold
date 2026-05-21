"""
Background activation extraction for unsupervised feature clustering.

Streams text from a HuggingFace dataset through Llama-3.1-8B (via nnsight),
collects layer-19 hidden states, and saves them as a memory-mapped float32
array of shape ``[n_tokens, d_model]``. ``unsupervised_clustering.py`` then
encodes these through the SAE to build feature similarity matrices.

Usage:
  uv run background.py --n-tokens 500000             # default dataset (C4 en)
  uv run background.py --n-tokens 200000 --dataset monology/pile-uncopyrighted

The activations cache lives at ``cache/background_acts_{n_tokens}.dat`` with
a sibling ``.shape.json``. Existing caches are reused — pass ``--force`` to
overwrite.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from data import MODEL_NAME, LAYER, D_MODEL, CACHE_DIR, load_llm


def background_cache_path(n_tokens):
    return CACHE_DIR / f"background_acts_{n_tokens}.dat"


def _stream_texts(dataset_name, dataset_config, split, text_field):
    """Yield short text snippets from a streaming HF dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    for row in ds:
        text = row.get(text_field, "")
        if not text or len(text) < 50:
            continue
        yield text


@torch.no_grad()
def extract_background(n_tokens=500_000,
                       dataset="allenai/c4",
                       dataset_config="en",
                       split="train",
                       text_field="text",
                       max_seq_len=512,
                       force=False):
    """Extract layer-{LAYER} activations from a streaming text dataset.

    Returns the path to a ``[n_tokens, d_model]`` float32 memmap on disk.
    """
    cache = background_cache_path(n_tokens)
    shape_path = cache.with_suffix(".shape.json")
    if cache.exists() and shape_path.exists() and not force:
        shape = tuple(json.load(open(shape_path)))
        print(f"Reusing cached background activations: {cache} {shape}")
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_llm()

    out = np.memmap(cache, dtype=np.float32, mode="w+",
                    shape=(n_tokens, D_MODEL))
    written = 0
    pbar = tqdm(total=n_tokens, desc="background activations")
    for text in _stream_texts(dataset, dataset_config, split, text_field):
        if written >= n_tokens:
            break
        toks = tokenizer(text, truncation=True, max_length=max_seq_len,
                         return_tensors="pt")
        seq_len = toks["input_ids"].shape[1]
        if seq_len < 8:
            continue
        with model.trace(toks) as tracer:
            acts = model.model.layers[LAYER].output[0].clone().save()
            tracer.stop()
        acts = acts.squeeze(0).float().cpu().numpy()  # [seq_len, d_model]
        # Skip the BOS position to avoid the well-known attention sink.
        acts = acts[1:]
        take = min(acts.shape[0], n_tokens - written)
        out[written:written + take] = acts[:take]
        written += take
        pbar.update(take)
    pbar.close()
    out.flush()
    del out

    # Truncate if the stream ran out of tokens early.
    if written < n_tokens:
        tmp_path = cache.with_suffix(".tmp.dat")
        tmp = np.memmap(cache, dtype=np.float32, mode="r",
                        shape=(n_tokens, D_MODEL))
        final = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                          shape=(written, D_MODEL))
        for i in range(0, written, 10_000):
            final[i:i + 10_000] = tmp[i:i + 10_000]
        del tmp, final
        cache.unlink()
        tmp_path.rename(cache)

    shape = (written, D_MODEL)
    with open(shape_path, "w") as f:
        json.dump(shape, f)
    print(f"Wrote {written} background activations to {cache}")
    return cache


def load_background(n_tokens):
    """Return the ``[N, d_model]`` mmap, or raise if not extracted yet."""
    cache = background_cache_path(n_tokens)
    shape_path = cache.with_suffix(".shape.json")
    if not (cache.exists() and shape_path.exists()):
        raise FileNotFoundError(
            f"No background activations at {cache}. "
            f"Run: uv run background.py --n-tokens {n_tokens}"
        )
    shape = tuple(json.load(open(shape_path)))
    return np.memmap(cache, dtype=np.float32, mode="r", shape=shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-tokens", type=int, default=500_000)
    p.add_argument("--dataset", type=str, default="allenai/c4")
    p.add_argument("--dataset-config", type=str, default="en")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    extract_background(
        n_tokens=a.n_tokens, dataset=a.dataset,
        dataset_config=a.dataset_config, split=a.split,
        text_field=a.text_field, max_seq_len=a.max_seq_len,
        force=a.force,
    )
