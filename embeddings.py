# embeddings.py
"""
High-level embedding pipeline.

Combines the Whisper encoder, vowel nucleus extraction, and disk cache
into a single get_or_compute_embeddings call consumed by the runner.

Memory-efficient two-pass extraction
-------------------------------------
Rather than accumulating all layers in RAM, extraction writes one small
compressed file per sample (Pass 1), then consolidates those files into
per-layer layer_NN.npz files (Pass 2). Peak memory during extraction is
one sample's embeddings; peak memory during consolidation is one full
layer (N, T, D) float16.

Return value
------------
get_or_compute_embeddings returns a lazy iterator over layers rather than
a materialised list. Each iteration yields (layer_idx, list[np.ndarray])
and loads only that layer from disk, allowing the caller to GC it before
moving to the next layer.

Pooling strategies
------------------
"mean"  : average over the time dimension  (default)
"max"   : element-wise max over time
"first" : first time step (CLS-like)
"last"  : last time step
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import numpy as np
from tqdm import tqdm

from whisper import WhisperEncoder
from dataset import extract_vowel_nucleus
from embedding_cache import (
    cache_dir_for,
    n_cached_layers,
    load_layer,
    save_sample,
    consolidate_samples,
    iter_layers,
)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

_POOLING_FNS = {
    "mean":  lambda h: h.mean(axis=0),
    "max":   lambda h: h.max(axis=0),
    "first": lambda h: h[0],
    "last":  lambda h: h[-1],
}


def pool(hidden: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Pool a (T, D) hidden-state array to a (D,) vector."""
    try:
        return _POOLING_FNS[strategy](hidden)
    except KeyError:
        raise ValueError(
            f"Unknown pooling strategy '{strategy}'. "
            f"Choose from: {list(_POOLING_FNS)}"
        )


def pool_layer_embeddings(
    layer_iter,
    strategy: str = "mean",
) -> list[list[np.ndarray]]:
    """
    Pool a layer iterator to flat vectors, materialising into a list.

    Parameters
    ----------
    layer_iter : iterable of (layer_idx, list[(T, D)]) — from iter_layers
                 or any compatible source
    strategy   : pooling strategy

    Returns
    -------
    [layer][sample] -> (D,) arrays  (fully in memory)
    """
    result = []
    for _layer_idx, samples in layer_iter:
        result.append([pool(s, strategy) for s in samples])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_compute_embeddings(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    pooling: str = "mean",
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
    store_sequences: bool = True,
) -> tuple[Generator, list[str], np.ndarray]:
    """
    Return a lazy layer iterator, labels, and formants for a dataset.

    On a cache hit the iterator streams layers directly from disk.
    On a cache miss the encoder runs in Pass 1 (per-sample temp files),
    then Pass 2 (layer consolidation), then streams from the new cache.
    Either way peak RAM is one layer at a time.

    Parameters
    ----------
    model_name      : HuggingFace Whisper model identifier
    hf_dataset      : iterable of dicts with "audio", "label", "formants"
    sr              : audio sample rate
    pooling         : pooling strategy — only used when store_sequences=False
    cache_root      : root directory for the embedding cache
    use_cache       : whether to read from / write to disk cache
    subset          : if set, use only the first N samples
    nucleus_ms      : vowel nucleus window in milliseconds
    store_sequences : if True (default), cache (T, D) sequences;
                      if False, cache pooled (D,) vectors

    Returns
    -------
    layer_iter : generator yielding (layer_idx, list[np.ndarray]) one layer
                 at a time — each array is (T, D) or (D,) depending on mode
    labels     : list[str] of per-sample vowel labels
    formants   : (N, 4) float32 array of per-sample formant means
    """
    if subset:
        hf_dataset = hf_dataset[:subset]

    labels   = [item["label"]   for item in hf_dataset]
    formants = np.stack([item["formants"] for item in hf_dataset], axis=0)

    cache_params = {
        "model_name":      model_name,
        "sample_rate":     sr,
        "nucleus_ms":      nucleus_ms,
        "dataset_size":    len(hf_dataset),
        "store_sequences": store_sequences,
        **({"pooling": pooling} if not store_sequences else {}),
    }

    cache_dir = cache_dir_for(cache_root, model_name, cache_params)

    # --- Cache hit ---
    if use_cache and n_cached_layers(cache_dir) > 0:
        probe = load_layer(cache_dir, 0)
        if probe is not None and len(probe) == len(labels):
            print(f"Streaming cached embeddings from {cache_dir}")
            return iter_layers(cache_dir), labels, formants
        print("Cache found but sample count mismatch — re-extracting.")

    # --- No cache requested: extract directly into memory ---
    # Only sensible for small datasets / debugging — avoids disk entirely.
    if not use_cache:
        encoder = WhisperEncoder(model_name)
        _mem: list[list[np.ndarray]] | None = None
        for idx, sample in tqdm(enumerate(hf_dataset), total=len(hf_dataset),
                                desc="Extracting (no cache)"):
            nucleus    = extract_vowel_nucleus(sample["audio"], sr=sr,
                                              center_ms=nucleus_ms)
            layer_embs = encoder.extract_all_layers(nucleus, sample_rate=sr)
            if not store_sequences:
                layer_embs = [pool(h, pooling) for h in layer_embs]
            if _mem is None:
                _mem = [[] for _ in layer_embs]
            for li, arr in enumerate(layer_embs):
                _mem[li].append(arr)
            del layer_embs
        return iter(enumerate(_mem)), labels, formants

    # --- Cache miss: two-pass extraction ---
    import gc
    print(f"Extracting embeddings for {model_name} ({len(hf_dataset)} samples)...")
    encoder = WhisperEncoder(model_name)

    with tempfile.TemporaryDirectory(prefix="whisper_emb_") as temp_dir:
        # Pass 1: extract one sample at a time, write to temp file, free immediately
        for idx, sample in tqdm(enumerate(hf_dataset), total=len(hf_dataset),
                                desc="Pass 1 — extracting"):
            nucleus    = extract_vowel_nucleus(sample["audio"], sr=sr,
                                              center_ms=nucleus_ms)
            layer_embs = encoder.extract_all_layers(nucleus, sample_rate=sr)
            if not store_sequences:
                layer_embs = [pool(h, pooling) for h in layer_embs]
            save_sample(temp_dir, idx, layer_embs)
            del layer_embs
            if idx % 50 == 0:
                gc.collect()

        # Pass 2: consolidate per-sample files into per-layer cache files.
        # temp_dir is still alive here — context manager hasn't exited yet.
        consolidate_samples(temp_dir, cache_dir, labels, delete_temp=False)
        # temp_dir cleaned up automatically when the with block exits

    print(f"Cache ready at {cache_dir}")
    return iter_layers(cache_dir), labels, formants
