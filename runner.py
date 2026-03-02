# runner.py
"""
High-level entry points for the Whisper formant probe experiments.

Typical usage
-------------
    from datasets_hf import load_hillenbrand
    from runner import run_probe, run_vowel_probe, run_formant_probe
    from runner import run_kfold, run_kfold_phys, run_averaged_model

    dataset = load_hillenbrand()

    # Both probes in one call
    run_probe("openai/whisper-small", dataset)

    # Or independently
    run_formant_probe("openai/whisper-small", dataset)
    run_vowel_probe("openai/whisper-small", dataset)

    # k-fold cross-validation (ThreeHead / sample-VTL pipeline)
    results = run_kfold("openai/whisper-small", dataset)

    # k-fold cross-validation (PhysHead / blended-VTL pipeline)
    results = run_kfold_phys("openai/whisper-small", dataset)

    # Train, weight-average, fine-tune, and save a production model
    model = run_averaged_model("openai/whisper-small", dataset, save_path="model.pt")
"""

from __future__ import annotations

import numpy as np
import torch

from embeddings import get_or_compute_embeddings, pool, pool_layer_embeddings
from probing import probe_all_layers
from training import ThreeHeadTrainer, PhysHeadTrainer
from utils import labels_to_ints
from viz import (
    plot_layerwise_accuracy,
    plot_pca_embeddings,
    plot_umap_embeddings,
    plot_formant_r2,
    plot_formant_r2_per_formant,
)


# ---------------------------------------------------------------------------
# Shared embedding config (passed through to every runner)
# ---------------------------------------------------------------------------

def _get_embeddings(model_name, hf_dataset, sr, pooling,
                    cache_root, use_cache, subset, nucleus_ms):
    """
    Return pooled (D,) embeddings as a materialised list for the multihead trainers.

    Tries to derive pooled embeddings from the sequence cache (store_sequences=True)
    to avoid re-running the encoder. Falls back to a dedicated pooled cache only
    if no sequence cache exists.
    """
    ds = hf_dataset[:subset] if subset else hf_dataset

    # Try sequence cache first — pool on the fly, much cheaper than re-extracting
    seq_iter, labels, formants = get_or_compute_embeddings(
        model_name, ds,
        sr=sr, cache_root=cache_root, use_cache=use_cache,
        subset=None, nucleus_ms=nucleus_ms,
        store_sequences=True,
    )
    embeddings = [
        [pool(s, pooling) for s in samples]
        for _, samples in seq_iter
    ]
    return embeddings, labels, formants


def _get_sequence_embeddings(model_name, hf_dataset, sr,
                              cache_root, use_cache, subset, nucleus_ms):
    """Return a lazy layer iterator over (T, D) sequences."""
    return get_or_compute_embeddings(
        model_name, hf_dataset,
        sr=sr,
        cache_root=cache_root, use_cache=use_cache,
        subset=subset, nucleus_ms=nucleus_ms,
        store_sequences=True,
    )


def _groups_from(hf_dataset) -> list:
    return [item["group"] for item in hf_dataset]


# ---------------------------------------------------------------------------
# Linear probe sweep
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Probe entry points
# ---------------------------------------------------------------------------

def _probe_formant_common(ds, layer_iter, labels, formants):
    """
    Shared internals for run_formant_probe: stream one layer at a time,
    train the MLP probe, collect results, then plot.
    """
    from preprocessing import FormantPreprocessor
    from probing import train_formant_mlp, _pool_if_needed

    formants_arr = np.asarray(formants)
    pre          = FormantPreprocessor(mode="sample").fit(formants_arr)
    mu_w         = pre.transform_formants(formants_arr)
    sigma_w      = None
    if ds and "formants_std" in ds[0]:
        sigma_w = pre.transform_formant_sigma(
            np.stack([s["formants_std"] for s in ds], axis=0)
        )

    overall_results     = []
    per_formant_results = []

    for layer_idx, sequences in layer_iter:
        X = _pool_if_needed(sequences)
        del sequences

        # Overall mean R²
        model, r2_mean = train_formant_mlp(X, mu_w, sigma_w)
        overall_results.append({"layer": layer_idx, "r2": r2_mean})
        del model

        # Per-formant R²
        r2s = []
        for f_idx in range(mu_w.shape[1]):
            s_f = sigma_w[:, f_idx:f_idx+1] if sigma_w is not None else None
            model, r2 = train_formant_mlp(X, mu_w[:, f_idx:f_idx+1], s_f)
            r2s.append(r2)
            del model
        per_formant_results.append({"layer": layer_idx, "r2": r2s})
        del X

        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return overall_results, per_formant_results


def run_formant_probe(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
) -> list[dict]:
    """
    Run the formant regression probe (MLP + Gaussian NLL) across all layers,
    streaming one layer at a time.

    Produces:
      - formant R² plot (mean across F1-F4)
      - per-formant R² plot

    Returns
    -------
    list of {"layer": int, "r2": float} dicts
    """
    ds = hf_dataset[:subset] if subset else hf_dataset
    layer_iter, labels, formants = _get_sequence_embeddings(
        model_name, ds, sr, cache_root, use_cache, subset=None,
        nucleus_ms=nucleus_ms,
    )
    overall, per_formant = _probe_formant_common(ds, layer_iter, labels, formants)
    plot_formant_r2(overall)
    plot_formant_r2_per_formant(per_formant)
    return overall


def run_vowel_probe(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
) -> tuple[list[dict], dict]:
    """
    Run the frame-level vowel probe (SGDClassifier) across all layers,
    streaming one layer at a time. PCA/UMAP are generated for the best
    layer by loading only that layer from disk.

    Produces:
      - layerwise accuracy plot
      - PCA plot of the best layer
      - UMAP plot of the best layer (skipped if umap-learn absent)

    Returns
    -------
    (probe_results, label_mapping)
    """
    from probing import probe_layer
    from embedding_cache import load_layer, cache_dir_for

    ds     = hf_dataset[:subset] if subset else hf_dataset
    groups = _groups_from(ds)

    layer_iter, labels, formants = _get_sequence_embeddings(
        model_name, ds, sr, cache_root, use_cache, subset=None,
        nucleus_ms=nucleus_ms,
    )
    _, mapping = labels_to_ints(labels)

    results = []
    for layer_idx, sequences in layer_iter:
        seqs = [s[None, :] if s.ndim == 1 else s for s in sequences]
        result = probe_layer(seqs, labels, groups=groups)
        result["layer"] = layer_idx
        results.append(result)
        from viz import _print_layer_result  # local to avoid circular
        _print_layer_result(layer_idx, result)
        del sequences, seqs

    plot_layerwise_accuracy(results)

    # PCA / UMAP: load only the best layer from disk
    best_layer = max(results, key=lambda r: r["accuracy"])["layer"]
    from embedding_cache import cache_dir_for as _cdir
    from embeddings import get_or_compute_embeddings as _gce, pool
    cache_params = {
        "model_name":      model_name,
        "sample_rate":     sr,
        "nucleus_ms":      nucleus_ms,
        "dataset_size":    len(ds),
        "store_sequences": True,
    }
    cache_dir  = cache_dir_for("cache", model_name, cache_params)
    best_seqs  = load_layer(cache_dir, best_layer)
    if best_seqs is not None:
        X_best = np.stack([pool(s, "mean") for s in best_seqs], axis=0)
        plot_pca_embeddings(X_best, labels, title=f"PCA Layer {best_layer}")
        try:
            plot_umap_embeddings(X_best, labels)
        except Exception:
            pass
        del best_seqs, X_best

    return results, mapping


def run_probe(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
) -> tuple[list[dict], dict]:
    """
    Convenience wrapper: run both formant and vowel probes.

    Returns
    -------
    (vowel_probe_results, label_mapping)
    """
    run_formant_probe(
        model_name, hf_dataset, sr=sr,
        cache_root=cache_root, use_cache=use_cache,
        subset=subset, nucleus_ms=nucleus_ms,
    )
    return run_vowel_probe(
        model_name, hf_dataset, sr=sr,
        cache_root=cache_root, use_cache=use_cache,
        subset=subset, nucleus_ms=nucleus_ms,
    )


# ---------------------------------------------------------------------------
# k-fold trainers
# ---------------------------------------------------------------------------

def run_kfold(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    pooling: str = "mean",
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
    trainer_kwargs: dict | None = None,
) -> list[dict]:
    """
    k-fold cross-validation using ThreeHeadPooled (sample-mode VTL pipeline).

    Parameters
    ----------
    trainer_kwargs : passed directly to ThreeHeadTrainer (e.g. k, epochs, lr)

    Returns
    -------
    list of per-fold result dicts
    """
    embeddings, labels, formants = _get_embeddings(
        model_name, hf_dataset, sr, pooling,
        cache_root, use_cache, subset, nucleus_ms
    )
    groups  = _groups_from(hf_dataset[:subset] if subset else hf_dataset)
    trainer = ThreeHeadTrainer(**(trainer_kwargs or {}))
    return trainer.fit(embeddings, labels, np.asarray(formants), groups)


def run_kfold_phys(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    pooling: str = "mean",
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
    trainer_kwargs: dict | None = None,
) -> list[dict]:
    """
    k-fold cross-validation using TwoHeadPooledPhys (blended-mode VTL pipeline).

    Parameters
    ----------
    trainer_kwargs : passed directly to PhysHeadTrainer (e.g. k, epochs, alpha)

    Returns
    -------
    list of per-fold result dicts
    """
    embeddings, labels, formants = _get_embeddings(
        model_name, hf_dataset, sr, pooling,
        cache_root, use_cache, subset, nucleus_ms
    )
    groups  = _groups_from(hf_dataset[:subset] if subset else hf_dataset)
    trainer = PhysHeadTrainer(**(trainer_kwargs or {}))
    return trainer.fit(embeddings, labels, np.asarray(formants), groups)


# ---------------------------------------------------------------------------
# Averaged production model
# ---------------------------------------------------------------------------

def run_averaged_model(
    model_name: str,
    hf_dataset,
    sr: int = 16000,
    pooling: str = "mean",
    cache_root: str = "cache",
    use_cache: bool = True,
    subset: int | None = None,
    nucleus_ms: int = 100,
    save_path: str = "model.pt",
    trainer_kwargs: dict | None = None,
    use_phys_pipeline: bool = False,
) -> torch.nn.Module:
    """
    Run k-fold, average fold weights, fine-tune on the full dataset, and save.

    Parameters
    ----------
    save_path          : path to write the final state_dict (.pt)
    trainer_kwargs     : passed to the trainer constructor
    use_phys_pipeline  : if True, use PhysHeadTrainer; otherwise ThreeHeadTrainer

    Returns
    -------
    Fine-tuned nn.Module
    """
    embeddings, labels, formants = _get_embeddings(
        model_name, hf_dataset, sr, pooling,
        cache_root, use_cache, subset, nucleus_ms
    )
    groups = _groups_from(hf_dataset[:subset] if subset else hf_dataset)

    TrainerClass = PhysHeadTrainer if use_phys_pipeline else ThreeHeadTrainer
    trainer = TrainerClass(**(trainer_kwargs or {}))
    model   = trainer.fit_averaged(embeddings, labels, np.asarray(formants), groups)

    torch.save(model.state_dict(), save_path)
    print(f"Saved averaged + fine-tuned model to {save_path}")
    return model
