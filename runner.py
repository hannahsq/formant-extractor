# runner.py
import torch
import numpy as np
from probing import probe_all_layers, run_formant_probe, run_formant_regression_per_formant
from viz import plot_layerwise_accuracy, plot_pca_embeddings, plot_umap_embeddings, plot_formant_r2, plot_formant_r2_per_formant, print_eval_report
from utils import labels_to_ints
from train_head import train_multihead, evaluate_multihead, train_multihead_kfold, train_multihead_kfold_phys
from cache_embeddings import get_or_compute_embeddings

def _run_probes(all_layers_embeddings, labels, pooling):
    """
    Shared helper to run probes + visualisations.
    """
    y_ints, mapping = labels_to_ints(labels)
    results = probe_all_layers(all_layers_embeddings, y_ints, pooling=pooling)

    plot_layerwise_accuracy(results)

    # PCA on best layer
    best = max(results, key=lambda r: r['accuracy'])
    best_layer_idx = best['layer']

    X_best = np.stack(all_layers_embeddings[best_layer_idx], axis=0)
    plot_pca_embeddings(X_best, labels, title=f"PCA Layer {best_layer_idx}")

    try:
        plot_umap_embeddings(X_best, labels)
    except Exception:
        pass

    return results, mapping

def run_layerwise_probe_on_hf_dataset(
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    all_layers_embeddings, labels, formants = get_or_compute_embeddings(
        model_name,
        hf_dataset,
        sr=16000,
        pooling=pooling,
        cache_root=cache_root,
        use_cache=use_cache,
        subset=subset,
        nucleus_ms=nucleus_ms
    )

    # Run formant probes
    reg_results = run_formant_probe(all_layers_embeddings, formants)
    plot_formant_r2(reg_results)

    reg_results = run_formant_regression_per_formant(all_layers_embeddings, formants)
    plot_formant_r2_per_formant(reg_results)

    # Run vowel probes
    return _run_probes(all_layers_embeddings, labels, pooling)

def run_train_multihead(
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    embeddings, labels, formants = get_or_compute_embeddings(
        model_name,
        hf_dataset,
        sr=sr,
        pooling=pooling,
        cache_root=cache_root,
        use_cache=use_cache,
        subset=subset,
        nucleus_ms=nucleus_ms
    )

    model = train_multihead(
        embeddings=embeddings,
        labels=labels,
        formants=formants,
        layer_formant=4,
        layer_vowel=12,
        epochs=300
    )

    torch.save(model.state_dict(), "twohead_pooled.pt")
    print("Saved trained pooled‑embedding heads to twohead_pooled.pt")

    return model

def run_eval_multihead(
    model,
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    embeddings, labels, formants = get_or_compute_embeddings(
        model_name,
        hf_dataset,
        sr=sr,
        pooling=pooling,
        cache_root=cache_root,
        use_cache=use_cache,
        subset=subset,
        nucleus_ms=nucleus_ms
    )

    results = evaluate_multihead(
        model,
        embeddings,
        labels,
        formants,
        layer_formant=4,
        layer_vowel=12
    )

    print_eval_report(results)
    return results

def run_train_eval_multihead_kfold(
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    embeddings, labels, formants = get_or_compute_embeddings(
        model_name,
        hf_dataset,
        sr=sr,
        pooling=pooling,
        cache_root=cache_root,
        use_cache=use_cache,
        subset=subset,
        nucleus_ms=nucleus_ms
    )
    
    groups = [item["group"] for item in hf_dataset]

    results = train_multihead_kfold(
        embeddings,
        labels,
        formants,
        groups=groups,
        k=5,
        epochs=300,
        lr=1e-3,
        batch_size=64
    )
    return results

def run_train_eval_multihead_phys_kfold(
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    embeddings, labels, formants = get_or_compute_embeddings(
        model_name,
        hf_dataset,
        sr=sr,
        pooling=pooling,
        cache_root=cache_root,
        use_cache=use_cache,
        subset=subset,
        nucleus_ms=nucleus_ms
    )
    
    groups = [item["group"] for item in hf_dataset]

    results = train_multihead_kfold_phys(
        embeddings,
        labels,
        formants,
        groups=groups,
        layer_formant=4,
        layer_vowel=12,
        k=5,
        epochs=300,
        lr=1e-3,
        batch_size=64
    )
    return results

# def run_layerwise_probe_on_hf_dataset(
#     model_name,
#     hf_dataset,
#     sr=16000,
#     pooling="mean",
#     cache_root="cache",
#     use_cache=True,
#     subset=None,
#     nucleus_ms=100
# ):
#     """
#     Main entry point for probing HF datasets with caching + pooling.
#     """

#     # Build cache parameters
#     cache_params = {
#         "model_name": model_name,
#         "pooling": pooling,
#         "sample_rate": sr,
#         "subset": subset,
#         "nucleus_ms": nucleus_ms,
#         "dataset_size": len(hf_dataset),
#     }

#     cache_dir = get_cache_dir(cache_root, model_name, cache_params)

#     # Try loading cache
#     if use_cache:
#         cached = load_cached_embeddings(cache_dir)
#         if cached is not None:
#             labels = [item["label"] for item in hf_dataset]
#             # After loading cached embeddings
#             for layer_idx, layer in enumerate(cached):
#                 if len(layer) != len(labels):
#                     print(f"Cache incomplete for layer {layer_idx}: "
#                         f"{len(layer)} vs {len(labels)} samples. Ignoring cache.")
#                     cached = None
#                     break
#             print(f"Loaded cached embeddings from {cache_dir}")
#             if subset:
#                 labels = labels[:subset]
#                 cached = [layer[:subset] for layer in cached]
#             formants = np.stack([s["formants"] for s in hf_dataset], axis=0)

#             reg_results = run_formant_probe(cached, formants)
#             plot_formant_r2(reg_results)
#             reg_results = run_formant_regression_per_formant(cached, formants)
#             plot_formant_r2_per_formant(reg_results)
#             return _run_probes(cached, labels, pooling)

#     # Otherwise extract fresh embeddings
#     print("No cache found — extracting embeddings...")
#     model, processor, device = load_model(model_name)

#     if subset:
#         hf_dataset = hf_dataset[:subset]

#     all_layers_embeddings = None
#     labels = []

#     for idx, sample in tqdm(enumerate(hf_dataset)):
#         audio = sample["audio"]
#         label = sample["label"]

#         nucleus = extract_vowel_nucleus(audio, sr=sr, center_ms=nucleus_ms)
#         layer_embs = extract_all_layers(model, processor, nucleus, sample_rate=sr, device=device)

#         # Pool immediately to save RAM
#         pooled = [emb.mean(dim=0).cpu().numpy() for emb in layer_embs]

#         # Save to cache
#         for layer_idx, vec in tqdm(enumerate(pooled)):
#             save_pooled_embedding(cache_dir, layer_idx, idx, vec)

#         # Accumulate for this run
#         if all_layers_embeddings is None:
#             all_layers_embeddings = [[] for _ in pooled]
#         for i, vec in tqdm(enumerate(pooled)):
#             all_layers_embeddings[i].append(vec)

#         labels.append(label)

#     print(f"Saved embeddings to cache: {cache_dir}")
#     formants = np.stack([s["formants"] for s in hf_dataset], axis=0)

#     reg_results = run_formant_probe(all_layers_embeddings, formants)
#     plot_formant_r2(reg_results)

#     return _run_probes(all_layers_embeddings, labels, pooling)


