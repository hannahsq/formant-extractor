# runner.py
from tqdm import tqdm
import numpy as np
from embedding_extraction import load_model, extract_all_layers
from dataset import extract_vowel_nucleus
from cache_utils import (
    get_cache_dir,
    load_cached_embeddings,
    save_pooled_embedding
)

def get_or_compute_embeddings(
    model_name,
    hf_dataset,
    sr=16000,
    pooling="mean",
    cache_root="cache",
    use_cache=True,
    subset=None,
    nucleus_ms=100
):
    """
    Returns:
        all_layers_embeddings: list of layers, each a list of pooled vectors
        labels: list of vowel labels
        formants: array (n_samples, 4)
    """

    # Build cache parameters
    cache_params = {
        "model_name": model_name,
        "pooling": pooling,
        "sample_rate": sr,
        "subset": subset,
        "nucleus_ms": nucleus_ms,
        "dataset_size": len(hf_dataset),
    }

    cache_dir = get_cache_dir(cache_root, model_name, cache_params)

    # Try loading cache
    if use_cache:
        cached = load_cached_embeddings(cache_dir)
        if cached is not None:
            labels = [item["label"] for item in hf_dataset]

            # Cache integrity check
            for layer_idx, layer in enumerate(cached):
                if len(layer) != len(labels):
                    print(f"Cache incomplete for layer {layer_idx}: "
                          f"{len(layer)} vs {len(labels)} samples. Ignoring cache.")
                    cached = None
                    break

            if cached is not None:
                print(f"Loaded cached embeddings from {cache_dir}")

                if subset:
                    labels = labels[:subset]
                    cached = [layer[:subset] for layer in cached]

                formants = np.stack([s["formants"] for s in hf_dataset], axis=0)
                if subset:
                    formants = formants[:subset]

                return cached, labels, formants

    # Otherwise compute fresh embeddings
    print("No cache found — extracting embeddings...")
    model, processor, device = load_model(model_name)

    if subset:
        hf_dataset = hf_dataset[:subset]

    all_layers_embeddings = None
    labels = []

    for idx, sample in tqdm(enumerate(hf_dataset)):
        audio = sample["audio"]
        label = sample["label"]

        nucleus = extract_vowel_nucleus(audio, sr=sr, center_ms=nucleus_ms)
        layer_embs = extract_all_layers(model, processor, nucleus, sample_rate=sr, device=device)

        pooled = [emb.mean(dim=0).cpu().numpy() for emb in layer_embs]

        # Save to cache
        for layer_idx, vec in enumerate(pooled):
            save_pooled_embedding(cache_dir, layer_idx, idx, vec)

        # Accumulate
        if all_layers_embeddings is None:
            all_layers_embeddings = [[] for _ in pooled]
        for i, vec in enumerate(pooled):
            all_layers_embeddings[i].append(vec)

        labels.append(label)

    print(f"Saved embeddings to cache: {cache_dir}")

    formants = np.stack([s["formants"] for s in hf_dataset], axis=0)

    return all_layers_embeddings, labels, formants