# cache_utils.py
import os
import json
import hashlib
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_cache_key(params: dict):
    """
    Create a stable hash from a dictionary of parameters.
    """
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def get_cache_dir(cache_root, model_name, params):
    """
    Returns a directory path for this model + parameter configuration.
    """
    key = make_cache_key(params)
    model_dir = model_name.replace("/", "_")
    return os.path.join(cache_root, model_dir, key)

def save_pooled_embedding(cache_dir, layer_idx, sample_idx, vector):
    """
    Save a pooled embedding vector to disk.
    """
    layer_dir = os.path.join(cache_dir, f"layer_{layer_idx:02d}")
    ensure_dir(layer_dir)
    np.save(os.path.join(layer_dir, f"{sample_idx:04d}.npy"), vector)

def load_cached_embeddings(cache_dir):
    """
    Load cached pooled embeddings if available.
    Returns: list of layers, each a list of vectors.
    """
    if not os.path.exists(cache_dir):
        return None

    layers = sorted(os.listdir(cache_dir))
    all_layers = []

    for layer in layers:
        layer_dir = os.path.join(cache_dir, layer)
        if not os.path.isdir(layer_dir):
            continue
        files = sorted(os.listdir(layer_dir))
        vectors = [np.load(os.path.join(layer_dir, f)) for f in files]
        all_layers.append(vectors)

    return all_layers
