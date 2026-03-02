# utils.py
import numpy as np

def labels_to_ints(labels):
    unique = sorted(list(set(labels)))
    mapping = {u:i for i,u in enumerate(unique)}
    ints = [mapping[l] for l in labels]
    return ints, mapping

def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]

# utils.py
def labels_to_ints(labels):
    unique = sorted(list(set(labels)))
    mapping = {u: i for i, u in enumerate(unique)}
    ints = [mapping[l] for l in labels]
    return ints, mapping

def build_concat_embeddings(all_layers_embeddings, layers):
    """
    all_layers_embeddings: list of length L, each entry is a list of N vectors
    layers: list of layer indices to concatenate
    returns: (N, D_total) array
    """
    mats = [np.stack(all_layers_embeddings[l], axis=0) for l in layers]
    return np.concatenate(mats, axis=1)