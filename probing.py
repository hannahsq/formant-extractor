# probing.py
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

# def flatten_embeddings_for_probe(embeddings_list, pooling="mean"):
#     """
#     embeddings_list: list of tensors shape (time_steps, dim)
#     pooling: 'mean' or 'max' or 'center'
#     returns: 2D numpy array (n_samples, dim)
#     """
#     pooled = []
#     for emb in embeddings_list:
#         if pooling == "mean":
#             pooled.append(emb.mean(axis=0))
#         elif pooling == "max":
#             pooled.append(emb.max(axis=0))
#         elif pooling == "center":
#             pooled.append(emb[emb.shape[0] // 2])
#         else:
#             pooled.append(emb.mean(axis=0))
#     return np.stack(pooled, axis=0)

def flatten_embeddings_for_probe(emb_list, pooling="mean"):
    """
    emb_list:
        - if sequence-level: list of arrays with shape (T, D)
        - if already pooled: list of arrays with shape (D,)
    returns:
        X with shape (n_samples, D)
    """
    arr = np.array(emb_list)

    # Case 1: already pooled → (n_samples, dim)
    if arr.ndim == 2:
        return arr

    # Case 2: sequence embeddings → (n_samples, T, dim)
    if arr.ndim == 3:
        if pooling == "mean":
            return arr.mean(axis=1)
        elif pooling == "max":
            return arr.max(axis=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    raise ValueError(f"Unexpected embedding shape: {arr.shape}")


def train_probe(X, y, C=1.0, cv=5):
    """
    X: (n_samples, dim)
    y: (n_samples,)
    returns: trained model and cross-validated accuracy
    """
    clf = LogisticRegression(max_iter=2000, C=C)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    return clf, float(scores.mean())

def probe_all_layers(layer_embeddings_list, labels, pooling="mean", C=1.0, cv=5):
    """
    layer_embeddings_list: list of lists: layers x samples -> tensor
    labels: list of labels length n_samples
    returns: list of (clf, accuracy) per layer
    """
    results = []
    n_layers = len(layer_embeddings_list)
    for layer_idx in range(n_layers):
        emb_list = layer_embeddings_list[layer_idx]  # list of tensors per sample
        X = flatten_embeddings_for_probe(emb_list, pooling=pooling)
        clf, acc = train_probe(X, labels, C=C, cv=cv)
        results.append({'layer': layer_idx, 'clf': clf, 'accuracy': acc})
    return results

def train_regression_probe(X, y, alpha=1.0, cv=5):
    """
    X: (n_samples, dim)
    y: (n_samples, n_targets) e.g. F1-F4
    Returns: model, mean R^2 score
    """
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    model.fit(X, y)
    return model, float(scores.mean())

def run_formant_probe(all_layers_embeddings, formants):
    """
    all_layers_embeddings: list of layers, each a list of pooled vectors
    formants: array shape (n_samples, 4)
    """
    results = []
    for layer_idx, emb_list in enumerate(all_layers_embeddings):
        X = np.stack(emb_list, axis=0)
        model, r2 = train_regression_probe(X, formants)
        results.append({'layer': layer_idx, 'r2': r2})
    return results

def run_formant_regression_per_formant(all_layers_embeddings, formants):
    """
    formants: shape (n_samples, 4)
    returns: list of dicts per layer with r2 for each formant
    """
    results = []
    for layer_idx, emb_list in enumerate(all_layers_embeddings):
        X = np.stack(emb_list, axis=0)

        layer_result = {"layer": layer_idx, "r2": []}
        for f_idx in range(formants.shape[1]):
            y = formants[:, f_idx]
            model, r2 = train_regression_probe(X, y.reshape(-1, 1))
            layer_result["r2"].append(r2)

        results.append(layer_result)

    return results
