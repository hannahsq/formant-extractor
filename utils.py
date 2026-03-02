# utils.py
import numpy as np


def labels_to_ints(labels: list) -> tuple[list[int], dict]:
    """Map string labels to contiguous integers. Returns (ints, mapping)."""
    unique = sorted(set(labels))
    mapping = {u: i for i, u in enumerate(unique)}
    return [mapping[l] for l in labels], mapping


def build_concat_embeddings(all_layers_embeddings, layers: list[int]) -> np.ndarray:
    """
    Concatenate pooled embeddings from selected encoder layers.

    Parameters
    ----------
    all_layers_embeddings : list[list[np.ndarray]]
        Outer list indexed by layer, inner list indexed by sample.
    layers : list[int]
        Layer indices to concatenate. Negative indices are resolved from the
        end (e.g. -1 = last layer). Out-of-range indices are clamped to the
        last available layer with a warning.

    Returns
    -------
    np.ndarray of shape (N, D_total)
    """
    n = len(all_layers_embeddings)
    resolved = []
    for l in layers:
        idx = l if l >= 0 else n + l
        if idx < 0 or idx >= n:
            import warnings
            clamped = max(0, min(idx, n - 1))
            warnings.warn(
                f"Layer index {l} out of range for model with {n} layers "
                f"— clamping to {clamped}.",
                stacklevel=3,
            )
            idx = clamped
        resolved.append(idx)
    mats = [np.stack(all_layers_embeddings[i], axis=0) for i in resolved]
    return np.concatenate(mats, axis=1)


def batch_iterable(iterable, batch_size: int):
    """Yield successive slices of `iterable` of length `batch_size`."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]
