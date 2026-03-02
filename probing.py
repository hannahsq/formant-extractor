# probing.py
"""
Linear and neural probes for Whisper encoder layers.

Vowel probe
-----------
Uses sklearn's SGDClassifier with partial_fit, so frames are streamed one
sample at a time and the full (N*T, D) matrix is never materialised. Each
frame of a sample is treated as an independent observation with the sample's
vowel label. Results are reported both overall and stratified by speaker group.

Formant regression probe
------------------------
A small PyTorch MLP (two hidden layers, residual connection) trained with
AdamW and early stopping. Operates on pooled (D,) vectors — if sequence
embeddings are passed they are mean-pooled internally before training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Vowel probe  (frame-level, streamed via partial_fit)
# ---------------------------------------------------------------------------

def _make_classifier(n_classes: int, alpha: float = 1e-4) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=1,           # one epoch per partial_fit call
        tol=None,
        random_state=42,
        n_jobs=-1,
    )


def probe_layer(
    sequences: list[np.ndarray],
    labels: list,
    groups: list | None = None,
    alpha: float = 1e-4,
    n_epochs: int = 5,
) -> dict:
    """
    Train a logistic probe on one layer's frame embeddings via partial_fit.

    Parameters
    ----------
    sequences : list of (T, D) arrays, one per sample
    labels    : sample-level vowel labels (repeated across frames internally)
    groups    : optional sample-level group labels for stratified reporting
    alpha     : SGD L2 regularisation strength
    n_epochs  : number of passes through all frames

    Returns
    -------
    dict with keys:
        "accuracy"        : overall frame-level accuracy on training data
        "clf"             : fitted SGDClassifier
        "label_encoder"   : fitted LabelEncoder
        "group_accuracy"  : dict {group: accuracy} if groups provided
    """
    le = LabelEncoder().fit(labels)
    classes = le.classes_
    clf = _make_classifier(n_classes=len(classes), alpha=alpha)

    for _ in range(n_epochs):
        for seq, label in zip(sequences, labels):
            # seq: (T, D) — treat every frame as a sample
            y_frames = np.full(len(seq), le.transform([label])[0])
            clf.partial_fit(seq, y_frames, classes=np.arange(len(classes)))

    # Evaluate: accumulate per-frame predictions, report per-sample majority vote
    all_true, all_pred = [], []
    sample_true, sample_pred = [], []

    for seq, label in zip(sequences, labels):
        frame_preds = clf.predict(seq)
        majority    = np.bincount(frame_preds).argmax()
        all_true.extend([le.transform([label])[0]] * len(seq))
        all_pred.extend(frame_preds.tolist())
        sample_true.append(le.transform([label])[0])
        sample_pred.append(majority)

    result = {
        "clf":           clf,
        "label_encoder": le,
        "frame_accuracy":  float(accuracy_score(all_true, all_pred)),
        "sample_accuracy": float(accuracy_score(sample_true, sample_pred)),
        # "accuracy" kept as the primary metric used by viz / runner
        "accuracy":        float(accuracy_score(sample_true, sample_pred)),
    }

    if groups is not None:
        group_acc = {}
        groups_arr = np.array(groups)
        st = np.array(sample_true)
        sp = np.array(sample_pred)
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            group_acc[g] = float(accuracy_score(st[mask], sp[mask]))
        result["group_accuracy"] = group_acc

    return result


def probe_all_layers(
    layer_embeddings: list[list[np.ndarray]],
    labels: list,
    groups: list | None = None,
    alpha: float = 1e-4,
    n_epochs: int = 5,
) -> list[dict]:
    """
    Run the vowel probe across all encoder layers.

    Parameters
    ----------
    layer_embeddings : [layer][sample] → (T, D) or (D,) arrays
                       — (D,) arrays are treated as single-frame sequences
    labels           : sample-level vowel labels
    groups           : optional sample-level group labels
    alpha            : SGD regularisation strength
    n_epochs         : training epochs per layer

    Returns
    -------
    list of result dicts (one per layer), each with at minimum:
        "layer", "accuracy", "frame_accuracy", "sample_accuracy"
    """
    results = []
    for layer_idx, sequences in enumerate(layer_embeddings):
        # Ensure everything is (T, D) — wrap pooled (D,) vectors
        sequences = [
            s[None, :] if s.ndim == 1 else s
            for s in sequences
        ]
        result = probe_layer(sequences, labels, groups=groups,
                             alpha=alpha, n_epochs=n_epochs)
        result["layer"] = layer_idx
        results.append(result)
        _print_layer_result(layer_idx, result)

    return results


def _print_layer_result(layer_idx: int, result: dict):
    line = (
        f"  Layer {layer_idx:2d}  "
        f"sample_acc={result['sample_accuracy']:.3f}  "
        f"frame_acc={result['frame_accuracy']:.3f}"
    )
    if "group_accuracy" in result:
        group_str = "  |  " + "  ".join(
            f"{g}={v:.3f}" for g, v in sorted(result["group_accuracy"].items())
        )
        line += group_str
    print(line)


# ---------------------------------------------------------------------------
# Formant regression probe  (PyTorch MLP, Gaussian NLL loss)
# ---------------------------------------------------------------------------

class _FormantMLP(nn.Module):
    """Small residual MLP for formant regression."""

    def __init__(self, in_dim: int, out_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h))
        return self.head(self.norm(h + h2))


def _pool_if_needed(sequences: list[np.ndarray]) -> np.ndarray:
    """Mean-pool (T, D) sequences to (N, D); pass (D,) vectors through unchanged."""
    arrs = [s.mean(axis=0) if s.ndim == 2 else s for s in sequences]
    return np.stack(arrs, axis=0)


def _gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Gaussian negative log-likelihood loss.

    loss = mean( log(σ) + (y - μ)² / (2σ²) )

    Samples with high σ (noisy measurements) contribute less to the
    squared error term and are effectively down-weighted during training.

    Parameters
    ----------
    pred   : (N, F) model predictions (μ in whitened space)
    target : (N, F) whitened formant targets
    sigma  : (N, F) whitened formant standard deviations — fixed, not learned
    """
    return torch.mean(
        torch.log(sigma) + (target - pred) ** 2 / (2.0 * sigma ** 2)
    )


def train_formant_mlp(
    X: np.ndarray,
    y_mu: np.ndarray,
    y_sigma: np.ndarray | None = None,
    hidden: int = 256,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 20,
    val_frac: float = 0.15,
) -> tuple[_FormantMLP, float]:
    """
    Train a small MLP to predict formants from pooled embeddings.

    Uses Gaussian NLL loss when y_sigma is provided, falling back to MSE
    otherwise.

    Parameters
    ----------
    X         : (N, D) embedding matrix
    y_mu      : (N, F) whitened formant targets (μ)
    y_sigma   : (N, F) whitened formant standard deviations (σ), optional.
                When provided, samples with high σ are penalised less.
    hidden    : hidden layer width
    epochs    : max training epochs
    lr        : AdamW learning rate
    batch_size: mini-batch size
    patience  : early-stopping patience
    val_frac  : fraction of data held out for early stopping

    Returns
    -------
    (trained model, best validation R² averaged across formants)
    """
    import copy
    from sklearn.metrics import r2_score as sk_r2

    N     = len(X)
    n_val = max(1, int(N * val_frac))
    idx   = np.random.default_rng(42).permutation(N)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    use_nll = y_sigma is not None

    def make_dl(rows, shuffle):
        arrays  = [torch.tensor(X[rows]).float(),
                   torch.tensor(y_mu[rows]).float()]
        dtypes  = [torch.float32, torch.float32]
        if use_nll:
            arrays.append(torch.tensor(y_sigma[rows]).float())
            dtypes.append(torch.float32)
        return DataLoader(
            TensorDataset(*arrays),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_dl = make_dl(tr_idx, shuffle=True)
    val_dl   = make_dl(va_idx, shuffle=False)

    model = _FormantMLP(in_dim=X.shape[1], out_dim=y_mu.shape[1], hidden=hidden)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    best_val   = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience

    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            xb, yb = batch[0], batch[1]
            sb     = batch[2] if use_nll else None
            opt.zero_grad()
            pred = model(xb)
            loss = _gaussian_nll(pred, yb, sb) if use_nll else mse(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                xb, yb = batch[0], batch[1]
                sb     = batch[2] if use_nll else None
                pred   = model(xb)
                val_loss += (
                    _gaussian_nll(pred, yb, sb) if use_nll else mse(pred, yb)
                ).item()
        val_loss /= len(val_dl)

        if val_loss < best_val:
            best_val      = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)

    # Evaluate R² on val set — y_mu is already whitened so compare directly
    model.eval()
    with torch.no_grad():
        pred_white = model(torch.tensor(X[va_idx]).float()).numpy()

    r2 = float(np.mean([
        sk_r2(y_mu[va_idx, i], pred_white[:, i])
        for i in range(y_mu.shape[1])
    ]))
    return model, r2


def run_formant_probe(
    layer_embeddings: list[list[np.ndarray]],
    formants_mu: np.ndarray,
    formants_sigma: np.ndarray | None = None,
    **mlp_kwargs,
) -> list[dict]:
    """
    Train a formant MLP probe on every layer and return mean R².

    Parameters
    ----------
    layer_embeddings : [layer][sample] → (T, D) or (D,) arrays
    formants_mu      : (N, 4) whitened formant targets
    formants_sigma   : (N, 4) whitened formant σ values, optional
    **mlp_kwargs     : forwarded to train_formant_mlp

    Returns
    -------
    list of {"layer": int, "r2": float} dicts
    """
    results = []
    for layer_idx, sequences in enumerate(layer_embeddings):
        X = _pool_if_needed(sequences)
        _, r2 = train_formant_mlp(X, formants_mu, formants_sigma, **mlp_kwargs)
        results.append({"layer": layer_idx, "r2": r2})
    return results


def run_formant_regression_per_formant(
    layer_embeddings: list[list[np.ndarray]],
    formants_mu: np.ndarray,
    formants_sigma: np.ndarray | None = None,
    **mlp_kwargs,
) -> list[dict]:
    """
    Train a separate formant MLP probe per formant on every layer.

    Parameters
    ----------
    layer_embeddings : [layer][sample] → (T, D) or (D,) arrays
    formants_mu      : (N, 4) whitened formant targets
    formants_sigma   : (N, 4) whitened formant σ values, optional

    Returns
    -------
    list of {"layer": int, "r2": [r2_F1, r2_F2, r2_F3, r2_F4]} dicts
    """
    results = []
    for layer_idx, sequences in enumerate(layer_embeddings):
        X   = _pool_if_needed(sequences)
        r2s = []
        for f_idx in range(formants_mu.shape[1]):
            y_f = formants_mu[:, f_idx : f_idx + 1]
            s_f = formants_sigma[:, f_idx : f_idx + 1] if formants_sigma is not None else None
            _, r2 = train_formant_mlp(X, y_f, s_f, **mlp_kwargs)
            r2s.append(r2)
        results.append({"layer": layer_idx, "r2": r2s})
    return results
