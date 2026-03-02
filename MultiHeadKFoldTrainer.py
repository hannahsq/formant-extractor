import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from utils import labels_to_ints
from heads import ThreeHeadPooled


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class FormantPreprocessor:
    """Fits a joint whitening transform on formant data."""

    def __init__(self):
        self.mean_ = None
        self.inv_sqrt_cov_ = None
        self.invL_mean_ = None
        self.invL_std_ = None

    def fit(self, F: np.ndarray) -> "FormantPreprocessor":
        F = np.asarray(F)
        self.mean_ = F.mean(axis=0)
        F_center = F - self.mean_
        cov = np.cov(F_center, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        self.inv_sqrt_cov_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        invL = self._vtl_inv(F)
        self.invL_mean_ = invL.mean()
        self.invL_std_ = invL.std()
        return self

    def transform_formants(self, F: np.ndarray) -> np.ndarray:
        return (F - self.mean_) @ self.inv_sqrt_cov_

    def inverse_transform_formants(self, F_white: np.ndarray) -> np.ndarray:
        return F_white @ np.linalg.inv(self.inv_sqrt_cov_) + self.mean_

    def transform_vtl(self, F: np.ndarray) -> np.ndarray:
        invL = self._vtl_inv(F)
        return (invL - self.invL_mean_) / self.invL_std_

    def inverse_transform_vtl(self, invL_norm: np.ndarray) -> np.ndarray:
        invL = invL_norm * self.invL_std_ + self.invL_mean_
        return 1.0 / invL

    @staticmethod
    def _vtl_inv(F: np.ndarray) -> np.ndarray:
        L_if = 350 * np.array([1, 3, 5, 7])[None, :] / (4 * F)
        L_sample = np.median(L_if[:, :3], axis=1)
        return 1.0 / L_sample

    @staticmethod
    def vtl_from_formants(F: np.ndarray) -> np.ndarray:
        L_if = 350 * np.array([1, 3, 5, 7])[None, :] / (4 * F)
        return np.median(L_if[:, :3], axis=1)


class EmbeddingBuilder:
    """Concatenates encoder-layer embeddings by layer index."""

    def __init__(self, layers: list[int]):
        self.layers = layers

    def build(self, layer_embeddings, layers:list[int]) -> np.ndarray:
        mats = [np.stack(layer_embeddings[l], axis=0) for l in layers]
        return np.concatenate(mats, axis=1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class FoldTrainer:
    """Runs a single training run (one fold) with early stopping."""

    def __init__(self, epochs: int = 200, lr: float = 1e-3,
                 batch_size: int = 64, patience: int = 20):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

    def train(self, model: nn.Module, train_ds: TensorDataset,
              val_ds: TensorDataset) -> nn.Module:
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        patience_left = self.patience

        for _ in range(self.epochs):
            model.train()
            for xb_phys, xb_vowel, yv, yf, yL in train_dl:
                opt.zero_grad()
                out = model(xb_phys, xb_vowel)
                loss = (
                    ce(out["vowels"], yv)
                    + mse(out["formants"], yf)
                    + 0.5 * mse(out["vtl"], yL)
                )
                loss.backward()
                opt.step()

            val_loss = self._validate(model, val_dl, ce, mse)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)
        return model

    @staticmethod
    def _validate(model: nn.Module, val_dl: DataLoader,
                  ce: nn.Module, mse: nn.Module) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb_phys, xb_vowel, yv, yf, yL in val_dl:
                out = model(xb_phys, xb_vowel)
                total += (
                    ce(out["vowels"], yv)
                    + mse(out["formants"], yf)
                    + 0.5 * mse(out["vtl"], yL)
                ).item()
        return total / len(val_dl)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class FoldEvaluator:
    """Runs inference on a fold and returns decoded metrics and group stats."""

    def evaluate(self, model: nn.Module, Xp_va: np.ndarray, Xv_va: np.ndarray,
                 yv_va: np.ndarray, formants_va: np.ndarray,
                 preprocessor: FormantPreprocessor,
                 groups_va: np.ndarray) -> dict:

        model.eval()
        with torch.no_grad():
            out = model(
                torch.tensor(Xp_va).float(),
                torch.tensor(Xv_va).float()
            )
            pred_vowel = out["vowels"].argmax(dim=1).cpu().numpy()
            pred_form_white = out["formants"].cpu().numpy()
            invL_pred_norm = out["vtl"].squeeze(-1).cpu().numpy()

        pred_form = preprocessor.inverse_transform_formants(pred_form_white)
        true_form = preprocessor.inverse_transform_formants(
            preprocessor.transform_formants(formants_va)
        )
        L_true = FormantPreprocessor.vtl_from_formants(formants_va)
        L_pred = preprocessor.inverse_transform_vtl(invL_pred_norm)

        group_stats = self._compute_group_stats(
            groups_va, L_true, L_pred, true_form, pred_form
        )

        metrics = {
            "vowel_acc": accuracy_score(yv_va, pred_vowel),
            "formant_r2": [r2_score(true_form[:, i], pred_form[:, i]) for i in range(4)],
            "vtl_r2": r2_score(L_true, L_pred),
            "vtl_mse": mean_squared_error(L_true, L_pred),
            "group_stats": group_stats,
        }
        return metrics

    @staticmethod
    def _compute_group_stats(groups_va, L_true, L_pred,
                             true_form, pred_form) -> list[dict]:
        stats = []
        for g in np.unique(groups_va):
            mask = groups_va == g
            stats.append({
                "group": g,
                "L_true_mean": float(L_true[mask].mean()),
                "L_pred_mean": float(L_pred[mask].mean()),
                "F_true_mean": true_form[mask].mean(axis=0).tolist(),
                "F_pred_mean": pred_form[mask].mean(axis=0).tolist(),  # bug fix
                "n": int(mask.sum()),
            })
        return stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MultiheadKFoldTrainer:
    """
    Orchestrates k-fold cross-validation for the three-head Whisper probe.

    Parameters
    ----------
    layers_phys  : encoder layers used for the physical (formant/VTL) heads
    layers_vowel : encoder layers used for the vowel classification head
    k            : number of folds
    epochs       : maximum training epochs per fold
    lr           : AdamW learning rate
    batch_size   : mini-batch size
    patience     : early-stopping patience (epochs)
    """

    def __init__(self, layers_phys: list[int] = None,
                 layers_vowel: list[int] = None,
                 k: int = 5, epochs: int = 200,
                 lr: float = 1e-3, batch_size: int = 64,
                 patience: int = 20):
        self.layers_phys = layers_phys or [0, 1, 2, 3, 4]
        self.layers_vowel = layers_vowel or [12]
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, embeddings, labels, formants, groups) -> list[dict]:
        # --- Preprocessing ---
        y_ints, mapping = labels_to_ints(labels)
        y_vowel = np.array(y_ints)
        formants = np.asarray(formants)

        preprocessor = FormantPreprocessor().fit(formants)
        F_white = preprocessor.transform_formants(formants)
        invL_norm = preprocessor.transform_vtl(formants)

        phys_builder = EmbeddingBuilder(self.layers_phys)
        vowel_builder = EmbeddingBuilder(self.layers_vowel)
        X_phys = phys_builder.build(embeddings)
        X_vowel = vowel_builder.build(embeddings)

        groups = np.array(groups)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_trainer = FoldTrainer(self.epochs, self.lr, self.batch_size, self.patience)
        evaluator = FoldEvaluator()
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_phys)):
            print(f"\n=== Fold {fold + 1}/{self.k} ===")

            train_ds, val_ds = self._make_datasets(
                X_phys, X_vowel, y_vowel, F_white, invL_norm,
                train_idx, val_idx
            )

            model = ThreeHeadPooled(
                d_phys=X_phys.shape[1],
                d_vowel=X_vowel.shape[1],
                num_classes=len(mapping),
                formant_dim=formants.shape[1]
            )
            model = fold_trainer.train(model, train_ds, val_ds)

            metrics = evaluator.evaluate(
                model,
                X_phys[val_idx], X_vowel[val_idx],
                y_vowel[val_idx], formants[val_idx],
                preprocessor, groups[val_idx]
            )

            self._print_fold_results(fold, metrics)
            fold_results.append({
                "vowel_acc": metrics["vowel_acc"],
                "formant_r2": metrics["vtl_r2"],   # kept for backward compat
                "group_stats": metrics["group_stats"],
            })

        return fold_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_datasets(X_phys, X_vowel, y_vowel, F_white, invL_norm,
                       train_idx, val_idx):
        def to_ds(idx):
            return TensorDataset(
                torch.tensor(X_phys[idx]).float(),
                torch.tensor(X_vowel[idx]).float(),
                torch.tensor(y_vowel[idx]).long(),
                torch.tensor(F_white[idx]).float(),
                torch.tensor(invL_norm[idx]).float(),
            )
        return to_ds(train_idx), to_ds(val_idx)

    @staticmethod
    def _print_fold_results(fold: int, metrics: dict):
        print("\nGroup-level means:")
        for s in metrics["group_stats"]:
            print(
                f"  {s['group']}: "
                f"L_true={s['L_true_mean']:.4f}, L_pred={s['L_pred_mean']:.4f}, "
                f"F_true={np.round(s['F_true_mean'], 1)}, "
                f"F_pred={np.round(s['F_pred_mean'], 1)}, "
                f"n={s['n']}"
            )
        print(f"Fold {fold + 1} vowel accuracy:  {metrics['vowel_acc']:.3f}")
        print(f"Fold {fold + 1} formant R²:      {metrics['formant_r2']}")
        print(f"Fold {fold + 1} VTL R²:          {metrics['vtl_r2']:.3f}, "
              f"MSE: {metrics['vtl_mse']:.6f}")