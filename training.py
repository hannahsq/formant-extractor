# training.py
"""
K-fold trainers for the Whisper formant probe.

Two concrete trainers are provided, one per model/preprocessing pipeline:

  ThreeHeadTrainer    — uses ThreeHeadPooled + FormantPreprocessor(mode="sample")
  PhysHeadTrainer     — uses TwoHeadPooledPhys + FormantPreprocessor(mode="blended")

Both inherit from MultiheadKFoldTrainer which provides:
  - fit()             : run k-fold cross-validation
  - fit_averaged()    : run k-fold, average fold weights, fine-tune on full data

Usage
-----
    trainer = ThreeHeadTrainer(k=5, epochs=300)
    fold_results = trainer.fit(embeddings, labels, formants, groups)

    # or, to get a production model:
    model = trainer.fit_averaged(embeddings, labels, formants, groups)
    torch.save(model.state_dict(), "model.pt")
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from utils import labels_to_ints, build_concat_embeddings
from preprocessing import FormantPreprocessor
from heads import ThreeHeadPooled, TwoHeadPooledPhys


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _to_tensor_dataset(*arrays, dtypes) -> TensorDataset:
    tensors = [torch.tensor(a, dtype=dt) for a, dt in zip(arrays, dtypes)]
    return TensorDataset(*tensors)


def _gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Gaussian negative log-likelihood loss.

        loss = mean( log(σ) + (y - μ)² / (2σ²) )

    sigma is a fixed, precomputed quantity (whitened measurement noise from
    the dataset) — it is not learned by the model. Samples with high σ
    contribute less to the squared-error term, reducing the penalty for
    inherently noisy measurements.
    """
    return torch.mean(
        torch.log(sigma) + (target - pred) ** 2 / (2.0 * sigma ** 2)
    )


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------

class MultiheadKFoldTrainer:
    """
    Abstract base class for k-fold multihead trainers.

    Subclasses must implement:
      _build_model()      : instantiate the model for one fold
      _make_datasets()    : build train/val TensorDatasets for one fold
      _compute_loss()     : compute scalar loss from model output and targets
      _evaluate_fold()    : run inference and return a metrics dict

    Parameters
    ----------
    k           : number of folds
    epochs      : maximum epochs per fold
    lr          : AdamW learning rate
    batch_size  : mini-batch size
    patience    : early-stopping patience in epochs
    finetune_epochs : epochs for full-data fine-tuning in fit_averaged()
    """

    def __init__(
        self,
        k: int = 5,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 20,
        finetune_epochs: int = 50,
    ):
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.finetune_epochs = finetune_epochs

        # set by _prepare_data(); available to subclasses during fit
        self._preprocessor: FormantPreprocessor | None = None
        self._mapping: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings,
        labels: list,
        formants: np.ndarray,
        groups: np.ndarray,
        formants_sigma: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Run k-fold cross-validation.

        Parameters
        ----------
        formants_sigma : (N, 4) per-sample formant standard deviations in Hz.
                         When provided, the Gaussian NLL loss is used for
                         formant prediction so that noisy samples are
                         penalised less. Falls back to MSE if None.

        Returns
        -------
        list of per-fold result dicts (keys vary by subclass).
        """
        data = self._prepare_data(embeddings, labels, formants, groups, formants_sigma)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data["X_phys"])):
            print(f"\n=== Fold {fold + 1}/{self.k} ===")
            model = self._train_fold(data, train_idx, val_idx)
            metrics = self._evaluate_fold(model, data, val_idx, groups)
            self._print_fold(fold, metrics)
            fold_results.append(metrics)

        return fold_results

    def fit_best(
        self,
        embeddings,
        labels: list,
        formants: np.ndarray,
        groups: np.ndarray,
        formants_sigma: np.ndarray | None = None,
    ) -> tuple[list[dict], nn.Module]:
        """
        Run k-fold cross-validation and return both the per-fold metrics and
        the single best fold model evaluated against the full dataset.

        "Best" is defined as the fold whose model achieves the highest vowel
        accuracy when run on all N samples (not just the validation split).
        This gives a fair comparison across folds since each fold's val split
        is a different subset.

        Returns
        -------
        fold_results : list of per-fold metric dicts (same as fit())
        best_model   : the nn.Module from the best fold, in eval() mode
        """
        data = self._prepare_data(embeddings, labels, formants, groups, formants_sigma)
        kf   = KFold(n_splits=self.k, shuffle=True, random_state=42)

        fold_results = []
        fold_models  = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data["X_phys"])):
            print(f"\n=== Fold {fold + 1}/{self.k} ===")
            model   = self._train_fold(data, train_idx, val_idx)
            metrics = self._evaluate_fold(model, data, val_idx, groups)
            self._print_fold(fold, metrics)
            fold_results.append(metrics)
            fold_models.append(model)

        # Evaluate each fold model on the full dataset to pick the best one
        print("\nEvaluating all fold models on full dataset to select best...")
        all_idx    = np.arange(len(data["y_vowel"]))
        best_acc   = -1.0
        best_model = fold_models[0]
        for fold, model in enumerate(fold_models):
            metrics_full = self._evaluate_fold(model, data, all_idx, groups)
            acc = metrics_full["vowel_acc"]
            print(f"  Fold {fold + 1}: full-set vowel acc = {acc:.4f}")
            if acc > best_acc:
                best_acc   = acc
                best_model = model

        print(f"\nSelected fold model with full-set vowel acc = {best_acc:.4f}")
        best_model.eval()
        return fold_results, best_model

    def fit_averaged(
        self,
        embeddings,
        labels: list,
        formants: np.ndarray,
        groups: np.ndarray,
        formants_sigma: np.ndarray | None = None,
    ) -> nn.Module:
        """
        Run k-fold, average the weights across all fold models, then
        fine-tune the averaged model on the full dataset.

        Parameters
        ----------
        formants_sigma : see fit() docstring

        Returns
        -------
        Fine-tuned nn.Module ready for inference / saving.
        """
        data = self._prepare_data(embeddings, labels, formants, groups, formants_sigma)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data["X_phys"])):
            print(f"\n=== Fold {fold + 1}/{self.k} (averaging run) ===")
            model = self._train_fold(data, train_idx, val_idx)
            fold_models.append(model)

        print("\n=== Averaging weights across folds ===")
        avg_model = self._average_weights(fold_models)

        print(f"\n=== Fine-tuning on full dataset ({self.finetune_epochs} epochs) ===")
        full_idx = np.arange(len(data["X_phys"]))
        full_ds = self._make_datasets(data, full_idx, full_idx)[0]  # train only
        avg_model = self._run_training(avg_model, full_ds, val_ds=None,
                                       epochs=self.finetune_epochs)
        return avg_model

    def fit_and_compare(
        self,
        embeddings,
        labels: list,
        formants: np.ndarray,
        groups: np.ndarray,
        formants_sigma: np.ndarray | None = None,
    ) -> dict:
        """
        Train fold models, average their weights, fine-tune on the full
        dataset, then evaluate both the fold models and the averaged model
        on the exact same validation splits.

        Parameters
        ----------
        formants_sigma : see fit() docstring

        Returns
        -------
        dict with keys:
          "fold_results"  : list of per-fold metrics for each fold model
          "avg_results"   : list of per-fold metrics for the averaged model
                            evaluated on the same val sets
          "avg_model"     : the fine-tuned averaged nn.Module
          "comparison"    : list of per-fold dicts summarising the delta
                            between averaged and fold model on key metrics
        """
        data   = self._prepare_data(embeddings, labels, formants, groups, formants_sigma)
        kf     = KFold(n_splits=self.k, shuffle=True, random_state=42)
        splits = list(kf.split(data["X_phys"]))

        # --- k-fold pass: train and evaluate each fold model ---
        fold_models  = []
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n=== Fold {fold + 1}/{self.k} ===")
            model   = self._train_fold(data, train_idx, val_idx)
            metrics = self._evaluate_fold(model, data, val_idx, groups)
            self._print_fold(fold, metrics)
            fold_models.append(model)
            fold_results.append(metrics)

        # --- average + fine-tune ---
        print("\n=== Averaging weights across folds ===")
        avg_model = self._average_weights(fold_models)

        print(f"\n=== Fine-tuning on full dataset ({self.finetune_epochs} epochs) ===")
        full_idx = np.arange(len(data["X_phys"]))
        full_ds  = self._make_datasets(data, full_idx, full_idx)[0]
        avg_model = self._run_training(avg_model, full_ds, val_ds=None,
                                       epochs=self.finetune_epochs)

        # --- evaluate averaged model on the same val splits ---
        print("\n=== Evaluating averaged model on fold val sets ===")
        avg_results = []
        for fold, (_, val_idx) in enumerate(splits):
            metrics = self._evaluate_fold(avg_model, data, val_idx, groups)
            print(f"\n--- Averaged model / Fold {fold + 1} val set ---")
            self._print_fold(fold, metrics)
            avg_results.append(metrics)

        # --- comparison summary ---
        comparison = self._compare_results(fold_results, avg_results)
        self._print_comparison(comparison)

        return {
            "fold_results": fold_results,
            "avg_results":  avg_results,
            "avg_model":    avg_model,
            "comparison":   comparison,
        }

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _prepare_data(self, embeddings, labels, formants, groups,
                      formants_sigma=None) -> dict:
        """Preprocess inputs and return a data dict shared across folds."""
        raise NotImplementedError

    def _build_model(self, data: dict) -> nn.Module:
        """Instantiate a fresh model using shapes stored in `data`."""
        raise NotImplementedError

    def _make_datasets(
        self, data: dict, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> tuple[TensorDataset, TensorDataset]:
        """Return (train_ds, val_ds) for this fold."""
        raise NotImplementedError

    def _compute_loss(
        self, out: dict, batch: tuple, ce: nn.Module, mse: nn.Module
    ) -> torch.Tensor:
        """Compute scalar loss from model output dict and batch targets."""
        raise NotImplementedError

    def _evaluate_fold(
        self, model: nn.Module, data: dict,
        val_idx: np.ndarray, groups: np.ndarray,
    ) -> dict:
        """Run inference on val set and return a metrics dict."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared training machinery
    # ------------------------------------------------------------------

    def _train_fold(
        self, data: dict, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> nn.Module:
        train_ds, val_ds = self._make_datasets(data, train_idx, val_idx)
        model = self._build_model(data)
        return self._run_training(model, train_ds, val_ds)

    def _run_training(
        self,
        model: nn.Module,
        train_ds: TensorDataset,
        val_ds: TensorDataset | None,
        epochs: int | None = None,
    ) -> nn.Module:
        epochs = epochs or self.epochs
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size) if val_ds else None

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        ce  = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        best_state    = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        patience_left = self.patience

        for _ in range(epochs):
            model.train()
            for batch in train_dl:
                opt.zero_grad()
                out  = model(batch[0], batch[1])
                loss = self._compute_loss(out, batch, ce, mse)
                loss.backward()
                opt.step()

            if val_dl is None:
                continue

            val_loss = self._validate(model, val_dl, ce, mse)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(model.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)
        return model

    def _validate(
        self, model: nn.Module, val_dl: DataLoader,
        ce: nn.Module, mse: nn.Module,
    ) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_dl:
                out = model(batch[0], batch[1])
                total += self._compute_loss(out, batch, ce, mse).item()
        return total / len(val_dl)

    @staticmethod
    def _average_weights(models: list[nn.Module]) -> nn.Module:
        """
        Return a new model whose parameters are the element-wise mean
        of the supplied fold models.
        """
        avg = copy.deepcopy(models[0])
        avg_sd = avg.state_dict()
        for key in avg_sd:
            avg_sd[key] = torch.stack(
                [m.state_dict()[key].float() for m in models], dim=0
            ).mean(dim=0)
        avg.load_state_dict(avg_sd)
        return avg

    @staticmethod
    def _group_stats(
        groups_va: np.ndarray,
        y_vowel_true: np.ndarray, y_vowel_pred: np.ndarray,
        L_true: np.ndarray, L_pred: np.ndarray,
        F_true: np.ndarray, F_pred: np.ndarray,
    ) -> list[dict]:
        stats = []
        for g in np.unique(groups_va):
            mask = groups_va == g
            stats.append({
                "group":        g,
                "vowel_acc":    float(accuracy_score(y_vowel_true[mask], y_vowel_pred[mask])),
                "L_true_mean":  float(L_true[mask].mean()),
                "L_pred_mean":  float(L_pred[mask].mean()),
                "F_true_mean":  F_true[mask].mean(axis=0).tolist(),
                "F_pred_mean":  F_pred[mask].mean(axis=0).tolist(),
                "n":            int(mask.sum()),
            })
        return stats

    @staticmethod
    def _print_fold(fold: int, metrics: dict):
        n = fold + 1
        print(f"\n--- Fold {n} results ---")
        print(f"  Vowel accuracy : {metrics['vowel_acc']:.3f}")
        print(f"  VTL  R² / MSE  : {metrics['vtl_r2']:.3f} / {metrics['vtl_mse']:.4f}")

        r2s  = metrics["formant_r2"]
        mses = metrics.get("formant_mse")
        header = f"  {'':6}  {'R²':>7}"
        if mses:
            header += f"  {'MSE':>10}"
        print(header)
        for i, label in enumerate(["F1", "F2", "F3", "F4"]):
            row = f"  {label:6}  {r2s[i]:7.3f}"
            if mses:
                row += f"  {mses[i]:10.2f}"
            print(row)

        if "group_stats" in metrics:
            print(f"\n  {'Group':>8}  {'Acc':>6}  {'L_true':>8}  {'L_pred':>8}  {'n':>5}")
            print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*5}")
            for s in metrics["group_stats"]:
                print(
                    f"  {s['group']:>8}  {s['vowel_acc']:6.3f}"
                    f"  {s['L_true_mean']:8.4f}  {s['L_pred_mean']:8.4f}"
                    f"  {s['n']:5d}"
                )

    @staticmethod
    def _compare_results(
        fold_results: list[dict], avg_results: list[dict]
    ) -> list[dict]:
        """
        Compute per-fold deltas between the averaged model and the fold model.
        Positive delta means the averaged model outperformed the fold model.
        """
        comparison = []
        for fold, (fr, ar) in enumerate(zip(fold_results, avg_results)):
            fold_f_r2 = np.mean(fr["formant_r2"])
            avg_f_r2  = np.mean(ar["formant_r2"])
            comparison.append({
                "fold":               fold + 1,
                "vowel_acc_fold":     fr["vowel_acc"],
                "vowel_acc_avg":      ar["vowel_acc"],
                "vowel_acc_delta":    ar["vowel_acc"] - fr["vowel_acc"],
                "vtl_r2_fold":        fr["vtl_r2"],
                "vtl_r2_avg":         ar["vtl_r2"],
                "vtl_r2_delta":       ar["vtl_r2"] - fr["vtl_r2"],
                "formant_r2_fold":    fold_f_r2,
                "formant_r2_avg":     avg_f_r2,
                "formant_r2_delta":   avg_f_r2 - fold_f_r2,
            })
        return comparison

    @staticmethod
    def _print_comparison(comparison: list[dict]):
        print("\n=== Fold model vs. Averaged model ===")
        print(
            f"  {'Fold':>5}  "
            f"{'VowAcc(f)':>10}  {'VowAcc(a)':>10}  {'Δ':>7}  "
            f"{'VTL R²(f)':>10}  {'VTL R²(a)':>10}  {'Δ':>7}  "
            f"{'FmtR²(f)':>9}  {'FmtR²(a)':>9}  {'Δ':>7}"
        )
        print(f"  {'-'*5}  " + (f"{'-'*10}  {'-'*10}  {'-'*7}  " * 3).rstrip())
        for c in comparison:
            print(
                f"  {c['fold']:>5}  "
                f"{c['vowel_acc_fold']:10.3f}  {c['vowel_acc_avg']:10.3f}  "
                f"{c['vowel_acc_delta']:+7.3f}  "
                f"{c['vtl_r2_fold']:10.3f}  {c['vtl_r2_avg']:10.3f}  "
                f"{c['vtl_r2_delta']:+7.3f}  "
                f"{c['formant_r2_fold']:9.3f}  {c['formant_r2_avg']:9.3f}  "
                f"{c['formant_r2_delta']:+7.3f}"
            )
        # Summary row: mean across folds
        print(f"  {'-'*5}  " + (f"{'-'*10}  {'-'*10}  {'-'*7}  " * 3).rstrip())
        print(
            f"  {'mean':>5}  "
            f"{np.mean([c['vowel_acc_fold']  for c in comparison]):10.3f}  "
            f"{np.mean([c['vowel_acc_avg']   for c in comparison]):10.3f}  "
            f"{np.mean([c['vowel_acc_delta'] for c in comparison]):+7.3f}  "
            f"{np.mean([c['vtl_r2_fold']     for c in comparison]):10.3f}  "
            f"{np.mean([c['vtl_r2_avg']      for c in comparison]):10.3f}  "
            f"{np.mean([c['vtl_r2_delta']    for c in comparison]):+7.3f}  "
            f"{np.mean([c['formant_r2_fold'] for c in comparison]):9.3f}  "
            f"{np.mean([c['formant_r2_avg']  for c in comparison]):9.3f}  "
            f"{np.mean([c['formant_r2_delta']for c in comparison]):+7.3f}"
        )


# ---------------------------------------------------------------------------
# ThreeHeadTrainer  (sample-mode VTL, ThreeHeadPooled)
# ---------------------------------------------------------------------------

class ThreeHeadTrainer(MultiheadKFoldTrainer):
    """
    Trainer for ThreeHeadPooled using FormantPreprocessor(mode="sample").

    Layer config
    ------------
    layers_phys  : encoder layers concatenated for the formant + VTL heads
    layers_vowel : encoder layers concatenated for the vowel head
    """

    def __init__(
        self,
        layers_phys:  list[int] = None,
        layers_vowel: list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers_phys  = layers_phys  or [0, 1, 2, 3, 4]
        self.layers_vowel = layers_vowel or [-1]

    def _prepare_data(self, embeddings, labels, formants, groups,
                      formants_sigma=None) -> dict:
        y_ints, mapping = labels_to_ints(labels)
        self._mapping = mapping
        formants = np.asarray(formants)

        preprocessor = FormantPreprocessor(mode="sample").fit(formants)
        self._preprocessor = preprocessor

        F_white   = preprocessor.transform_formants(formants)
        invL_norm = preprocessor.transform_vtl(formants)

        # Whiten sigma in the same space as the targets; fall back to
        # unit sigma (equivalent to plain MSE) when not provided.
        if formants_sigma is not None:
            F_sigma_white = preprocessor.transform_formant_sigma(
                np.asarray(formants_sigma)
            )
        else:
            F_sigma_white = np.ones_like(F_white)

        return {
            "X_phys":        build_concat_embeddings(embeddings, self.layers_phys),
            "X_vowel":       build_concat_embeddings(embeddings, self.layers_vowel),
            "y_vowel":       np.array(y_ints),
            "F_white":       F_white,
            "F_sigma_white": F_sigma_white,
            "invL_norm":     invL_norm,
            "formants":      formants,
            "groups":        np.array(groups),
            "num_classes":   len(mapping),
            "formant_dim":   formants.shape[1],
        }

    def _build_model(self, data: dict) -> nn.Module:
        return ThreeHeadPooled(
            d_phys      = data["X_phys"].shape[1],
            d_vowel     = data["X_vowel"].shape[1],
            num_classes = data["num_classes"],
            formant_dim = data["formant_dim"],
        )

    def _make_datasets(self, data, train_idx, val_idx):
        def ds(idx):
            return _to_tensor_dataset(
                data["X_phys"][idx], data["X_vowel"][idx],
                data["y_vowel"][idx], data["F_white"][idx],
                data["F_sigma_white"][idx], data["invL_norm"][idx],
                dtypes=[torch.float32, torch.float32, torch.long,
                        torch.float32, torch.float32, torch.float32],
            )
        return ds(train_idx), ds(val_idx)

    def _compute_loss(self, out, batch, ce, mse):
        _, _, yv, yf, sf, yL = batch
        return (
            ce(out["vowels"], yv)
            + _gaussian_nll(out["formants"], yf, sf)
            + 0.5 * mse(out["vtl"], yL)
        )

    def _evaluate_fold(self, model, data, val_idx, groups) -> dict:
        pre = self._preprocessor
        Xp  = torch.tensor(data["X_phys"][val_idx]).float()
        Xv  = torch.tensor(data["X_vowel"][val_idx]).float()

        model.eval()
        with torch.no_grad():
            out = model(Xp, Xv)
            pred_vowel      = out["vowels"].argmax(dim=1).cpu().numpy()
            pred_form_white = out["formants"].cpu().numpy()
            invL_pred_norm  = out["vtl"].cpu().numpy()

        formants_va = data["formants"][val_idx]

        # L_true: invert the stored normalised targets so it matches exactly
        # what the model was trained to predict — the preprocessor's L_star
        # (sample mode: median VTL over F1-F3), not a freshly computed estimate.
        L_true = pre.inverse_transform_vtl(data["invL_norm"][val_idx])
        L_pred = pre.inverse_transform_vtl(invL_pred_norm)

        # In sample mode inverse_transform_formants returns Hz directly.
        # true_form uses the raw formants (the sample-mode whitening is
        # invertible and round-trips losslessly, so this is equivalent).
        pred_form = pre.inverse_transform_formants(pred_form_white)
        true_form = formants_va

        groups_va  = np.array(groups)[val_idx]
        y_vowel_va = data["y_vowel"][val_idx]
        return {
            "vowel_acc":   accuracy_score(y_vowel_va, pred_vowel),
            "formant_r2":  [r2_score(true_form[:, i], pred_form[:, i]) for i in range(4)],
            "formant_mse": [mean_squared_error(true_form[:, i], pred_form[:, i]) for i in range(4)],
            "vtl_r2":      r2_score(L_true, L_pred),
            "vtl_mse":     mean_squared_error(L_true, L_pred),
            "group_stats": self._group_stats(groups_va, y_vowel_va, pred_vowel,
                                             L_true, L_pred, true_form, pred_form),
        }


# ---------------------------------------------------------------------------
# PhysHeadTrainer  (blended-mode VTL, TwoHeadPooledPhys)
# ---------------------------------------------------------------------------

class PhysHeadTrainer(MultiheadKFoldTrainer):
    """
    Trainer for TwoHeadPooledPhys using FormantPreprocessor(mode="blended").

    The physical head predicts a 5-vector [invL_norm, F1_white, …, F4_white]
    where formants are normalised relative to the speaker-blended ideal tube.
    The vowel head conditions on the predicted normalised formants.

    Layer config
    ------------
    layers_phys  : encoder layers for the physical head (default: 0–4)
    layers_vowel : encoder layers for the vowel head   (default: [12])
    alpha        : sample/speaker VTL blend weight (passed to preprocessor)
    vtl_loss_weight : weight on invL term in physical loss
    """

    def __init__(
        self,
        layers_phys:      list[int] = None,
        layers_vowel:     list[int] = None,
        alpha:            float = 0.4,
        vtl_loss_weight:  float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers_phys     = layers_phys  or [0, 1, 2, 3, 4]
        self.layers_vowel    = layers_vowel or [-1]
        self.alpha           = alpha
        self.vtl_loss_weight = vtl_loss_weight

    def _prepare_data(self, embeddings, labels, formants, groups,
                      formants_sigma=None) -> dict:
        y_ints, mapping = labels_to_ints(labels)
        self._mapping = mapping
        formants = np.asarray(formants)
        groups   = np.asarray(groups)

        preprocessor = FormantPreprocessor(
            mode="blended", alpha=self.alpha
        ).fit(formants, groups)
        self._preprocessor = preprocessor

        F_white   = preprocessor.transform_formants(formants, groups)
        invL_norm = preprocessor.transform_vtl(formants, groups)
        y_phys    = np.concatenate([invL_norm[:, None], F_white], axis=1)

        # Whiten sigma in the same dimensionless space as the μ targets.
        # Prepend a unit column for the invL slot (VTL still uses MSE).
        if formants_sigma is not None:
            F_sigma_white = preprocessor.transform_formant_sigma(
                np.asarray(formants_sigma), groups=groups, formants_hz=formants
            )
        else:
            F_sigma_white = np.ones_like(F_white)

        sigma_phys = np.concatenate(
            [np.ones((len(formants), 1)), F_sigma_white], axis=1
        )  # (N, 5) — first column is placeholder for invL (MSE, not NLL)

        return {
            "X_phys":      build_concat_embeddings(embeddings, self.layers_phys),
            "X_vowel":     build_concat_embeddings(embeddings, self.layers_vowel),
            "y_vowel":     np.array(y_ints),
            "y_phys":      y_phys,
            "sigma_phys":  sigma_phys,
            "formants":    formants,
            "groups":      groups,
            "num_classes": len(mapping),
        }

    def _build_model(self, data: dict) -> nn.Module:
        return TwoHeadPooledPhys(
            d_phys      = data["X_phys"].shape[1],
            d_vowel     = data["X_vowel"].shape[1],
            num_classes = data["num_classes"],
        )

    def _make_datasets(self, data, train_idx, val_idx):
        def ds(idx):
            return _to_tensor_dataset(
                data["X_phys"][idx], data["X_vowel"][idx],
                data["y_vowel"][idx], data["y_phys"][idx],
                data["sigma_phys"][idx],
                dtypes=[torch.float32, torch.float32,
                        torch.long, torch.float32, torch.float32],
            )
        return ds(train_idx), ds(val_idx)

    def _compute_loss(self, out, batch, ce, mse):
        _, _, yv, yp, sp = batch
        pred_invL,   true_invL   = out["phys"][:, 0], yp[:, 0]
        pred_Fwhite, true_Fwhite = out["phys"][:, 1:], yp[:, 1:]
        sigma_F = sp[:, 1:]   # formant σ only; invL still uses weighted MSE
        loss_phys = (
            self.vtl_loss_weight * mse(pred_invL, true_invL)
            + _gaussian_nll(pred_Fwhite, true_Fwhite, sigma_F)
        )
        return ce(out["vowels"], yv) + loss_phys

    def _evaluate_fold(self, model, data, val_idx, groups) -> dict:
        pre         = self._preprocessor
        formants_va = data["formants"][val_idx]
        groups_va   = np.asarray(groups)[val_idx]
        y_phys_va   = data["y_phys"][val_idx]

        Xp = torch.tensor(data["X_phys"][val_idx]).float()
        Xv = torch.tensor(data["X_vowel"][val_idx]).float()

        model.eval()
        with torch.no_grad():
            out        = model(Xp, Xv)
            pred_vowel = out["vowels"].argmax(dim=1).cpu().numpy()
            pred_phys  = out["phys"].cpu().numpy()

        # Recover predicted VTL and formants in Hz
        invL_norm_pred = pred_phys[:, 0]
        L_pred  = pre.inverse_transform_vtl(invL_norm_pred)
        F_pred  = pre.recover_formants_hz(pred_phys[:, 1:], L_pred)

        # L_true: invert the stored normalised targets to get the blended
        # L_star the model was trained against — consistent with L_pred.
        L_true  = pre.inverse_transform_vtl(y_phys_va[:, 0])

        # F_true: recover Hz using the true L_star (same inversion path as pred)
        F_true  = pre.recover_formants_hz(y_phys_va[:, 1:], L_true)

        y_vowel_va = data["y_vowel"][val_idx]
        return {
            "vowel_acc":   accuracy_score(y_vowel_va, pred_vowel),
            "formant_r2":  [r2_score(F_true[:, i], F_pred[:, i]) for i in range(4)],
            "formant_mse": [mean_squared_error(F_true[:, i], F_pred[:, i])
                            for i in range(4)],
            "vtl_r2":      r2_score(L_true, L_pred),
            "vtl_mse":     mean_squared_error(L_true, L_pred),
            "group_stats": self._group_stats(groups_va, y_vowel_va, pred_vowel,
                                             L_true, L_pred, F_true, F_pred),
        }
