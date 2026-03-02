# preprocessing.py
"""
Formant preprocessing and VTL target construction.

FormantPreprocessor supports two VTL estimation modes:

  - "sample"  : per-sample median across F1–F3 (fast, no group info needed)
  - "blended" : alpha * sample_VTL + (1-alpha) * speaker_VTL
                Speaker VTL is estimated as the per-group median.
                Requires `groups` to be passed to fit().

The blended mode corresponds to the original targets.py pipeline and
produces speaker-normalised formant targets (F / F_ideal).
The sample mode corresponds to the simpler pipeline used in the original
train_head.py and is the default for backwards compatibility.

Uncertainty (σ) propagation
----------------------------
When per-sample formant standard deviations are available (e.g. from
Hillenbrand's 10ms frame measurements), they can be passed to fit() and
then whitened via transform_formant_sigma(). The whitened σ values are in
the same space as the whitened μ targets and should be passed to the
Gaussian NLL loss during training so that noisy samples are penalised less.

The whitening transform for σ is W (no mean subtraction — σ is already
a scale quantity centred at zero). In sample mode this is just:

    σ_white = σ @ inv_sqrt_cov

In blended mode σ is first divided by F_ideal to put it in the same
dimensionless ratio space as the μ targets, then whitened:

    σ_white = (σ / F_ideal) @ inv_sqrt_cov

Typical usage
-------------
    pre = FormantPreprocessor(mode="sample").fit(formants_mu, formants_sigma)

    # during training
    F_white     = pre.transform_formants(formants_mu)
    F_sigma_w   = pre.transform_formant_sigma(formants_sigma)
    invL_norm   = pre.transform_vtl(formants_mu)

    # during evaluation
    F_hz_pred   = pre.inverse_transform_formants(F_white_pred)
    L_pred      = pre.inverse_transform_vtl(invL_norm_pred)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C    = 350.0                               # speed of sound (m/s)
_HARM = np.array([1, 3, 5, 7], dtype=float) # odd harmonics for quarter-wave tube


# ---------------------------------------------------------------------------
# Low-level helpers (stateless, reusable)
# ---------------------------------------------------------------------------

def vtl_per_formant(formants_hz: np.ndarray) -> np.ndarray:
    """
    Estimate anatomical VTL from each formant independently.

    Uses the quarter-wave resonator formula with an end correction factor
    to convert acoustic tube length to anatomical length.

    Parameters
    ----------
    formants_hz : (N, 4) array of F1-F4 in Hz

    Returns
    -------
    L_if : (N, 4) anatomical VTL estimates in metres
    """
    return _C * _HARM / (4.0 * formants_hz)


def vtl_from_formants(
    formants_hz: np.ndarray,
    use_formants: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """
    Per-sample VTL estimate: median over selected formants.

    Returns
    -------
    L : (N,) array in cm
    """
    L_if = vtl_per_formant(formants_hz)
    return np.median(L_if[:, list(use_formants)], axis=1)


def speaker_vtl(
    groups: np.ndarray,
    formants_hz: np.ndarray,
    use_formants: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """
    Estimate a single VTL per speaker group as the median of per-sample VTLs.

    Returns
    -------
    dict mapping group label → scalar VTL (cm)
    """
    L_sample = vtl_from_formants(formants_hz, use_formants)
    per_group: dict = defaultdict(list)
    for g, L in zip(groups, L_sample):
        per_group[g].append(L)
    return {g: float(np.median(vals)) for g, vals in per_group.items()}


# ---------------------------------------------------------------------------
# FormantPreprocessor
# ---------------------------------------------------------------------------

class FormantPreprocessor:
    """
    Fit/transform pipeline for formant whitening and VTL normalisation.

    Parameters
    ----------
    mode : {"sample", "blended"}
        VTL estimation strategy (see module docstring).
    alpha : float
        Blend weight for sample VTL when mode="blended". Ignored otherwise.
    use_formants : tuple[int]
        Which formant indices to use for VTL estimation.
    """

    def __init__(
        self,
        mode: str = "sample",
        alpha: float = 0.4,
        use_formants: tuple[int, ...] = (0, 1, 2),
    ):
        if mode not in ("sample", "blended"):
            raise ValueError(f"mode must be 'sample' or 'blended', got '{mode}'")
        self.mode = mode
        self.alpha = alpha
        self.use_formants = use_formants

        # set by fit()
        self._F_mean: np.ndarray | None = None
        self._inv_sqrt_cov: np.ndarray | None = None
        self._invL_mean: float | None = None
        self._invL_std: float | None = None
        self._speaker_L: dict | None = None  # blended mode only

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        formants_hz: np.ndarray,
        groups: np.ndarray | None = None,
        formants_sigma: np.ndarray | None = None,
    ) -> "FormantPreprocessor":
        """
        Compute whitening statistics and VTL normalisation constants.

        Parameters
        ----------
        formants_hz    : (N, 4) per-sample formant means (μ) in Hz
        groups         : (N,) speaker group labels — required for mode="blended"
        formants_sigma : (N, 4) per-sample formant std devs (σ) in Hz — optional;
                         only used to validate that transform_formant_sigma will
                         work correctly; no statistics are derived from σ at fit time
        """
        formants_hz = np.asarray(formants_hz, dtype=float)

        if self.mode == "blended" and groups is None:
            raise ValueError("groups must be provided when mode='blended'")

        # --- VTL targets ---
        L_star = self._compute_L_star(formants_hz, groups)
        invL = 1.0 / L_star
        self._invL_mean = float(invL.mean())
        self._invL_std  = float(invL.std())

        # --- Formant whitening ---
        # In blended mode we whiten F/F_ideal (speaker-normalised).
        # In sample mode we whiten raw formants directly.
        F_for_whitening = self._normalise_formants(formants_hz, L_star)
        self._F_mean = F_for_whitening.mean(axis=0)
        F_centered = F_for_whitening - self._F_mean
        cov = np.cov(F_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        self._inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return self

    # ------------------------------------------------------------------
    # Transform / inverse transform  (μ)
    # ------------------------------------------------------------------

    def transform_formants(
        self,
        formants_hz: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Whiten formant means. Returns (N, 4) array."""
        self._check_fitted()
        formants_hz = np.asarray(formants_hz, dtype=float)
        L_star = self._compute_L_star(formants_hz, groups)
        F = self._normalise_formants(formants_hz, L_star)
        return (F - self._F_mean) @ self._inv_sqrt_cov

    def inverse_transform_formants(self, F_white: np.ndarray) -> np.ndarray:
        """
        Invert whitening. In sample mode returns Hz directly.
        In blended mode returns the speaker-normalised ratio F/F_ideal —
        use recover_formants_hz() to get absolute Hz.
        """
        self._check_fitted()
        F_centered = F_white @ np.linalg.inv(self._inv_sqrt_cov)
        return F_centered + self._F_mean

    def recover_formants_hz(
        self,
        F_white: np.ndarray,
        L_star: np.ndarray,
    ) -> np.ndarray:
        """
        Full inversion for blended mode: whitened → Hz.

        Parameters
        ----------
        F_white : (N, 4) whitened formants
        L_star  : (N,) VTL estimates used during transform
        """
        F_norm = self.inverse_transform_formants(F_white)
        if self.mode == "blended":
            F_ideal = (_HARM[None, :] * _C) / (4.0 * L_star[:, None])
            return F_norm * F_ideal
        return F_norm  # sample mode: already in Hz

    # ------------------------------------------------------------------
    # Transform  (σ)
    # ------------------------------------------------------------------

    def transform_formant_sigma(
        self,
        formants_sigma: np.ndarray,
        groups: np.ndarray | None = None,
        formants_hz: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Scale per-sample formant standard deviations into the whitened target
        space so they can be used directly in the Gaussian NLL loss.

        σ is a per-sample *scale* quantity, not a vector in formant space, so
        it must not be rotated by the full whitening matrix W. Instead we apply
        only the marginal scaling: divide each formant's σ by the square root
        of its marginal variance, which is the reciprocal of the corresponding
        diagonal entry of W (i.e. the per-formant std in the pre-whitened space).

        In blended mode σ is first divided by F_ideal to convert from Hz into
        the same dimensionless ratio space as the μ targets before scaling.

        The result is clamped to a small positive floor (1e-3) so that
        log(σ) in the NLL loss never produces NaN.

        Parameters
        ----------
        formants_sigma : (N, 4) per-sample σ in Hz
        groups         : (N,) required for blended mode
        formants_hz    : (N, 4) μ in Hz — required for blended mode

        Returns
        -------
        (N, 4) scaled σ values — pass directly to _gaussian_nll()
        """
        self._check_fitted()
        sigma = np.asarray(formants_sigma, dtype=float).copy()

        if self.mode == "blended":
            if formants_hz is None or groups is None:
                raise ValueError(
                    "formants_hz and groups are required for sigma transform "
                    "in blended mode"
                )
            L_star  = self._compute_L_star(
                np.asarray(formants_hz, dtype=float), groups
            )
            F_ideal = (_HARM[None, :] * _C) / (4.0 * L_star[:, None])
            sigma   = sigma / F_ideal   # dimensionless ratio

        # Marginal scaling only: divide by per-formant std in pre-whitened space.
        # The diagonal of inv_sqrt_cov gives 1/std_i for each formant dimension.
        marginal_scale = np.diag(self._inv_sqrt_cov)   # (4,)
        sigma_scaled   = sigma * marginal_scale[None, :]

        # Clamp to avoid log(0) or log(negative) in the NLL loss
        return np.maximum(sigma_scaled, 1e-3)

    # ------------------------------------------------------------------
    # VTL transforms
    # ------------------------------------------------------------------

    def transform_vtl(
        self,
        formants_hz: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalised inverse-VTL targets. Returns (N,) array."""
        self._check_fitted()
        L_star = self._compute_L_star(formants_hz, groups)
        invL = 1.0 / L_star
        return (invL - self._invL_mean) / self._invL_std

    def inverse_transform_vtl(self, invL_norm: np.ndarray) -> np.ndarray:
        """Normalised inverse-VTL → VTL in cm. Returns (N,) array."""
        self._check_fitted()
        invL = invL_norm * self._invL_std + self._invL_mean
        return 1.0 / invL

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def vtl_from_formants(self, formants_hz: np.ndarray) -> np.ndarray:
        """Per-sample VTL in cm (no normalisation). Useful for evaluation."""
        return vtl_from_formants(formants_hz, self.use_formants)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_L_star(
        self,
        formants_hz: np.ndarray,
        groups: np.ndarray | None,
    ) -> np.ndarray:
        """Return per-sample VTL under the configured mode."""
        L_sample = vtl_from_formants(formants_hz, self.use_formants)

        if self.mode == "sample":
            return L_sample

        # blended mode
        if groups is None:
            raise ValueError("groups must be provided for mode='blended'")

        # Use stored speaker VTL if available (i.e. transforming unseen data)
        spk_L = self._speaker_L if self._speaker_L is not None else speaker_vtl(
            groups, formants_hz, self.use_formants
        )

        L_star = np.array([
            self.alpha * L_s + (1 - self.alpha) * spk_L[g]
            for L_s, g in zip(L_sample, groups)
        ])
        return L_star

    def _normalise_formants(
        self,
        formants_hz: np.ndarray,
        L_star: np.ndarray,
    ) -> np.ndarray:
        """Return formants in the space that will be whitened."""
        if self.mode == "blended":
            F_ideal = (_HARM[None, :] * _C) / (4.0 * L_star[:, None])
            return formants_hz / F_ideal   # dimensionless ratio
        return formants_hz                  # raw Hz for sample mode

    def _check_fitted(self):
        if self._F_mean is None:
            raise RuntimeError("Call fit() before transform().")

    def __repr__(self) -> str:
        return (
            f"FormantPreprocessor(mode={self.mode!r}, alpha={self.alpha}, "
            f"use_formants={self.use_formants})"
        )

