# targets.py
import numpy as np
from collections import defaultdict

C = 350.0
HARM = np.array([1, 3, 5, 7], dtype=float)

def estimate_L_per_formant(formants_hz):
    return C * HARM / (4.0 * formants_hz)

def estimate_speaker_L(groups, L_if, use_formants=(0,1,2)):
    per_group = defaultdict(list)
    for g, L_row in zip(groups, L_if):
        per_group[g].append(np.median(L_row[list(use_formants)]))
    return {g: np.median(vals) for g, vals in per_group.items()}

def blend_sample_and_speaker_L(groups, L_if, speaker_L, alpha=0.4, use_formants=(0,1,2)):
    N = L_if.shape[0]
    L_star = np.zeros(N)
    for i in range(N):
        L_sample = np.median(L_if[i, list(use_formants)])
        L_star[i] = alpha * L_sample + (1-alpha) * speaker_L[groups[i]]
    return L_star

def build_phys_targets(formants_hz, groups, alpha=0.4):
    formants_hz = np.asarray(formants_hz)
    N = formants_hz.shape[0]

    # 1) per-formant VTL
    L_if = estimate_L_per_formant(formants_hz)

    # 2) speaker-level VTL
    speaker_L = estimate_speaker_L(groups, L_if)

    # 3) blended VTL
    L_star = blend_sample_and_speaker_L(groups, L_if, speaker_L, alpha=alpha)

    # 4) inverse-VTL
    invL = 1.0 / L_star

    # 5) normalise inverse-VTL
    invL_mean = invL.mean()
    invL_std  = invL.std()
    invL_norm = (invL - invL_mean) / invL_std

    # 6) predicted ideal formants from L*
    F_pred = (HARM[None, :] * C) / (4.0 * L_star[:, None])

    # 7) raw normalised formants
    F_norm = formants_hz / F_pred  # (N, 4)

    # 8) joint whitening of F_norm
    F_mean = F_norm.mean(axis=0)
    F_centered = F_norm - F_mean

    cov = np.cov(F_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    F_whitened = F_centered @ inv_sqrt_cov

    # 9) build target
    y_phys = np.zeros((N, 5))
    y_phys[:, 0] = invL_norm
    y_phys[:, 1:] = F_whitened

    meta = {
        "L_star": L_star,
        "invL_mean": invL_mean,
        "invL_std": invL_std,
        "F_mean": F_mean,
        "inv_sqrt_cov": inv_sqrt_cov,
    }
    return y_phys, meta

def invert_phys_targets(y_phys, meta):
    # inverse-VTL
    invL_norm = y_phys[:, 0]
    invL = invL_norm * meta["invL_std"] + meta["invL_mean"]
    L_star_pred = 1.0 / invL

    # de-whiten F_norm
    F_whitened = y_phys[:, 1:]
    F_centered = F_whitened @ np.linalg.inv(meta["inv_sqrt_cov"])
    F_norm = F_centered + meta["F_mean"]

    # reconstruct formants
    F_pred_base = (HARM[None, :] * C) / (4.0 * L_star_pred[:, None])
    F_hz = F_norm * F_pred_base
    return L_star_pred, F_hz
