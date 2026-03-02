# train_head.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from utils import labels_to_ints, build_concat_embeddings
from heads import WhisperTwoHead, TwoHeadPooled, TwoHeadPooledPhys, ThreeHeadPooled


def normalise_formants(formants):
    mean = formants.mean(axis=0)
    std  = formants.std(axis=0)
    return (formants - mean) / std, mean, std


def train_one_split(model, train_dl, val_dl, epochs=200, lr=1e-3, patience=20):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce  = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_left = patience
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb_phys, xb_vowel, yv, yf in train_dl:
            opt.zero_grad()
            out = model(xb_phys, xb_vowel)
            loss = ce(out["vowels"], yv) + mse(out["formants"], yf)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb_phys, xb_vowel, yv, yf in val_dl:
                out = model(xb_phys, xb_vowel)
                loss = ce(out["vowels"], yv) + mse(out["formants"], yf)
                val_loss += loss.item()

        val_loss /= len(val_dl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss

from targets import build_phys_targets, invert_phys_targets
from heads import TwoHeadPooled
from utils import labels_to_ints
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_multihead_kfold_phys(
    embeddings,
    labels,
    formants,
    groups,
    layer_formant=4,
    layer_vowel=12,
    k=5,
    epochs=200,
    lr=1e-3,
    batch_size=64
):
    # labels → ints
    y_ints, mapping = labels_to_ints(labels)
    y_vowel = np.array(y_ints)

    # build physical targets (invL_norm + whitened F_norm)
    y_phys, meta = build_phys_targets(formants, groups, alpha=0.4)  # (N, 5)

    # Layers for physical head (VTL + normalised/whitened formants)
    layers_phys = [0, 1, 2, 3, 4]

    # Layers for vowel head (vowel identity)
    layers_vowel = [12]

    X_phys  = build_concat_embeddings(embeddings, layers_phys)
    X_vowel = build_concat_embeddings(embeddings, layers_vowel)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_phys)):
        print(f"\n=== Fold {fold+1}/{k} ===")

        X_phys_tr, X_phys_va   = X_phys[train_idx], X_phys[val_idx]
        X_vowel_tr, X_vowel_va = X_vowel[train_idx], X_vowel[val_idx]
        yv_tr, yv_va   = y_vowel[train_idx], y_vowel[val_idx]
        yp_tr, yp_va   = y_phys[train_idx], y_phys[val_idx]
        F_va_true      = formants[val_idx]

        # ground truth L* from phys targets
        invL_true_norm = yp_va[:, 0]
        invL_true = invL_true_norm * meta["invL_std"] + meta["invL_mean"]
        L_true = 1.0 / invL_true

        train_ds = TensorDataset(
            torch.tensor(X_phys_tr).float(),
            torch.tensor(X_vowel_tr).float(),
            torch.tensor(yv_tr).long(),
            torch.tensor(yp_tr).float()
        )
        val_ds = TensorDataset(
            torch.tensor(X_phys_va).float(),
            torch.tensor(X_vowel_va).float(),
            torch.tensor(yv_va).long(),
            torch.tensor(yp_va).float()
        )

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=batch_size)

        model = TwoHeadPooledPhys(
            d_phys=X_phys.shape[1],
            d_vowel=X_vowel.shape[1],
            num_classes=len(mapping),
        )

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        ce  = torch.nn.CrossEntropyLoss()
        mse = torch.nn.MSELoss()

        best_state = None
        best_val = float("inf")
        patience = 20
        patience_left = patience

        for epoch in range(epochs):
            model.train()
            for xb_phys, xb_vowel, yv, yp in train_dl:
                opt.zero_grad()
                out = model(xb_phys, xb_vowel)

                # phys: [invL_norm, F1_white, F2_white, F3_white, F4_white]
                pred_invL   = out["phys"][:, 0]
                pred_Fwhite = out["phys"][:, 1:]

                true_invL   = yp[:, 0]
                true_Fwhite = yp[:, 1:]

                loss_phys = 2.0 * mse(pred_invL, true_invL) + mse(pred_Fwhite, true_Fwhite)
                loss = ce(out["vowels"], yv) + loss_phys

                loss.backward()
                opt.step()

            # val
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb_phys, xb_vowel, yv, yp in val_dl:
                    out = model(xb_phys, xb_vowel)

                    pred_invL   = out["phys"][:, 0]
                    pred_Fwhite = out["phys"][:, 1:]

                    true_invL   = yp[:, 0]
                    true_Fwhite = yp[:, 1:]

                    loss_phys = 2.0 * mse(pred_invL, true_invL) + mse(pred_Fwhite, true_Fwhite)
                    loss = ce(out["vowels"], yv) + loss_phys

                    val_loss += loss.item()
            val_loss /= len(val_dl)

            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)

        # evaluate on this fold in Hz space
        model.eval()
        with torch.no_grad():
            out = model(
                torch.tensor(X_phys_va).float(),
                torch.tensor(X_vowel_va).float()
            )
            pred_vowel = out["vowels"].argmax(dim=1).cpu().numpy()
            pred_phys  = out["phys"].cpu().numpy()  # (N_val, 5)

        # reconstruct VTL + formants from whitened space
        L_va_pred, F_va_pred = invert_phys_targets(pred_phys, meta)

        # --- group-level diagnostics ---
        groups_va = np.array(groups)[val_idx]
        unique_groups = np.unique(groups_va)

        group_stats = []
        for g in unique_groups:
            mask = (groups_va == g)

            L_true_g = L_true[mask]
            L_pred_g = L_va_pred[mask]

            F_true_g = F_va_true[mask]
            F_pred_g = F_va_pred[mask]

            group_stats.append({
                "group": g,
                "L_true_mean": float(L_true_g.mean()),
                "L_pred_mean": float(L_pred_g.mean()),
                "F_true_mean": F_true_g.mean(axis=0).tolist(),
                "F_pred_mean": F_pred_g.mean(axis=0).tolist(),
                "n": int(mask.sum())
            })

        print("\nGroup-level means:")
        for s in group_stats:
            print(
                f"  {s['group']}: "
                f"L_true={s['L_true_mean']:.4f}, L_pred={s['L_pred_mean']:.4f}, "
                f"F_true={np.round(s['F_true_mean'],1)}, "
                f"F_pred={np.round(s['F_pred_mean'],1)}, "
                f"n={s['n']}"
            )

        # metrics
        vowel_acc = accuracy_score(yv_va, pred_vowel)

        vtl_r2  = r2_score(L_true, L_va_pred)
        vtl_mse = mean_squared_error(L_true, L_va_pred)

        formant_r2 = [r2_score(F_va_true[:, i], F_va_pred[:, i]) for i in range(4)]
        formant_mse = [mean_squared_error(F_va_true[:, i], F_va_pred[:, i]) for i in range(4)]

        fold_results.append({
            "val_loss": best_val,
            "vowel_acc": vowel_acc,
            "vtl_r2": vtl_r2,
            "vtl_mse": vtl_mse,
            "formant_r2": formant_r2,
            "formant_mse": formant_mse,
        })

        print(f"Fold {fold+1} vowel accuracy: {vowel_acc:.3f}")
        print(f"Fold {fold+1} VTL R²: {vtl_r2:.3f}, MSE: {vtl_mse:.3f}")
        print(f"Fold {fold+1} formant R²: {formant_r2}")

    return fold_results

def train_multihead_kfold(
    embeddings,
    labels,
    formants,
    groups,
    layers_phys=[0,1,2,3,4],
    layers_vowel=[12],
    k=5,
    epochs=200,
    lr=1e-3,
    batch_size=64
):
    # labels → ints
    y_ints, mapping = labels_to_ints(labels)
    y_vowel = np.array(y_ints)

    # --- JOINT WHITENING OF FORMANTS ---
    F = np.asarray(formants)
    F_mean = F.mean(axis=0)
    F_center = F - F_mean
    cov = np.cov(F_center, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    F_white = F_center @ inv_sqrt_cov

    # --- VTL TARGET (OPTIONAL, NOT USED FOR FORMANTS) ---
    L_if = 350 * np.array([1,3,5,7])[None,:] / (4 * F)  # per-formant VTL
    L_sample = np.median(L_if[:, :3], axis=1)
    invL = 1.0 / L_sample
    invL_mean = invL.mean()
    invL_std  = invL.std()
    invL_norm = (invL - invL_mean) / invL_std

    # --- CONCAT EMBEDDINGS ---
    X_phys  = build_concat_embeddings(embeddings, layers_phys)
    X_vowel = build_concat_embeddings(embeddings, layers_vowel)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_phys)):
        print(f"\n=== Fold {fold+1}/{k} ===")

        Xp_tr, Xp_va = X_phys[train_idx], X_phys[val_idx]
        Xv_tr, Xv_va = X_vowel[train_idx], X_vowel[val_idx]

        yv_tr, yv_va = y_vowel[train_idx], y_vowel[val_idx]
        yf_tr, yf_va = F_white[train_idx], F_white[val_idx]
        yL_tr, yL_va = invL_norm[train_idx], invL_norm[val_idx]

        L_if = 350 * np.array([1,3,5,7])[None,:] / (4 * formants[val_idx])
        L_true = np.median(L_if[:, :3], axis=1)

        train_ds = TensorDataset(
            torch.tensor(Xp_tr).float(),
            torch.tensor(Xv_tr).float(),
            torch.tensor(yv_tr).long(),
            torch.tensor(yf_tr).float(),
            torch.tensor(yL_tr).float()
        )
        val_ds = TensorDataset(
            torch.tensor(Xp_va).float(),
            torch.tensor(Xv_va).float(),
            torch.tensor(yv_va).long(),
            torch.tensor(yf_va).float(),
            torch.tensor(yL_va).float()
        )

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=batch_size)

        model = ThreeHeadPooled(
            d_phys=Xp_tr.shape[1],
            d_vowel=Xv_tr.shape[1],
            num_classes=len(mapping),
            formant_dim=formants.shape[1]
        )

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        ce  = torch.nn.CrossEntropyLoss()
        mse = torch.nn.MSELoss()

        best_state = None
        best_val = float("inf")
        patience = 20
        patience_left = patience

        for epoch in range(epochs):
            model.train()
            for xb_phys, xb_vowel, yv, yf, yL in train_dl:
                opt.zero_grad()
                out = model(xb_phys, xb_vowel)

                loss_vowel = ce(out["vowels"], yv)
                loss_form  = mse(out["formants"], yf)
                loss_vtl   = mse(out["vtl"], yL)

                loss = loss_vowel + loss_form + 0.5 * loss_vtl
                loss.backward()
                opt.step()

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb_phys, xb_vowel, yv, yf, yL in val_dl:
                    out = model(xb_phys, xb_vowel)
                    loss = (
                        ce(out["vowels"], yv)
                        + mse(out["formants"], yf)
                        + 0.5 * mse(out["vtl"], yL)
                    )
                    val_loss += loss.item()
            val_loss /= len(val_dl)

            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            out = model(
                torch.tensor(Xp_va).float(),
                torch.tensor(Xv_va).float()
            )
            # --- Model outputs ---
            pred_vowel = out["vowels"].argmax(dim=1).cpu().numpy()
            pred_form_white = out["formants"].cpu().numpy()
            invL_pred_norm = out["vtl"].cpu().numpy()

        # --- Unwhiten predicted formants ---
        pred_form = (pred_form_white @ np.linalg.inv(inv_sqrt_cov)) + F_mean

        # --- Unwhiten true formants (from validation targets) ---
        true_form_white = yf_va
        true_form = (true_form_white @ np.linalg.inv(inv_sqrt_cov)) + F_mean

        # --- Compute true VTL from raw formants ---
        L_if = 350 * np.array([1,3,5,7])[None,:] / (4 * formants[val_idx])
        L_true = np.median(L_if[:, :3], axis=1)

        # --- Compute predicted VTL ---
        invL_pred = invL_pred_norm * invL_std + invL_mean
        L_pred = 1.0 / invL_pred

        groups_va = np.array(groups)[val_idx]
        unique_groups = np.unique(groups_va)
        
        group_stats = []
        for g in unique_groups:
            mask = (groups_va == g)

            L_true_g = L_true[mask]
            L_pred_g = L_pred[mask]

            F_true_g = true_form[mask]
            F_pred_g = true_form[mask]

            group_stats.append({
                "group": g,
                "L_true_mean": float(L_true_g.mean()),
                "L_pred_mean": float(L_pred_g.mean()),
                "F_true_mean": F_true_g.mean(axis=0).tolist(),
                "F_pred_mean": F_pred_g.mean(axis=0).tolist(),
                "n": int(mask.sum())
            })

        print("\nGroup-level means:")
        for s in group_stats:
            print(
                f"  {s['group']}: "
                f"L_true={s['L_true_mean']:.4f}, L_pred={s['L_pred_mean']:.4f}, "
                f"F_true={np.round(s['F_true_mean'],1)}, "
                f"F_pred={np.round(s['F_pred_mean'],1)}, "
                f"n={s['n']}"
            )

        # --- Metrics ---
        vowel_acc = accuracy_score(yv_va, pred_vowel)
        formant_r2 = [r2_score(true_form[:, i], pred_form[:, i]) for i in range(4)]
        vtl_r2  = r2_score(L_true, L_pred)
        vtl_mse = mean_squared_error(L_true, L_pred)

        print(f"Fold {fold+1} vowel accuracy: {vowel_acc:.3f}")
        print(f"Fold {fold+1} formant R²: {formant_r2}")
        print(f"Fold {fold+1} VTL R²: {vtl_r2:.3f}, MSE: {vtl_mse:.6f}")


        fold_results.append({
            "val_loss": best_val,
            "vowel_acc": vowel_acc,
            "formant_r2": vtl_r2
        })

    return fold_results


def train_multihead(
    embeddings,
    labels,
    formants,
    layer_formant=4,
    layer_vowel=12,
    epochs=20,
    lr=1e-3,
    batch_size=64
):
    # Extract pooled embeddings for each head
    X_phys  = torch.tensor(np.stack(embeddings[layer_formant])).float()
    X_vowel = torch.tensor(np.stack(embeddings[layer_vowel])).float()

    y_ints, mapping = labels_to_ints(labels)
    y_vowel = torch.tensor(y_ints).long()
    y_form  = torch.tensor(formants).float()

    ds = TensorDataset(X_phys, X_vowel, y_vowel, y_form)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TwoHeadPooled(
        d_model=X_phys.shape[1],
        num_classes=len(set(labels)),
        formant_dim=formants.shape[1]
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce  = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb_phys, xb_vowel, yv, yf in dl:
            opt.zero_grad()
            out = model(xb_phys, xb_vowel)

            loss = ce(out["vowels"], yv) + mse(out["formants"], yf)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb_phys.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/len(ds):.4f}")

    return model



def train_whisper_multihead(embeddings, labels, num_classes, epochs=20, lr=1e-3, batch_size=64):
    X = torch.from_numpy(embeddings).float()      # (N, D)
    y = torch.from_numpy(labels).long()           # (N,)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = WhisperTwoHead(in_dim=X.shape[1], num_classes=num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}: loss={total_loss/len(ds):.4f}")

    return model


def evaluate_multihead(
    model,
    embeddings,
    labels,
    formants,
    layer_formant=4,
    layer_vowel=12
):
    """
    Evaluate the trained multi-head model on pooled embeddings.
    Returns a dict with:
        - vowel_accuracy
        - formant_mse (per formant)
        - formant_r2 (per formant)
    """

    # Convert labels → ints
    y_ints, mapping = labels_to_ints(labels)
    y_vowel = np.array(y_ints)
    y_form  = np.array(formants)

    # Extract pooled embeddings
    X_phys  = np.stack(embeddings[layer_formant])
    X_vowel = np.stack(embeddings[layer_vowel])

    X_phys_t  = torch.tensor(X_phys).float()
    X_vowel_t = torch.tensor(X_vowel).float()

    model.eval()
    with torch.no_grad():
        out = model(X_phys_t, X_vowel_t)
        pred_vowel = out["vowels"].argmax(dim=1).cpu().numpy()
        pred_form  = out["formants"].cpu().numpy()

    # Compute metrics
    vowel_acc = accuracy_score(y_vowel, pred_vowel)

    mse_per_formant = []
    r2_per_formant  = []

    for i in range(y_form.shape[1]):
        mse_per_formant.append(mean_squared_error(y_form[:, i], pred_form[:, i]))
        r2_per_formant.append(r2_score(y_form[:, i], pred_form[:, i]))

    return {
        "vowel_accuracy": vowel_acc,
        "formant_mse": mse_per_formant,
        "formant_r2": r2_per_formant,
        "mapping": mapping
    }
