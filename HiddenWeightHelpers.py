# =========================
# HWHelpers.py — plotting, stats, saving, normalization, symmetry, open-loop (Py3.6 compatible)
# =========================
import os, json, re
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- Plotting ----------


def plot_weight_all(
    W, title="Weights", bins=60, show_unit_circle=True, unit_radius=1.0
):
    """Heatmap, histogram, eigenspectrum (with unit circle)."""
    eig = np.linalg.eigvals(W)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # heatmap
    im = axes[0].imshow(W, aspect="auto", origin="upper")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"{title} Heatmap")

    # histogram
    axes[1].hist(W.ravel(), bins=bins)
    axes[1].set_title(f"{title} Histogram")

    # eigenspectrum
    radius = float(np.max(np.abs(eig))) if eig.size else 0.0
    axes[2].scatter(eig.real, eig.imag, s=10)

    # spectral-radius circle
    circle = plt.Circle((0, 0), radius, fill=False, linestyle="--")
    axes[2].add_artist(circle)

    # unit circle
    if show_unit_circle and unit_radius is not None and unit_radius > 0:
        unit = plt.Circle(
            (0, 0),
            unit_radius,
            fill=False,
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            color="tab:red",
        )
        axes[2].add_artist(unit)

    axes[2].axhline(0, lw=0.5, color="k")
    axes[2].axvline(0, lw=0.5, color="k")
    axes[2].set_aspect("equal", "box")

    lim = max(
        radius,
        unit_radius if show_unit_circle else 0.0,
        np.max(np.abs(eig.real)) if eig.size else 1.0,
        np.max(np.abs(eig.imag)) if eig.size else 1.0,
    )
    lim = 1.05 * (lim if lim > 0 else 1.0)
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_title(f"{title} eigvals | spectral radius ≈ {radius:.3f}")

    plt.tight_layout()
    plt.show()


def plot_hidden_heatmap(H: np.ndarray, title: str = "Hidden trajectory (linear OL)"):
    plt.figure(figsize=(10, 3))
    plt.imshow(H.T, aspect="auto", origin="lower")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------- Stats, summaries, saving ----------


def svd_sigma_max_np(W: np.ndarray) -> float:
    return float(np.linalg.svd(W, compute_uv=False)[0])


def with_stats_meta(W: np.ndarray, extra: Optional[Dict] = None) -> Dict:
    eig = np.linalg.eigvals(W)
    meta = {
        "shape": list(W.shape),
        "mean": float(W.mean()),
        "var": float(((W - W.mean()) ** 2).mean()),
        "fro_norm": float(np.linalg.norm(W)),
        "sigma_max": svd_sigma_max_np(W),
        "spectral_radius_abs_eigs": float(np.max(np.abs(eig))) if eig.size else 0.0,
        "asymmetry_ratio": float(np.linalg.norm(W - W.T) / (np.linalg.norm(W) + 1e-12)),
    }
    if extra:
        meta.update(extra)
    return meta


def summarize_matrix(W: np.ndarray, name: str, target_gain: Optional[float] = None):
    m = with_stats_meta(W)
    line = (
        f"[{name:>20}] var={m['var']:.6g}  σ_max={m['sigma_max']:.4f}  "
        f"ρ={m['spectral_radius_abs_eigs']:.4f}  asym={m['asymmetry_ratio']:.4f}  mean={m['mean']:.3e}"
    )
    if target_gain is not None:
        err = abs(m["sigma_max"] - target_gain)
        rel = err / max(target_gain, 1e-12)
        line += f"  |  gain_err={err:.3e} (rel {100*rel:.2f}%)"
    print(line)


def summarize_many(named_mats, target_gain: Optional[float] = None):
    print("=== Weight init summary ===")
    for name, W in named_mats:
        summarize_matrix(W, name, target_gain)
    print("===========================")


def save_matrix(
    W: np.ndarray, save_dir: str, name: str, meta: Optional[Dict] = None
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, name)
    np.save(base + ".npy", W)
    meta_all = with_stats_meta(W, extra=(meta or {}))
    with open(base + ".json", "w") as f:
        json.dump(meta_all, f, indent=2)
    print(f"Saved: {base}.npy and .json")


def gain_tag(g: float) -> str:
    return "gain" + re.sub(r"\.", "p", f"{g:.2f}")


# ---------- Normalizations ----------


def normalize_by_variance(W: np.ndarray, target_var: float) -> Tuple[np.ndarray, Dict]:
    W = W.astype(np.float32, copy=True)
    mu = float(W.mean())
    var = float(((W - mu) ** 2).mean())
    if var <= 0:
        return W, {
            "scale": 1.0,
            "var_before": var,
            "var_after": var,
            "status": "degenerate",
        }
    s = np.sqrt(target_var / var)
    W2 = s * W
    var2 = float(((W2 - float(W2.mean())) ** 2).mean())
    return W2, {"scale": s, "var_before": var, "var_after": var2, "status": "ok"}


def normalize_by_fro(W: np.ndarray, target_fro: float) -> Tuple[np.ndarray, Dict]:
    W = W.astype(np.float32, copy=True)
    fro = float(np.linalg.norm(W, ord="fro"))
    if fro <= 0:
        return W, {
            "scale": 1.0,
            "fro_before": fro,
            "fro_after": fro,
            "status": "degenerate",
        }
    s = target_fro / fro
    W2 = s * W
    fro2 = float(np.linalg.norm(W2, ord="fro"))
    return W2, {"scale": s, "fro_before": fro, "fro_after": fro2, "status": "ok"}


def normalize_by_spectral(
    W: np.ndarray, target_sigma: float
) -> Tuple[np.ndarray, Dict]:
    W = W.astype(np.float32, copy=True)
    smax = svd_sigma_max_np(W)
    if smax <= 0:
        return W, {
            "scale": 1.0,
            "sigma_before": smax,
            "sigma_after": smax,
            "status": "degenerate",
        }
    s = target_sigma / smax
    W2 = s * W
    smax2 = svd_sigma_max_np(W2)
    return W2, {"scale": s, "sigma_before": smax, "sigma_after": smax2, "status": "ok"}


# ---- Structured noise + Frobenius renorm ------------------------------------
def add_noise_preserve_structure(
    W: np.ndarray,
    noise_std: float = 1e-2,
    mode: str = "support_only",  # "support_only" | "offdiag" | "all"
    sym_mode: str = "none",  # "none" | "sym" | "skew" | "mix"
    sym_mix: float = 0.2,  # if sym_mode == "mix": W' = (1-sym_mix)*sym + sym_mix*skew
    seed: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Add small Gaussian noise without destroying the motif; then Frobenius-renormalize
    back to the original Frobenius norm.

    Recommended defaults:
      - shift / shiftcyc        -> mode="offdiag" or "support_only"
      - mex-hat (toeplitz/cyc)  -> mode="support_only"
    """
    # remember target Frobenius (scale)
    target_fro = float(np.linalg.norm(W, ord="fro"))

    # 1) add small Gaussian noise (your util scales by 1/sqrt(H))
    Wn, info_n = add_gaussian_noise_np(W, noise_std=noise_std, seed=seed, mode=mode)

    # 2) (optional) project to symmetry/skew subspace
    if sym_mode != "none":
        S, K = decompose_sym_skew(Wn)
        if sym_mode == "sym":
            Wn = S
        elif sym_mode == "skew":
            Wn = K
        elif sym_mode == "mix":
            Wn = (1.0 - sym_mix) * S + sym_mix * K

    # 3) renormalize back to same Frobenius norm
    Wn, info_f = normalize_by_fro(Wn, target_fro=target_fro)

    info = {
        "noise": info_n,
        "fro_renorm": info_f,
        "sym_mode": sym_mode,
        "sym_mix": sym_mix,
    }
    return Wn.astype(np.float32), info


# ---------- Symmetric / skew decompositions ----------


def decompose_sym_skew(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    W_sym = 0.5 * (W + W.T)
    W_skew = 0.5 * (W - W.T)
    return W_sym.astype(np.float32), W_skew.astype(np.float32)


def plot_sym_asym(W: np.ndarray, base_title: str = "W"):
    W_sym, W_skew = decompose_sym_skew(W)
    summarize_matrix(W_sym, f"{base_title} — symmetric")
    summarize_matrix(W_skew, f"{base_title} — skew")
    plot_weight_all(W_sym, title=f"{base_title} (symmetric)")
    plot_weight_all(W_skew, title=f"{base_title} (skew / asymmetric)")


# ---------- Model helpers (PyTorch Elman) ----------


def get_numpy_weights(model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      W_hh: (H,H), W_xh: (H,I), W_hy: (O,H)
    """
    W_hh = model.rnn.weight_hh_l0.detach().cpu().numpy()
    W_xh = model.rnn.weight_ih_l0.detach().cpu().numpy()
    W_hy = model.linear.weight.detach().cpu().numpy()
    return W_hh, W_xh, W_hy


@torch.no_grad()
def estimate_J_tanh_from_model(
    net: nn.Module,
    X_batch: torch.Tensor,
    h0: torch.Tensor,
    W_hh_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model to estimate J_tanh ≈ E[diag(1 - h^2)] from hidden states.
    Optionally evaluate with a temporary W_hh_override (then restore).
    Returns:
      gamma: (H,) np array, and J: diag matrix (H,H) np array.
    """
    # optional temporary override
    orig = None
    if W_hh_override is not None:
        p = net.rnn.weight_hh_l0
        orig = p.detach().cpu().clone()
        p.data.copy_(torch.tensor(W_hh_override, dtype=p.dtype, device=p.device))

    net.eval()
    out, z = net(X_batch, h0)  # z: (B,T,H)
    gamma = (1.0 - z**2).mean(dim=(0, 1)).detach().cpu().numpy()  # (H,)
    J = np.diag(gamma.astype(np.float32))

    # restore original W_hh
    if orig is not None:
        net.rnn.weight_hh_l0.data.copy_(orig.to(net.rnn.weight_hh_l0.device))

    return gamma, J


def make_A_open(W_hh: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    A_open = J_tanh @ W_hh   (row-wise scaling by gamma)
    """
    return gamma[:, None].astype(np.float32) * W_hh.astype(np.float32)


def scale_hidden_to_open_loop_gain(
    W_hh: np.ndarray,
    gamma: np.ndarray,
    target_gain: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Scale ONLY W_hh -> α W_hh so that σ_max(J_tanh · (α W_hh)) = target_gain.
    Since σ_max(J·(αW)) = |α| σ_max(J·W), we can do it in one step.
    """
    A0 = make_A_open(W_hh, gamma)
    s0 = svd_sigma_max_np(A0)
    if s0 == 0.0:
        return W_hh.copy(), {
            "alpha": 1.0,
            "s_before": 0.0,
            "s_after": 0.0,
            "status": "degenerate",
        }
    alpha = float(target_gain / s0)
    W_scaled = (alpha * W_hh).astype(np.float32)
    A_after = make_A_open(W_scaled, gamma)
    return W_scaled, {
        "alpha": alpha,
        "s_before": s0,
        "s_after": svd_sigma_max_np(A_after),
        "status": "ok",
    }


@torch.no_grad()
def open_loop_gain_match_and_plots(
    net: nn.Module,
    X_batch: torch.Tensor,
    h0: torch.Tensor,
    W_hh_init: np.ndarray,
    target_gain: float,
    label: str = "init",
    do_plots: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    1) Sym/Skew plots for W_hh_init.
    2) Estimate J_tanh on data (with temporary override to W_hh_init).
    3) Build A_open and print stats/gain.
    4) Rescale W_hh to hit target_gain for A_open; verify; plot.
    """
    if do_plots:
        plot_sym_asym(W_hh_init, base_title=f"{label}: W_hh (before)")

    gamma, J = estimate_J_tanh_from_model(net, X_batch, h0, W_hh_override=W_hh_init)
    summarize_matrix(J, f"{label}: J_tanh (diag)")
    if do_plots:
        plot_weight_all(J, title=f"{label}: J_tanh (diag)")

    A_open_before = make_A_open(W_hh_init, gamma)
    summarize_matrix(A_open_before, f"{label}: A_open (before)")
    if do_plots:
        plot_weight_all(A_open_before, title=f"{label}: A_open (before)")

    W_scaled, info = scale_hidden_to_open_loop_gain(W_hh_init, gamma, target_gain)
    print(f"{label}: open-loop gain match info -> {info}")

    A_open_after = make_A_open(W_scaled, gamma)
    summarize_matrix(A_open_after, f"{label}: A_open (after)", target_gain=target_gain)
    if do_plots:
        plot_weight_all(W_scaled, title=f"{label}: W_hh (after gain match)")
        plot_weight_all(
            A_open_after, title=f"{label}: A_open (after, target={target_gain})"
        )

    return W_scaled, info


def add_gaussian_noise_np(
    W: np.ndarray,
    noise_std: float,
    seed: Optional[int] = None,
    mode: str = "all",  # "all" | "offdiag" | "support_only"
    scale_by_sqrtN: bool = True,  # scale noise by 1/sqrt(H)
) -> Tuple[np.ndarray, Dict]:
    """
    Add small i.i.d. Gaussian noise to W.

    Effective noise added is: noise_std * N(0,1) / sqrt(H) if scale_by_sqrtN=True.

    Args
    ----
    W : (H,H) np.ndarray
    noise_std : float
        Amplitude multiplier for the Gaussian noise.
    seed : Optional[int]
    mode : str
        "all"          -> add noise to all entries
        "offdiag"      -> no noise on diagonal
        "support_only" -> add noise only where |W|>0 (preserve sparsity pattern)
    scale_by_sqrtN : bool
        If True, divide the raw noise by sqrt(H).

    Returns
    -------
    W_noisy : (H,H) np.ndarray
    info : dict with summary of added noise
    """
    rng = np.random.default_rng(seed)
    H = W.shape[0]
    noise = rng.normal(size=W.shape).astype(np.float32)

    if scale_by_sqrtN and H > 0:
        noise = noise / np.sqrt(H)

    if mode == "offdiag":
        np.fill_diagonal(noise, 0.0)
    elif mode == "support_only":
        mask = (np.abs(W) > 0).astype(np.float32)
        noise = noise * mask

    W_noisy = W.astype(np.float32) + float(noise_std) * noise

    info = {
        "mode": mode,
        "noise_std": float(noise_std),
        "effective_entry_std": float(noise_std / np.sqrt(H))
        if scale_by_sqrtN and H > 0
        else float(noise_std),
        "scale_by_sqrtN": bool(scale_by_sqrtN),
    }
    return W_noisy, info
