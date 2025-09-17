# =========================
# Setup & plotting
# =========================
import os, json, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import torch.nn as nn


def plot_weight_all(
    W, title="Weights", bins=60, show_unit_circle=True, unit_radius=1.0
):
    """Plot heatmap, histogram, and eigenspectrum in one figure (3 cols)."""

    eig = np.linalg.eigvals(W)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Heatmap ---
    im = axes[0].imshow(
        W,
        aspect="auto",
        origin="upper",
    )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"{title} Heatmap")

    # --- Histogram ---
    axes[1].hist(W.ravel(), bins=bins)
    axes[1].set_title(f"{title} Histogram")

    # --- Eigenspectrum ---
    radius = float(np.max(np.abs(eig)))
    axes[2].scatter(eig.real, eig.imag, s=10)

    # spectral-radius circle (existing)
    circle = plt.Circle((0, 0), radius, fill=False, linestyle="--")
    axes[2].add_artist(circle)

    # NEW: unit circle overlay
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

    # axes styling
    axes[2].axhline(0, lw=0.5, color="k")
    axes[2].axvline(0, lw=0.5, color="k")
    axes[2].set_aspect("equal", "box")

    # set limits to fit points and the largest of {radius, unit_radius}
    lim = max(
        radius,
        unit_radius if show_unit_circle else 0.0,
        np.max(np.abs(eig.real)),
        np.max(np.abs(eig.imag)),
    )
    lim = 1.05 * (lim if lim > 0 else 1.0)
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)

    axes[2].set_title(f"{title} eigvals | spectral radius ≈ {radius:.3f}")

    plt.tight_layout()
    plt.show()


# =========================
# gain matching + utilities
# =========================
def svd_sigma_max_np(W: np.ndarray) -> float:
    return float(np.linalg.svd(W, compute_uv=False)[0])


def scale_to_gain_np(W: np.ndarray, target_gain: float) -> Tuple[np.ndarray, float]:
    """Return (scaled_W, sigma_before). Scales W so σ_max == target_gain."""
    smax = svd_sigma_max_np(W)
    if smax > 0.0:
        W = (target_gain / smax) * W
    return W, smax


def gain_tag(g: float) -> str:
    # e.g., 0.90 -> "gain0p90"
    return "gain" + re.sub(r"\.", "p", f"{g:.2f}")


def with_stats_meta(W: np.ndarray, extra: Optional[Dict] = None) -> Dict:
    eig = np.linalg.eigvals(W)
    meta: Dict = {
        "shape": list(W.shape),
        "mean": float(W.mean()),
        "var": float(((W - W.mean()) ** 2).mean()),
        "fro_norm": float(np.linalg.norm(W)),
        "sigma_max": svd_sigma_max_np(W),
        "spectral_radius_abs_eigs": float(np.max(np.abs(eig))),
        "asymmetry_ratio": float(np.linalg.norm(W - W.T) / (np.linalg.norm(W) + 1e-12)),
    }
    if extra:
        meta.update(extra)
    return meta


def save_matrix(
    W: np.ndarray, save_dir: str, name: str, meta: Optional[Dict] = None
) -> None:
    """
    Saves W as .npy and metadata as .json (always writes JSON with auto stats).
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, name)
    np.save(base + ".npy", W)
    meta_all = with_stats_meta(W, extra=(meta or {}))
    with open(base + ".json", "w") as f:
        json.dump(meta_all, f, indent=2)
    print(f"Saved: {base}.npy and .json")


# ----- Optional knobs to apply BEFORE the final gain scaling -----
def apply_sparsity_np(
    W: np.ndarray, p: float, seed: Optional[int] = None
) -> np.ndarray:
    """Keep each entry with prob p; zero diagonal afterwards."""
    if p >= 1.0:
        return W
    rng = np.random.default_rng(seed)
    mask = (rng.random(W.shape) < p).astype(W.dtype)
    W = W * mask
    np.fill_diagonal(W, 0.0)
    return W


def impose_dale_law_np(
    W: np.ndarray,
    frac_exc: float,
    balanced_rows: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Dale's law by rows (outgoing sign). Optionally zero-mean each row.
    """
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    E = int(round(frac_exc * n))
    signs = np.concatenate([np.ones(E, dtype=W.dtype), -np.ones(n - E, dtype=W.dtype)])
    rng.shuffle(signs)
    W = W * signs[:, None]
    if balanced_rows:
        W = W - W.mean(axis=1, keepdims=True)
    np.fill_diagonal(W, 0.0)
    return W


def add_gaussian_noise_np(
    W: np.ndarray, noise_std: float, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add small i.i.d. Gaussian noise (std scaled by 1/sqrt(n)) to avoid exact symmetries.
    """
    if noise_std <= 0:
        return W
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    W = W + (noise_std / np.sqrt(n)) * rng.normal(size=W.shape).astype(W.dtype)
    return W


# --- quick, readable stats for one matrix ---
def summarize_matrix(W: np.ndarray, name: str, target_gain: float = None):
    meta = with_stats_meta(W)
    mean = meta["mean"]
    var = meta["var"]
    smax = meta["sigma_max"]
    rho = meta["spectral_radius_abs_eigs"]
    asym = meta["asymmetry_ratio"]

    line = f"[{name:>20}] var={var:.6g}  σ_max={smax:.4f}  ρ={rho:.4f}  asym={asym:.4f}  mean={mean:.3e}"
    if target_gain is not None:
        err = abs(smax - target_gain)
        rel = err / max(target_gain, 1e-12)
        line += f"  |  gain_err={err:.3e} (rel {100*rel:.2f}%)"
    print(line)


# --- summarize many matrices at once ---
def summarize_many(named_mats, target_gain: float = None):
    """
    named_mats: list of (name, W) tuples
    """
    print("=== Weight init summary ===")
    for name, W in named_mats:
        summarize_matrix(W, name, target_gain)
    print("===========================")


# --- assert/check that gains are aligned ---
def check_gain_alignment(named_mats, target_gain: float, rtol: float = 1e-3):
    """
    Print OK/FAIL per matrix and return a boolean.
    rtol=1e-3 means within 0.1% relative error by default.
    """
    ok_all = True
    print(f"=== Gain alignment check (target_gain={target_gain}) ===")
    for name, W in named_mats:
        smax = svd_sigma_max_np(W)
        rel_err = abs(smax - target_gain) / max(target_gain, 1e-12)
        status = "OK" if rel_err <= rtol else "FAIL"
        print(f"[{name:>20}] σ_max={smax:.6f}  rel_err={100*rel_err:.3f}%  -> {status}")
        ok_all &= rel_err <= rtol
    print("===============================================")
    return ok_all


# =========================
# Builders (add target_gain while preserving your API)
# =========================


def build_he(
    n_hidden: int,
    target_gain: Optional[float] = None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Xavier-for-tanh baseline (std = 1/sqrt(n)), then optional gain match.
    Returns a torch.Tensor for parity with your original code.
    """
    n = n_hidden
    std = (1.0 / n) ** 0.5
    W_t = torch.randn((n, n), device=device, dtype=dtype) * std
    W = W_t.detach().cpu().numpy()
    if target_gain is not None:
        W, _ = scale_to_gain_np(W, target_gain)
    return torch.tensor(W, device=device, dtype=dtype)


def build_xavier(n_in, n_out, gain=1.0, seed=None, target_gain: Optional[float] = None):
    """Xavier/Glorot normal; optional gain match."""
    rng = np.random.default_rng(seed)
    std = gain * np.sqrt(2.0 / (n_in + n_out))
    W = rng.normal(0.0, std, size=(n_out, n_in)).astype(np.float32)
    if target_gain is not None:
        W, _ = scale_to_gain_np(W, target_gain)
    mean = W.mean()
    var = ((W - mean) ** 2).mean()
    return W, mean, var


def build_shift(
    n,
    off=1.0,
    target_var=None,
    cyclic=False,
    target_gain: Optional[float] = None,
    verbose=False,
):
    """Upper-shift (optionally cyclic), then optional variance report and gain match."""
    W = np.zeros((n, n), dtype=np.float32)
    idx = np.arange(n - 1)
    W[idx, idx + 1] = off
    if cyclic:
        W[-1, 0] = off

    # (optional) variance rescale for reporting parity
    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[shift] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[shift] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    # (primary) gain match
    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[shift] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")
    return W


def build_mexican_hat(
    n,
    sigma=None,
    target_var=None,
    cyclic=False,
    target_gain: Optional[float] = None,
    verbose=False,
):
    """
    1D Mexican-hat Toeplitz (or circulant if cyclic). Then optional variance rescale and gain match.
    """
    if sigma is None:
        sigma = n / 10
    if cyclic:
        d_vals = np.arange(n, dtype=np.int64)
        d_vals = np.minimum(d_vals, n - d_vals).astype(np.float64)
        k = (1.0 - (d_vals**2) / (sigma**2)) * np.exp(
            -(d_vals**2) / (2.0 * sigma**2)
        )
        idx = np.arange(n)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, n - d)
        W = k[d].astype(np.float32)
    else:
        center = n // 2
        x = np.arange(-center, n - center, dtype=np.float64)
        k = (1.0 - (x**2) / (sigma**2)) * np.exp(-(x**2) / (2.0 * sigma**2))
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        off = (i - j) + center
        mask = (off >= 0) & (off < n)
        W = np.zeros((n, n), dtype=np.float32)
        W[mask] = k[off[mask]].astype(np.float32)

    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[MH] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[MH] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[MH] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")
    return W


def build_tridiag(
    n,
    diag=1.0,
    off=-1.0,
    cyclic=False,
    target_var=None,
    target_gain: Optional[float] = None,
    verbose=False,
):
    """(Cyclic) tridiagonal; optional variance rescale and gain match."""
    W = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n)
    W[i, i] = diag
    W[i[1:], i[:-1]] = off
    W[i[:-1], i[1:]] = off
    if cyclic:
        W[0, -1] = off
        W[-1, 0] = off

    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[tri] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[tri] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[tri] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")
    return W


def build_orthogonal(
    n,
    scale=1.0,
    seed=None,
    target_var=None,
    target_gain: Optional[float] = None,
    verbose=False,
    ensure_det_pos=False,
):
    """Random orthogonal (QR), optional variance rescale, gain match."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n), loc=0.0, scale=1.0).astype(np.float64)
    Q, R = np.linalg.qr(A, mode="reduced")
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    if ensure_det_pos and np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    W = (scale * Q).astype(np.float32)

    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[orth] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[orth] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[orth] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")
    return W


@torch.no_grad()
def rnn_default_init_stats(
    input_dim,
    hidden_dim,
    *,
    nonlinearity="tanh",
    num_layers=1,
    bidirectional=False,
    repeats=1,
    seed=None,
    device="cpu",
    dtype=torch.float32,
):
    if seed is not None:
        torch.manual_seed(seed)

    def one_sample():
        rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
        ).to(device=device, dtype=dtype)

        stats = {}
        dirs = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(dirs):
                suf = f"l{layer}" + ("" if d == 0 else f"_reverse")
                Wih = getattr(
                    rnn, f"weight_ih_{suf}"
                )  # (H, I) for layer 0 else (H, D*H)
                Whh = getattr(rnn, f"weight_hh_{suf}")  # (H, H)
                bih = getattr(rnn, f"bias_ih_{suf}")
                bhh = getattr(rnn, f"bias_hh_{suf}")

                stats[f"Wih_{suf}_mean"] = float(Wih.mean())
                stats[f"Wih_{suf}_var"] = float(Wih.var(unbiased=False))
                stats[f"Whh_{suf}_mean"] = float(Whh.mean())
                stats[f"Whh_{suf}_var"] = float(Whh.var(unbiased=False))
                stats[f"bih_{suf}_var"] = float(bih.var(unbiased=False))
                stats[f"bhh_{suf}_var"] = float(bhh.var(unbiased=False))
        return stats

    # repeat to average across random seeds
    out = None
    for i in range(repeats):
        if seed is not None:
            torch.manual_seed(seed + i)
        s = one_sample()
        if out is None:
            out = {k: 0.0 for k in s}
        for k, v in s.items():
            out[k] += v
    if repeats > 1:
        for k in out:
            out[k] /= repeats
    return out
