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

    # unit circle overlay
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
    """
    Scale W by a scalar so that its spectral norm (largest singular value) == target_gain.
    Returns (scaled_W, original_sigma_max).
    """
    smax = svd_sigma_max_np(W)
    if smax > 0.0:
        W = (target_gain / smax) * W
    return W, smax


def gain_tag(g: float) -> str:
    """e.g. 0.90 -> 'gain0p90'"""
    import re

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


def get_numpy_weights(model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      W_hh: (H,H), W_xh: (H,I), W_hy: (O,H)
    NOTE: if you've fixed W_xh to identity in the notebook, this simply reads that identity.
    """
    W_hh = model.rnn.weight_hh_l0.detach().cpu().numpy()
    W_xh = model.rnn.weight_ih_l0.detach().cpu().numpy()
    W_hy = model.linear.weight.detach().cpu().numpy()
    return W_hh, W_xh, W_hy


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


def summarize_many(named_mats, target_gain: float = None):
    """
    named_mats: list of (name, W) tuples
    """
    print("=== Weight init summary ===")
    for name, W in named_mats:
        summarize_matrix(W, name, target_gain)
    print("===========================")


def decompose_sym_skew(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (W_sym, W_skew) where:
      W_sym  = (W + W.T)/2         (symmetric component)
      W_skew = (W - W.T)/2         (skew-symmetric (asymmetric) component)
    """
    W_sym = 0.5 * (W + W.T)
    W_skew = 0.5 * (W - W.T)
    return W_sym, W_skew


def plot_sym_asym(W: np.ndarray, base_title: str = "W"):
    """Plot symmetric and skew components with your plot_weight_all."""
    W_sym, W_skew = decompose_sym_skew(W)
    summarize_matrix(W_sym, f"{base_title} — symmetric")
    summarize_matrix(W_skew, f"{base_title} — skew")
    plot_weight_all(W_sym, title=f"{base_title} (symmetric)")
    plot_weight_all(W_skew, title=f"{base_title} (skew / asymmetric)")


def estimate_J_tanh_from_model(
    net: nn.Module,
    X_batch: torch.Tensor,
    h0: torch.Tensor,
    W_hh_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model (optionally with a temporary W_hh override) to estimate:
      gamma: (H,)  with gamma_i = E[ 1 - h_i^2 ] over batch & time
      J_tanh = diag(gamma) as a dense (H,H) numpy array
    """
    # optional temporary override of W_hh so J_tanh matches *that* W
    old = None
    if W_hh_override is not None:
        with torch.no_grad():
            old = net.rnn.weight_hh_l0.detach().cpu().clone()
            net.rnn.weight_hh_l0.copy_(
                torch.tensor(
                    W_hh_override,
                    dtype=net.rnn.weight_hh_l0.dtype,
                    device=net.rnn.weight_hh_l0.device,
                )
            )

    net.eval()
    with torch.no_grad():
        # forward uses your Elman class: returns (probs, hidden_states)
        probs, z = net(X_batch, h0)  # z: (B,T,H)
        gamma_t = (1.0 - z**2).mean(dim=(0, 1))  # (H,)
        gamma = gamma_t.detach().cpu().numpy()

    # restore original W_hh if overridden
    if old is not None:
        with torch.no_grad():
            net.rnn.weight_hh_l0.copy_(old)

    J = np.diag(gamma.astype(np.float32))
    return gamma, J


def make_A_open(W_hh: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    A_open = diag(gamma) @ W_hh  (implemented as row-wise scaling gamma[:,None]*W_hh)
    """
    return (gamma[:, None] * W_hh).astype(np.float32)


def open_loop_gain(W_hh: np.ndarray, gamma: np.ndarray) -> float:
    """σ_max(A_open)"""
    return svd_sigma_max_np(make_A_open(W_hh, gamma))


def scale_hidden_to_open_loop_gain(
    W_hh: np.ndarray,
    gamma: np.ndarray,
    target_gain: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Scale ONLY W_hh -> alpha * W_hh so that σ_max( J_tanh · (alpha W_hh) ) = target_gain.
    With fixed J_tanh estimate (from current data), this is a one-liner:
      alpha = target_gain / σ_max( J_tanh · W_hh )
    Returns (scaled_W_hh, info).
    """
    A = make_A_open(W_hh, gamma)
    s_before = svd_sigma_max_np(A)
    if s_before <= 0:
        return W_hh.copy(), {
            "alpha": 1.0,
            "s_before": s_before,
            "s_after": s_before,
            "status": "degenerate",
        }
    alpha = float(target_gain / s_before)
    W_scaled = (alpha * W_hh).astype(np.float32)
    s_after = svd_sigma_max_np(make_A_open(W_scaled, gamma))
    info = {"alpha": alpha, "s_before": s_before, "s_after": s_after, "status": "ok"}
    return W_scaled, info


# ---------- simple linear open-loop simulation h_t = W_hh h_{t-1} + x_t ----------
def simulate_linear_open_loop(
    W_hh: np.ndarray,
    X_one: np.ndarray,  # shape (T, H) — one sequence of encodings
    h0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate h_t = W_hh h_{t-1} + x_t (ignoring tanh), useful for intuition/debug.
    Returns H of shape (T, H).
    """
    T, H = X_one.shape
    h = np.zeros(H, dtype=np.float32) if h0 is None else h0.astype(np.float32)
    Hs = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        h = (W_hh @ h) + X_one[t]
        Hs[t] = h
    return Hs


def plot_hidden_heatmap(H: np.ndarray, title: str = "Hidden trajectory (linear OL)"):
    """Quick heatmap for the simulated linear open-loop trajectory."""
    plt.figure(figsize=(10, 3))
    plt.imshow(H.T, aspect="auto", origin="lower")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------- one-stop helper you can call from the notebook ----------
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
    For a proposed hidden matrix W_hh_init:
      1) Split into symmetric/skew and plot.
      2) Estimate J_tanh from data with *this* W_hh (temporary override).
      3) Build A_open, print stats & gain.
      4) Scale W_hh for target open-loop gain, verify, and plot.
    Returns (W_hh_scaled, info).
    """
    # (1) sym / skew
    if do_plots:
        plot_sym_asym(W_hh_init, base_title=f"{label}: W_hh (before)")

    # (2) J_tanh estimated on this W_hh
    gamma, J = estimate_J_tanh_from_model(net, X_batch, h0, W_hh_override=W_hh_init)
    summarize_matrix(J, f"{label}: J_tanh (diag)", target_gain=None)
    if do_plots:
        plot_weight_all(J, title=f"{label}: J_tanh (diag)")

    # (3) A_open before scaling
    A_open = make_A_open(W_hh_init, gamma)
    summarize_matrix(A_open, f"{label}: A_open (before)")
    if do_plots:
        plot_weight_all(A_open, title=f"{label}: A_open (before)")

    # (4) scale W to hit target open-loop gain
    W_scaled, info = scale_hidden_to_open_loop_gain(W_hh_init, gamma, target_gain)
    print(f"{label}: open-loop gain match info ->", info)

    # verify & plots
    A_after = make_A_open(W_scaled, gamma)
    summarize_matrix(A_after, f"{label}: A_open (after)", target_gain=target_gain)
    if do_plots:
        plot_weight_all(W_scaled, title=f"{label}: W_hh (after gain match)")
        plot_weight_all(A_after, title=f"{label}: A_open (after, target={target_gain})")

    return W_scaled, info


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
    n: int,
    value: float = 1.0,
    offset: int = 1,
    cyclic: bool = False,
    target_var: Optional[float] = None,
    target_gain: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Single shifted band:
      W[i, i+offset] = value   (wraps if cyclic, else clipped)
    Special case: offset=0, value=1.0 -> identity.

    Args
    ----
    n : int
    value : float
        The constant placed on the shifted diagonal.
    offset : int
        0 = main diagonal, +1 = superdiagonal, -1 = subdiagonal, etc.
    cyclic : bool
        If True, wrap indices modulo n.
    target_var : float or None
        If provided, rescale to match this entrywise variance (report only).
    target_gain : float or None
        If provided, rescale (last) so σ_max(W) == target_gain.
    verbose : bool
        Print pre/post scaling variances and gains.

    Returns
    -------
    W : (n,n) float32
    """
    W = np.zeros((n, n), dtype=np.float32)
    idx = np.arange(n)

    if cyclic:
        j = (idx + offset) % n
        W[idx, j] = value
    else:
        j = idx + offset
        mask = (j >= 0) & (j < n)
        W[idx[mask], j[mask]] = value

    # optional variance rescale (for reporting parity)
    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[shift] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[shift] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    # optional gain match (σ_max)
    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[shift] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")

    return W


def build_mexican_hat(
    n: int,
    sigma: Optional[float] = None,
    diag_offset: int = 0,
    cyclic: bool = False,
    target_var: Optional[float] = None,
    target_gain: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    1D Mexican-hat kernel placed as a Toeplitz/circulant matrix.
    You can 'shift' the kernel off the main diagonal by diag_offset.

    Kernel (even, centered at 0):
      k(m) = [1 - (d(m)^2 / sigma^2)] * exp( - d(m)^2 / (2 sigma^2) )
    where d(m) =
      - non-cyclic: abs(m)
      - cyclic    : min(|m|, n - |m|)  (circular distance)

    Non-cyclic (Toeplitz):
      W[i,j] = k( (i - j) - diag_offset ), clipped to bounds.

    Cyclic (circulant):
      W[i,j] = k( (i - j - diag_offset) mod n ), using the *circular distance*
               to compute k so it's still an even “bump”, just phase-shifted.

    Setting diag_offset=+1 moves the bump one above the diagonal.

    Returns
    -------
    W : (n,n) float32
    """
    if sigma is None:
        sigma = n / 10.0

    if cyclic:
        # signed offsets in range [-floor(n/2), ..., floor((n-1)/2)]
        offs = np.arange(n)
        # represent modulo-n offsets with a signed view:
        offs_signed = ((offs + n // 2) % n) - n // 2
        # circular distance for the even kernel
        d = np.abs(offs_signed).astype(np.float64)
        k = (1.0 - (d**2) / (sigma**2)) * np.exp(-(d**2) / (2.0 * sigma**2))
        k = k.astype(np.float32)

        # build circulant with a phase shift = diag_offset
        W = np.zeros((n, n), dtype=np.float32)
        i = np.arange(n)
        for j in range(n):
            # offset (i - j - diag_offset) mod n
            m = (i - j - diag_offset) % n
            W[i, j] = k[m]

    else:
        # Toeplitz with shift: use integer offsets m = (i - j) - diag_offset
        center = 0  # kernel centered at 0 offset
        # Precompute a reasonably wide kernel vector k[m] for m in [-(n-1)..(n-1)]
        m = np.arange(-(n - 1), (n - 1) + 1)
        d = np.abs(m).astype(np.float64)
        k = (1.0 - (d**2) / (sigma**2)) * np.exp(-(d**2) / (2.0 * sigma**2))
        k = k.astype(np.float32)

        W = np.zeros((n, n), dtype=np.float32)
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        off = (i - j) - diag_offset  # shift the bump
        # map off to index in k (centered at index (n-1))
        k_idx = off + (n - 1)
        mask = (k_idx >= 0) & (k_idx < k.shape[0])
        W[mask] = k[k_idx[mask]]

    # optional variance rescale
    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[MH] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[MH] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    # optional gain match
    if target_gain is not None:
        W, s0 = scale_to_gain_np(W, target_gain)
        if verbose:
            print(f"[MH] σ_max before {s0:.4f} → after {svd_sigma_max_np(W):.4f}")

    return W


def build_tridiag(
    n: int,
    diag: float = 1.0,
    off: float = -1.0,
    diag_offset: int = 0,
    cyclic: bool = False,
    target_var: Optional[float] = None,
    target_gain: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Tridiagonal band *shifted* by diag_offset:
      main band at (i, i+diag_offset) gets 'diag'
      neighbors at (i, i+diag_offset±1) get 'off'
    If cyclic=True, indices wrap around.

    Examples
    --------
    diag_offset=0  : standard tri-diagonal (main on the diagonal)
    diag_offset=+1 : main band on superdiagonal; neighbors at +2 and 0

    Returns
    -------
    W : (n,n) float32
    """
    W = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n)

    def set_band(delta: int, val: float):
        if cyclic:
            j = (i + delta) % n
            W[i, j] += val
        else:
            j = i + delta
            mask = (j >= 0) & (j < n)
            W[i[mask], j[mask]] += val

    # central band at offset=diag_offset
    set_band(diag_offset, diag)
    # neighbors
    set_band(diag_offset + 1, off)
    set_band(diag_offset - 1, off)

    # optional variance rescale
    if target_var is not None:
        var_emp = float(((W - W.mean()) ** 2).mean())
        if verbose:
            print(f"[tri] var before: {var_emp:.6f}")
        if var_emp > 0:
            W *= np.sqrt(target_var / var_emp)
        if verbose:
            print(f"[tri] var after : {float(((W - W.mean()) ** 2).mean()):.6f}")

    # optional gain match
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
