# =========================
# BuildHiddenWeights.py  — builders only (NumPy, Py<=3.10 safe)
# =========================
from typing import Optional
import numpy as np
from HiddenWeightHelpers import add_noise_preserve_structure

# All builders return NumPy arrays (float32).
# Any normalization/plots/saving are in HiddenWeightHelpers.py.


def build_he(n_hidden: int) -> np.ndarray:
    """
    Xavier-for-tanh style baseline: std = 1/sqrt(n).
    Returns (n, n) float32.
    """
    n = n_hidden
    std = (1.0 / n) ** 0.5
    W = np.random.randn(n, n).astype(np.float32) * std
    return W


def build_xavier(
    n_in: int, n_out: int, gain: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """
    Xavier/Glorot normal: std = gain * sqrt(2 / (fan_in + fan_out)).
    Returns (n_out, n_in) float32.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    std = gain * np.sqrt(2.0 / (n_in + n_out))
    W = rng.normal(0.0, std, size=(n_out, n_in)).astype(np.float32)
    return W


def build_shift(
    n: int, value: float = 1.0, offset: int = 1, cyclic: bool = False
) -> np.ndarray:
    """
    Single shifted band:
      W[i, i+offset] = value   (wraps if cyclic, else clipped)
    Special: offset=0 & value=1.0 -> identity.
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
    return W


def build_tridiag(
    n: int,
    diag: float = 1.0,
    off: float = -1.0,
    diag_offset: int = 0,
    cyclic: bool = False,
) -> np.ndarray:
    """
    Tridiagonal band *shifted* by diag_offset:
      main band at (i, i+diag_offset) gets 'diag'
      neighbors at (i, i+diag_offset±1) get 'off'
    If cyclic=True, indices wrap.
    """
    W = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n)

    def set_band(delta: int, val: float) -> None:
        if cyclic:
            j = (i + delta) % n
            W[i, j] += val
        else:
            j = i + delta
            mask = (j >= 0) & (j < n)
            W[i[mask], j[mask]] += val

    set_band(diag_offset, diag)
    set_band(diag_offset + 1, off)
    set_band(diag_offset - 1, off)
    return W


def build_mexican_hat(
    n: int,
    sigma: Optional[float] = None,
    diag_offset: int = 0,
    cyclic: bool = False,
) -> np.ndarray:
    """
    1D Mexican-hat kernel as Toeplitz (non-cyclic) or circulant (cyclic).
    Shift the 'bump' off the main diagonal with diag_offset.
    """
    if sigma is None:
        sigma = n / 10.0

    if cyclic:
        offs = np.arange(n)
        offs_signed = ((offs + n // 2) % n) - n // 2
        d = np.abs(offs_signed).astype(np.float64)
        k = (1.0 - (d**2) / (sigma**2)) * np.exp(-(d**2) / (2.0 * sigma**2))
        k = k.astype(np.float32)

        W = np.zeros((n, n), dtype=np.float32)
        i = np.arange(n)
        for j in range(n):
            m = (i - j - diag_offset) % n
            W[i, j] = k[m]
        return W

    # Toeplitz with shift
    m = np.arange(-(n - 1), (n - 1) + 1)
    d = np.abs(m).astype(np.float64)
    k = (1.0 - (d**2) / (sigma**2)) * np.exp(-(d**2) / (2.0 * sigma**2))
    k = k.astype(np.float32)

    W = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    off = (i - j) - diag_offset
    k_idx = off + (n - 1)
    mask = (k_idx >= 0) & (k_idx < k.shape[0])
    W[mask] = k[k_idx[mask]]
    return W


def build_orthogonal(
    n: int, scale: float = 1.0, seed: Optional[int] = None, ensure_det_pos: bool = False
) -> np.ndarray:
    """
    Random orthogonal (QR). Returns (n,n) with columns orthonormal, scaled by 'scale'.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n), loc=0.0, scale=1.0).astype(np.float64)
    Q, R = np.linalg.qr(A, mode="reduced")
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    if ensure_det_pos and np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return (scale * Q).astype(np.float32)


def build_with_noise(
    builder_fn,  # e.g., build_shift, build_mexican_hat, ...
    *bargs,  # args for that builder
    noise_std: float = 1e-2,
    mode: str = "support_only",
    sym_mode: str = "none",
    sym_mix: float = 0.2,
    seed: int = 0,
    **bkwargs,
):
    """
    Convenience: build a motif, add structured noise, renorm by Frobenius.
    Example:
      W = build_with_noise(build_shift, n=100, value=1.0, offset=1, cyclic=False,
                           noise_std=1e-2, mode="offdiag", sym_mode="none")
    """
    W0 = builder_fn(*bargs, **bkwargs)
    Wn, info = add_noise_preserve_structure(
        W0,
        noise_std=noise_std,
        mode=mode,
        sym_mode=sym_mode,
        sym_mix=sym_mix,
        seed=seed,
    )
    return Wn, info
