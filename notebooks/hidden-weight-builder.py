# Dumb script implementation of hidden-weight-builder.py for external use

import sys

sys.path.append("..")
import BuildHiddenWeights as bhw  # builders
import HiddenWeightHelpers as hw  # plots, stats, norms, saving, open-loop
from RNN_Class import *
import math, torch, torch.nn as nn

SAVE_DIR = "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean"
N = 100
NOISE_STD = 1e-2  # try 5e-3 to 1e-2
SEED = 42
HIDDEN_N = 100
THEO_VAR = 1.0 / (3.0 * HIDDEN_N)

# prepare input -- asymmetric Gaussian
loaded = torch.load("../data/Ns100_SeqN100/encodings/Ns100_SeqN100_asym1.pth.tar")
pred = 1
X = loaded["X_mini"]
Y = loaded["Target_mini"]
X = X[:, :-pred]
Y = Y[:, pred:]

# build model (sigmoid head, W_xh identity) -- used for baseline
model = ElmanRNN_pytorch_module_v2(HIDDEN_N, HIDDEN_N, HIDDEN_N)
model.act = nn.Sigmoid()

# --- Baseline (random) ---
# Baseline random initialization
W, W_xh, W_hy = hw.get_numpy_weights(model)
emp_var = float(W.var())
print(f"empirical var ≈ {emp_var:.6f}, theoretical var ≈ {THEO_VAR:.6f}")
print("INITIAL MATRICES")
hw.plot_weight_all(W, title=f"Baseline Whh (H={HIDDEN_N})")
hw.plot_weight_all(W_xh, title=f"Default Wxh (H={HIDDEN_N})")
hw.plot_weight_all(W_hy, title=f"Default Why (H={HIDDEN_N})")

# 1) Symmetric / skew plots
print("INITIAL SYMMETRY")
hw.plot_sym_asym(W, base_title="Random (raw)")

# 3) Save the baseline matrix
save_dir = "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/random/"
hw.save_matrix(
    W,
    save_dir,
    f"random_baseline",
    meta={
        "n": HIDDEN_N,
    },
)

# --- identity matrix ---
W = bhw.build_shift(n=HIDDEN_N, value=1.0, offset=0, cyclic=False)

# quick plots
print("INITIAL MATRICES")
hw.plot_weight_all(W, title=f"Identity Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
hw.plot_sym_asym(W, base_title="Identity (raw)")

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5
W_fro, _ = hw.normalize_by_fro(W, target_fro=target_fro)
print("NORMALIZATION COMPARISON")
hw.plot_weight_all(W_fro, title="Whh (Frobenius normalized)")

# save variants
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/shift-variants/identity"
)
hw.save_matrix(
    W_fro,
    f"{save_root}",
    f"identity",
    meta={
        "norm": "fro",
        "n": HIDDEN_N,
    },
)


def mix_ratio_tag(a: float) -> str:
    """0.90 -> 'sym0p90', 1.0 -> 'sym1p00'"""
    pct = int(round(a * 100))
    major = pct // 100
    minor = pct % 100
    return f"sym{major}p{minor:02d}"


def short_norm_tag(norm):
    return {
        "raw": "raw",
        "frobenius": "fro",
    }[norm]


# --- shift (acyclic) matrix ---
W = bhw.build_shift(n=HIDDEN_N, value=1.0, offset=1, cyclic=False)

# quick plots
print("INITIAL MATRICES")
hw.plot_weight_all(W, title=f"Shift (acyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
hw.plot_sym_asym(W, base_title="Shift (acyclic, raw)")

# symmetric / skew-symmetric split
S = 0.5 * (W + W.T)
A = 0.5 * (W - W.T)

# choose mixing ratios you want
mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

# where to save this structure's init
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/shift-variants/shift"
)

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

for alpha in mix_ratios:
    # alpha-blended hidden matrix: W_eff = αS + (1-α)A
    W_mix = alpha * S + (1 - alpha) * A

    # (optional) quick diagnostics
    print(f"λ = {alpha:.2f}")
    hw.plot_sym_asym(W_mix, base_title=f"shift (acyclic) α={alpha:.2f}")
    hw.plot_weight_all(W_mix, title=f"Whh (α={alpha:.2f}, raw)")

    # apply your three normalizations per alpha (and keep a raw copy)
    W_fro, _ = hw.normalize_by_fro(W_mix, target_fro=target_fro)

    # save under norm subfolders, with a alpha subdir (easy to browse)
    meta_base = {
        "structure": "shift (acyclic)",
        "hidden_n": int(HIDDEN_N),
        "lambda": float(alpha),
        "decomposition": "W=αS+(1-α)A",
        "notes": "normalized after α-mix",
    }
    sub = mix_ratio_tag(alpha)

    # frobenius
    hw.save_matrix(
        W_fro,
        f"{save_root}/{sub}",
        f"shift_{sub}",
        meta={**meta_base, "norm": "frobenius", "target_fro": float(target_fro)},
    )

# --- shift (cyclic) matrix ---
W = bhw.build_shift(n=HIDDEN_N, value=1.0, offset=1, cyclic=True)

# quick plots
print("INITIAL MATRICES")
# hw.plot_weight_all(W, title=f"Shift (cyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Shift (cyclic, raw)")

# symmetric / skew-symmetric split
S = 0.5 * (W + W.T)
A = 0.5 * (W - W.T)

# choose mixing ratios you want
mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

# where to save this structure's init
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/shift-variants/cyc-shift"
)

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

for alpha in mix_ratios:
    # alpha-blended hidden matrix: W_eff = αS + (1-α)A
    W_mix = alpha * S + (1 - alpha) * A

    # (optional) quick diagnostics
    print(f"λ = {alpha:.2f}")
    # hw.plot_sym_asym(W_mix, base_title=f"shift (cyclic) α={alpha:.2f}")
    # hw.plot_weight_all(W_mix, title=f"Whh (α={alpha:.2f}, raw)")

    # apply your three normalizations per alpha (and keep a raw copy)
    W_raw = W_mix
    W_fro, _ = hw.normalize_by_fro(W_mix, target_fro=target_fro)

    # save under norm subfolders, with a alpha subdir (easy to browse)
    meta_base = {
        "structure": "shift (cyclic)",
        "hidden_n": int(HIDDEN_N),
        "lambda": float(alpha),
        "decomposition": "W=αS+(1-α)A",
        "notes": "normalized after α-mix",
    }
    sub = mix_ratio_tag(alpha)

    # frobenius
    hw.save_matrix(
        W_fro,
        f"{save_root}/{sub}",
        f"cycshift_{sub}",
        meta={**meta_base, "norm": "frobenius", "target_fro": float(target_fro)},
    )

# --- mexican hat (centered, cyclic) matrix ---
W = bhw.build_mexican_hat(n=HIDDEN_N, sigma=None, diag_offset=0, cyclic=True)

# quick plots
print("INITIAL MATRICES")
# hw.plot_weight_all(W, title=f"Centered Mexican Hat (cyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Centered Mexican Hat (cyclic, raw)")

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

W_fro, _ = hw.normalize_by_fro(W, target_fro=target_fro)

print("NORMALIZATION COMPARISON")
# hw.plot_weight_all(W_fro, title="Whh (Frobenius normalized)")

# save variants
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/mh-variants/cyc-centered"
)
hw.save_matrix(
    W_fro,
    f"{save_root}",
    f"cyccentmh",
    meta={"norm": "fro", "hidden_n": HIDDEN_N},
)

# --- mexican hat (centered, acyclic) matrix ---
W = bhw.build_mexican_hat(n=HIDDEN_N, sigma=None, diag_offset=0, cyclic=False)

# quick plots
print("INITIAL MATRICES")
hw.plot_weight_all(W, title=f"Centered Mexican Hat (acyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Centered Mexican Hat (acyclic, raw)")

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

W_fro, _ = hw.normalize_by_fro(W, target_fro=target_fro)

print("NORMALIZATION COMPARISON")
# hw.plot_weight_all(W_fro, title="Whh (Frobenius normalized)")

# save variants
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/mh-variants/centered"
)
hw.save_matrix(
    W_fro, f"{save_root}", f"centmh", meta={"norm": "fro", "hidden_n": HIDDEN_N}
)

# --- mexican hat (shifted, cyclic) matrix ---
W = bhw.build_mexican_hat(n=HIDDEN_N, sigma=None, diag_offset=-19, cyclic=True)

# quick plots
print("INITIAL MATRICES")
# hw.plot_weight_all(W, title=f"Shifted Mexican Hat (cyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Shifted Mexican Hat (cyclic, raw)")

# symmetric / skew-symmetric split
S = 0.5 * (W + W.T)
A = 0.5 * (W - W.T)

# choose mixing ratios you want
mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

# where to save this structure's init
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/mh-variants/cyc-shifted"
)

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

for alpha in mix_ratios:
    # alpha-blended hidden matrix: W_eff = αS + (1-α)A
    W_mix = alpha * S + (1 - alpha) * A

    # (optional) quick diagnostics
    print(f"λ = {alpha:.2f}")
    # hw.plot_sym_asym(W_mix, base_title=f"Shifted MH (cyclic) α={alpha:.2f}")
    # hw.plot_weight_all(W_mix, title=f"Whh (α={alpha:.2f}, raw)")

    # apply your three normalizations per alpha (and keep a raw copy)
    W_raw = W_mix
    W_fro, _ = hw.normalize_by_fro(W_mix, target_fro=target_fro)

    # save under norm subfolders, with a alpha subdir (easy to browse)
    meta_base = {
        "structure": "Shifted MH (cyclic)",
        "hidden_n": int(HIDDEN_N),
        "lambda": float(alpha),
        "decomposition": "W=αS+(1-α)A",
        "notes": "normalized after α-mix",
    }
    sub = mix_ratio_tag(alpha)
    # frobenius
    hw.save_matrix(
        W_fro,
        f"{save_root}/{sub}",
        f"cycshiftmh_{sub}",
        meta={**meta_base, "norm": "frobenius", "target_fro": float(target_fro)},
    )

# --- mexican hat (shifted, acyclic) matrix ---
W = bhw.build_mexican_hat(n=HIDDEN_N, sigma=None, diag_offset=-19, cyclic=False)

# quick plots
print("INITIAL MATRICES")
# hw.plot_weight_all(W, title=f"Shifted Mexican Hat (acyclic) Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Shifted Mexican Hat (acyclic, raw)")

# symmetric / skew-symmetric split
S = 0.5 * (W + W.T)
A = 0.5 * (W - W.T)

# choose mixing ratios you want
mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

# where to save this structure's init
save_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/clean/mh-variants/shifted"
)

# normalization comparisons
target_fro = (W.size * THEO_VAR) ** 0.5

for alpha in mix_ratios:
    # alpha-blended hidden matrix: W_eff = αS + (1-α)A
    W_mix = alpha * S + (1 - alpha) * A

    # (optional) quick diagnostics
    print(f"λ = {alpha:.2f}")
    # hw.plot_sym_asym(W_mix, base_title=f"Shifted MH (acyclic) α={alpha:.2f}")
    # hw.plot_weight_all(W_mix, title=f"Whh (α={alpha:.2f}, raw)")

    # apply your three normalizations per alpha (and keep a raw copy)
    W_raw = W_mix
    W_fro, _ = hw.normalize_by_fro(W_mix, target_fro=target_fro)

    # save under norm subfolders, with a alpha subdir (easy to browse)
    meta_base = {
        "structure": "Shifted MH (acyclic)",
        "hidden_n": int(HIDDEN_N),
        "lambda": float(alpha),
        "decomposition": "W=αS+(1-α)A",
        "notes": "normalized after α-mix",
    }
    sub = mix_ratio_tag(alpha)

    # frobenius
    hw.save_matrix(
        W_fro,
        f"{save_root}/{sub}",
        f"shiftmh_{sub}",
        meta={**meta_base, "norm": "frobenius", "target_fro": float(target_fro)},
    )
