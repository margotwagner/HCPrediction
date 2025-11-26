import sys

sys.path.append("..")
import BuildHiddenWeights as bhw  # builders
import HiddenWeightHelpers as hw  # plots, stats, norms, saving, open-loop
from ElmanRNN import *
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def mix_ratio_tag(a: float) -> str:
    """0.90 -> 'sym0p90', 1.0 -> 'sym1p00'"""
    pct = int(round(a * 100))
    major = pct // 100
    minor = pct % 100
    return f"sym{major}p{minor:02d}"


HIDDEN_N = 100

# ------------------------------------------------
# Vanilla PyTorch random baseline (dense + circulant)
# ------------------------------------------------

H = HIDDEN_N  # assuming you already have this defined
seeds = [0, 1, 2, 3, 4]  # or whatever seeds you want

dense_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/dense/vanilla_pytorch"
circ_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/11252025/circulant/vanilla_pytorch"
)

for seed in seeds:
    print(f"\n=== Vanilla PyTorch random, seed={seed} ===")
    torch.manual_seed(seed)

    # Build model exactly as in your training setup
    model = ElmanRNN_pytorch_module_v2(
        input_dim=H,
        hidden_dim=H,
        output_dim=H,
        rnn_act="tanh",  # or whatever you actually use in Main_clean
    )

    # Extract the dense hidden weight: this is PyTorch's default init
    W_dense = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
    emp_var = float(W_dense.var())
    print(f"[vanilla] empirical var(Whh) ≈ {emp_var:.6f}")

    # Optional: sanity plots
    # hw.plot_weight_all(W_dense, title=f"Vanilla PyTorch Whh (H={H}, seed={seed})")
    # hw.plot_sym_asym(W_dense, base_title=f\"Vanilla PyTorch (raw, seed={seed})\")

    # Check circulant-ness (it should be False for a generic random matrix)
    ok = hw.is_circulant(W_dense, tol=1e-7)
    print(f"[vanilla] circulant? {ok} (tol=1e-7)")

    # ---------- Save dense version ----------
    save_dir_dense = os.path.join(dense_root, f"seed{seed:03d}")
    fname_dense = "Whh"
    meta_dense = {
        "backend": "dense",
        "family": "vanilla_pytorch",
        "hidden_n": int(H),
        "seed": int(seed),
        "norm": "raw",  # Frobenius normalization happens in Main_clean.py
        "source": "nn.RNN.weight_hh_l0_default_init",
    }
    hw.save_matrix(W_dense, save_dir_dense, fname_dense, meta=meta_dense)

    # ---------- Save circulant pseudo-random version (row0) ----------
    row0 = hw.extract_first_row(W_dense)
    save_dir_circ = os.path.join(circ_root, f"seed{seed:03d}")
    fname_row0 = "row0"
    hw.extract_and_optionally_save_first_row(
        W_dense,
        save_dir=save_dir_circ,
        name=fname_row0,
    )

import os
import numpy as np
import HiddenWeightHelpers as hw

# ------------------------------------------------
# Random elliptic family (dense + circulant)
# ------------------------------------------------

H = HIDDEN_N  # assuming you already have this global
alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
seeds = [0, 1, 2, 3, 4]  # or however many seeds you want

dense_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/dense/random_elliptic"
circ_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/11252025/circulant/random_elliptic"
)

for seed in seeds:
    print(f"\n==============================")
    print(f" Random elliptic family, seed={seed}")
    print(f"==============================")

    rng = np.random.default_rng(seed)

    # 1) Base Gaussian random matrix G ~ N(0,1)
    G = rng.standard_normal(size=(H, H)).astype(np.float32)
    emp_var = float(G.var())
    print(f"[random-elliptic] base G var ≈ {emp_var:.6f}")

    # 2) Symmetric / skew-symmetric decomposition
    S = 0.5 * (G + G.T)
    A = 0.5 * (G - G.T)

    for alpha in alpha_grid:
        # Make a friendly string tag like "0p00", "0p25", etc.
        sub = mix_ratio_tag(alpha)  # you already have this helper

        print(f"\n--- seed={seed}, alpha={alpha:.2f} (sub='{sub}') ---")
        # 3) Construct W_alpha = α S + (1-α) A
        W_alpha = alpha * S + (1.0 - alpha) * A

        # Optional: symmetry/structure diagnostics
        # hw.plot_sym_asym(W_alpha, base_title=f\"Random elliptic α={alpha:.2f}, seed={seed}\")
        ok_circ = hw.is_circulant(W_alpha, tol=1e-7)
        print(f"[random-elliptic] circulant? {ok_circ} (tol=1e-7)")  # generally False

        # ---------- Save dense Whh ----------
        save_dir_dense = os.path.join(
            dense_root,
            f"seed{seed:03d}",
            f"alpha{sub}",
        )
        fname_dense = "Whh"  # keep filenames simple; path encodes details

        meta_dense = {
            "backend": "dense",
            "family": "random_elliptic",
            "hidden_n": int(H),
            "seed": int(seed),
            "alpha": float(alpha),
            "decomposition": "W=αS+(1-α)A",
            "norm": "raw",  # Frobenius normalization happens in Main_clean.py
        }

        hw.save_matrix(W_alpha, save_dir_dense, fname_dense, meta=meta_dense)

        # ---------- Save circulant pseudo-random version (row0) ----------
        row0 = hw.extract_first_row(W_alpha)
        save_dir_circ = os.path.join(
            circ_root,
            f"seed{seed:03d}",
            f"alpha{sub}",
        )
        fname_row0 = "row0"

        hw.extract_and_optionally_save_first_row(
            W_alpha,
            save_dir=save_dir_circ,
            name=fname_row0,
        )

# --- identity matrix ---
W = bhw.build_shift(n=HIDDEN_N, value=1.0, offset=0, cyclic=True)

# quick plots
print("INITIAL MATRICES")
# hw.plot_weight_all(W, title=f"Identity Whh (H={HIDDEN_N})")

# symmetry
print("INITIAL SYMMETRY")
# hw.plot_sym_asym(W, base_title="Identity (raw)")

# verify circulant
ok = hw.is_circulant(W, tol=1e-7)
print("[builder] circulant? {} (tol=1e-7)".format(ok))
assert ok, "Expected a circulant matrix"

# save variants
dense_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/dense/identity"
circ_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/circulant/identity"

hw.save_matrix(
    W,
    dense_root,
    "Whh",
    meta={
        "backend": "dense",
        "family": "identity",
        "hidden_n": HIDDEN_N,
        "norm": "raw",
    },
)

# extract first row
row0 = hw.extract_first_row(W)

hw.extract_and_optionally_save_first_row(
    W,
    save_dir=circ_root,
    name="row0",
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
dense_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/dense/shift/cyclic"
circ_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/circulant/shift"

for alpha in mix_ratios:
    # alpha-blended hidden matrix: W_eff = αS + (1-α)A
    W_mix = alpha * S + (1 - alpha) * A

    # (optional) quick diagnostics
    print(f"λ = {alpha:.2f}")
    # hw.plot_sym_asym(W_mix, base_title=f"shift (cyclic) α={alpha:.2f}")
    # hw.plot_weight_all(W_mix, title=f"Whh (α={alpha:.2f}, raw)")

    # verify circulant
    ok = hw.is_circulant(W_mix, tol=1e-7)
    print("[builder] circulant? {} (tol=1e-7)".format(ok))

    sub = mix_ratio_tag(alpha)

    # save under norm subfolders, with a alpha subdir (easy to browse)
    meta_dense = {
        "backend": "dense",
        "family": "shift",
        "geometry": "cyclic",
        "hidden_n": int(HIDDEN_N),
        "alpha": float(alpha),
        "decomposition": "W=αS+(1-α)A",
        "norm": "raw",  # Frobenius normalization happens in Main_clean.py
    }

    # ---- save dense Whh ----
    save_dir_dense = f"{dense_root}/alpha{sub}"
    fname_dense = "Whh"
    hw.save_matrix(
        W_mix,
        save_dir_dense,
        fname_dense,
        meta=meta_dense,
    )

    # ---- extract and save row0 for circulant version ----
    row0 = hw.extract_first_row(W_mix)
    save_dir_circ = f"{circ_root}/alpha{sub}"
    fname_row0 = "row0"

    hw.extract_and_optionally_save_first_row(
        W_mix,
        save_dir=save_dir_circ,
        name=fname_row0,
    )

    # quick visualization if you want
    plt.imshow(row0.reshape(1, HIDDEN_N))
    plt.show()
    print(row0)

import os
import numpy as np
import matplotlib.pyplot as plt

import BuildHiddenWeights as bhw  # has build_mexican_hat_dog
import HiddenWeightHelpers as hw  # extract_first_row, save helpers

# -------------------------------------------------------------------
# DoG parameter grid: (shortname, params)
# -------------------------------------------------------------------
dog_configs = [
    ("dog1", dict(sigma_e=4.0, sigma_i=12.0, a=1.0, b=0.25)),  # stable bump
    ("dog2", dict(sigma_e=5.0, sigma_i=15.0, a=1.0, b=0.30)),  # medium MH
    ("dog3", dict(sigma_e=6.0, sigma_i=20.0, a=1.0, b=0.35)),  # stronger surround
]

# Offsets: centered bump vs mild traveling wave
offset_configs = [
    ("k0", 0),  # diag_offset = 0 (centered kernel)
    ("k5", -5),  # diag_offset = -5 (shifted kernel, slow drift)
]

# Alpha grid for symmetry mixing (used for k5)
mix_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

# Base dirs for saving (new scheme)
dense_root = "../data/Ns100_SeqN100/hidden-weight-inits/11252025/dense/mexican_hat/dog"
circ_root = (
    "../data/Ns100_SeqN100/hidden-weight-inits/11252025/circulant/mexican_hat/dog"
)

os.makedirs(dense_root, exist_ok=True)
os.makedirs(circ_root, exist_ok=True)

for dog_name, params in dog_configs:
    for offset_tag, diag_offset in offset_configs:
        print(f"\n=== Building {dog_name}_{offset_tag} (diag_offset={diag_offset}) ===")

        # 1. Build DoG-based Mexican hat (cyclic = True for ring)
        W = bhw.build_mexican_hat_dog(
            n=HIDDEN_N,
            diag_offset=diag_offset,
            cyclic=True,
            **params,
        )

        # 2. Sanity check: is circulant?
        ok = hw.is_circulant(W, tol=1e-7)
        print("[builder] circulant? {} (tol=1e-7)".format(ok))

        # 3. Optional quick visualization (raw kernel)
        # hw.plot_weight_all(W, title=f"{dog_name}_{offset_tag} (DoG, raw)")

        # Common metadata fields
        base_meta = {
            "structure": "DoG Mexican Hat (cyclic)",
            "family": "mexican_hat_dog",
            "hidden_n": int(HIDDEN_N),
            "sigma_e": float(params["sigma_e"]),
            "sigma_i": float(params["sigma_i"]),
            "a": float(params["a"]),
            "b": float(params["b"]),
            "diag_offset": int(diag_offset),
            "norm": "raw",  # Frobenius normalization happens in Main_clean.py
        }

        # ------------------------------------------------------------------
        # Case 1: centered (k0) – just save the raw kernel (no α sweep)
        # ------------------------------------------------------------------
        if offset_tag == "k0":
            save_dir_dense = os.path.join(dense_root, dog_name, offset_tag)
            os.makedirs(save_dir_dense, exist_ok=True)
            fname_dense = "Whh"

            hw.save_matrix(W, save_dir_dense, fname_dense, meta=base_meta)

            # row0 for circulant
            row0 = hw.extract_first_row(W)
            save_dir_circ = os.path.join(circ_root, dog_name, offset_tag)
            os.makedirs(save_dir_circ, exist_ok=True)
            hw.extract_and_optionally_save_first_row(
                W,
                save_dir=save_dir_circ,
                name="row0",
            )

        # ------------------------------------------------------------------
        # Case 2: shifted (k5) – build α-family via S/A mixing
        # ------------------------------------------------------------------
        elif offset_tag == "k5":
            # symmetric / skew-symmetric split of this shifted DoG kernel
            S = 0.5 * (W + W.T)
            A = 0.5 * (W - W.T)

            for alpha in mix_ratios:
                sub = mix_ratio_tag(alpha)  # e.g., "0p00", "0p25", ...

                print(f"  λ = {alpha:.2f} (sub='{sub}')")

                # α-blended kernel: W_eff = αS + (1-α)A
                W_mix = alpha * S + (1.0 - alpha) * A

                # Optional diagnostics
                ok_mix = hw.is_circulant(W_mix, tol=1e-7)
                print("[builder]   circulant (W_mix)? {} (tol=1e-7)".format(ok_mix))

                meta = {
                    **base_meta,
                    "alpha": float(alpha),
                    "decomposition": "W=αS+(1-α)A",
                }

                # Dense save: Whh
                save_dir_dense = os.path.join(
                    dense_root,
                    dog_name,
                    offset_tag,
                    f"alpha{sub}",
                )
                os.makedirs(save_dir_dense, exist_ok=True)
                fname_dense = "Whh"

                hw.save_matrix(W_mix, save_dir_dense, fname_dense, meta=meta)

                # Circulant: row0 from W_mix
                row0 = hw.extract_first_row(W_mix)
                save_dir_circ = os.path.join(
                    circ_root,
                    dog_name,
                    offset_tag,
                    f"alpha{sub}",
                )
                os.makedirs(save_dir_circ, exist_ok=True)
                hw.extract_and_optionally_save_first_row(
                    W_mix,
                    save_dir=save_dir_circ,
                    name="row0",
                )

                # Optional: visualize row0
                # plt.figure()
                # plt.imshow(row0.reshape(1, -1), aspect="auto")
                # plt.title(f"{dog_name}_{offset_tag} α={alpha:.2f} row0")
                # plt.colorbar()
                # plt.show()

        else:
            raise ValueError(f"Unexpected offset_tag={offset_tag!r}")
