#!/usr/bin/env python3
"""
Sweep noise levels with a guaranteed initial mixing ratio (alpha0).

Example:
  python sweep_noise_hidden.py \
    --variant identity \
    --hidden-n 100 \
    --alpha0 0.50 \
    --save-root ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/noisy/identity/frobenius \
    --base-name identity_n100_fro \
    --noise-stds 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2 \
    --seed 42
"""

import os
import argparse
from pathlib import Path
import numpy as np

# Builders & helpers from your project:
from BuildHiddenWeights import (
    build_shift,
    build_tridiag,
    build_mexican_hat,
    build_he,
    build_orthogonal,
)
from HiddenWeightHelpers import (
    add_noise_and_fro_norm,
    save_matrix,
    summarize_matrix,
    mix_sym_skew_to_alpha,
    compute_alpha_mixing_ratio,
)


# ---------- small internal helpers --------------------------------------------
def noise_tag(std: float) -> str:
    return "nstd" + f"{std:.3f}".replace(".", "p")


def alpha_tag(a: float) -> str:
    # e.g., 0.50 -> "sym0p50"
    return "sym" + f"{a:.2f}".replace(".", "p")


def variant_to_builder(variant: str):
    v = variant.lower()
    if v in {"identity", "id"}:
        # identity as a shift with offset=0
        return lambda n: build_shift(n, value=1.0, offset=0, cyclic=False).astype(
            np.float32
        )
    if v in {"shift", "shift-cyc", "shifted-cyc", "shiftcyc"}:
        cyclic = "cyc" in v
        return lambda n: build_shift(n, value=1.0, offset=1, cyclic=cyclic).astype(
            np.float32
        )
    if v in {"tridiag", "tri", "tri-cyc"}:
        cyclic = "cyc" in v
        return lambda n: build_tridiag(
            n, diag=1.0, off=-1.0, diag_offset=0, cyclic=cyclic
        ).astype(np.float32)
    if v in {"mexican-hat", "mh", "mh-cyc", "shifted-cyc-mh", "shiftcycmh"}:
        cyclic = "cyc" in v
        return lambda n: build_mexican_hat(
            n, sigma=None, diag_offset=0, cyclic=cyclic
        ).astype(np.float32)


def default_noise_grid():
    return [0.000, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]


# ---------- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        type=str,
        required=True,
        help="identity | shift | shift-cyc | tridiag | mexican-hat | mh-cyc | he | orth",
    )
    ap.add_argument("--hidden-n", type=int, required=True)
    ap.add_argument(
        "--alpha0",
        type=float,
        required=True,
        help="Target initial mixing ratio in [0,1]",
    )
    ap.add_argument(
        "--save-root",
        type=str,
        required=True,
        help="Root folder under ./data/.../ElmanRNN/noisy/<variant>/<norm_dir>",
    )
    ap.add_argument(
        "--base-name",
        type=str,
        required=True,
        help="Base filename stem (e.g., identity_n100_fro). We append _<alphaTag>_<noiseTag>",
    )
    ap.add_argument(
        "--noise-stds",
        type=str,
        default="",
        help="Comma list (e.g., 0.001,0.01,0.1). If empty, uses default grid.",
    )
    ap.add_argument(
        "--target-fro",
        type=float,
        default=None,
        help="If None, preserve base Fro; else renormalize to this Fro after mixing/noise.",
    )
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # Build base W
    builder = variant_to_builder(args.variant)
    W0 = builder(args.hidden_n)
    fro0 = float(np.linalg.norm(W0))
    summarize_matrix(W0, "base W0", target_gain=None)

    # Force alpha0 on the base (and keep Fro scale comparable)
    target_fro_for_mix = fro0 if args.target_fro is None else args.target_fro
    W0_forced = mix_sym_skew_to_alpha(
        W0, alpha=args.alpha0, renorm_to=target_fro_for_mix
    )
    alpha0_measured = compute_alpha_mixing_ratio(W0_forced)

    # Report (useful sanity check)
    print(
        f"[INFO] requested alpha0={args.alpha0:.3f}  measured alpha0≈{alpha0_measured:.3f} "
        f"(||W0_forced||_F={np.linalg.norm(W0_forced):.4f})"
    )

    # Parse noise list
    grid = (
        default_noise_grid()
        if not args.noise_stds.strip()
        else [float(x) for x in args.noise_stds.split(",")]
    )

    # Directory layout: .../<variant>/<norm_dir>/**sym*/**nstd*/<files>
    a_tag = alpha_tag(args.alpha0)
    for std in grid:
        n_tag = noise_tag(std)
        save_dir = os.path.join(args.save_root, a_tag, n_tag)

        Wn, info = add_noise_and_fro_norm(
            W0_forced,
            noise_std=std,
            target_fro=(
                args.target_fro if args.target_fro is not None else target_fro_for_mix
            ),
            seed=args.seed,
            scale_by_sqrtN=True,
        )

        fname = f"{args.base_name}_{a_tag}_{n_tag}"
        meta = {
            "variant": args.variant,
            "hidden_n": args.hidden_n,
            "alpha0_req": args.alpha0,
            "alpha0_measured": alpha0_measured,
            "noise_std": std,
            "target_fro": (
                args.target_fro if args.target_fro is not None else target_fro_for_mix
            ),
            "seed": args.seed,
            "notes": "gaussian noise scaled by 1/sqrt(H); Fro renormed after noise; alpha0 forced before noise",
        }
        save_matrix(Wn, save_dir, fname, meta=meta)


if __name__ == "__main__":
    main()
