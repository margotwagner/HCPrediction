#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
- Matplotlib-only figure suite for your Elman RNN project.
- Works both as a CLI script and as an importable module for notebooks.
- Python 3.6.7-compatible; no torch.* usage.

Inputs (flexible):
- Aggregated CSVs:
    run_level.csv           (per-run rows)
    condition_summary.csv   (per-condition rows)
- Optional per-run offline & eval CSVs discovered by glob if not pre-merged.

Outputs:
- PNGs saved to --figdir (default: ./figs)

Usage (auto all):
    python make_figures.py --agg_dir ./runs_agg --figdir ./figs --fontsize 12

Notebook usage:
    import make_figures as mf
    rl, cs, off = mf.load_all(agg_dir="./runs_agg")
    mf.fig_convergence_speed(cs, savepath="figs/fig1_convergence.png", fontsize=14)
    ...
"""

from __future__ import print_function
import os
import sys
import glob
import json
import math
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path

matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt

# ---------------------------
# Data loading & safe helpers
# ---------------------------


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _read_csv_safe(path):
    if path and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("[WARN] Failed to read CSV:", path, "error:", e)
            return None
    return None


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _col(df, name, default=None):
    """Get column by name if exists, else default."""
    if df is None or name not in df.columns:
        return None if default is None else default
    return df[name]


def _has_cols(df, cols):
    return df is not None and all(c in df.columns for c in cols)


def _filter_df(df, condition_regex=None, run_regex=None):
    if df is None:
        return df
    out = df.copy()
    if condition_regex and "condition_id" in out.columns:
        out = out[
            out["condition_id"].astype(str).str.contains(condition_regex, regex=True)
        ]
    if run_regex and "run_id" in out.columns:
        out = out[out["run_id"].astype(str).str.contains(run_regex, regex=True)]
    return out


def load_all(
    agg_dir="./runs_agg",
    run_level_csv=None,
    condition_csv=None,
    offline_csv=None,
    eval_csv=None,
):
    """
    Load aggregated CSVs, with fallbacks if paths are not provided.
    Returns: (run_level_df, condition_df, offline_df)
    """
    if run_level_csv is None:
        run_level_csv = _first_existing(
            [os.path.join(agg_dir, "run_level.csv"), "run_level.csv"]
        )
    if condition_csv is None:
        condition_csv = _first_existing(
            [os.path.join(agg_dir, "condition_summary.csv"), "condition_summary.csv"]
        )
    if offline_csv is None:
        offline_csv = _first_existing(
            [os.path.join(agg_dir, "offline_all.csv"), "offline_all.csv"]
        )

    rl = _read_csv_safe(run_level_csv)
    cs = _read_csv_safe(condition_csv)
    off = _read_csv_safe(offline_csv)

    # If offline_all missing, try discover per-run offline CSVs
    if off is None:
        cand = glob.glob(os.path.join(agg_dir, "**", "*offline*.csv"), recursive=True)
        if cand:
            frames = []
            for c in cand:
                df = _read_csv_safe(c)
                if df is not None:
                    frames.append(df)
            if frames:
                try:
                    off = pd.concat(frames, sort=False, ignore_index=True)
                except Exception:
                    off = None

    # If evaluation-level CSV exists and you want to merge/use it later, load it similarly.
    # (Most of the time, eval metrics are already merged into run_level.csv by your aggregator.)
    ev = _read_csv_safe(eval_csv) if eval_csv else None

    return rl, cs, off, ev


def _infer_condition_id_from_root(root):
    # Try to make a readable condition_id from the folder path
    # e.g., ".../shifted-cyc/frobenius/sym0p90/shiftmh_n100_fro" -> "shifted-cyc/frobenius/sym0p90/shiftmh_n100_fro"
    parts = []
    cur = os.path.normpath(root).split(os.sep)
    # take last 3-4 components; adjust to taste
    take = min(4, len(cur))
    parts = cur[-take:]
    return "/".join(parts)


def load_many_conditions(condition_roots):
    """
    Read aggregated CSVs for each condition root and concatenate:
      - expects each root to contain run_level.csv and condition_summary.csv
      - also globs per-run offline CSVs like *_offline_metrics.csv under each root
    Returns: (run_level_df, condition_df, offline_df, eval_df_or_None)
    """
    rl_list, cs_list, off_list = [], [], []

    for root in condition_roots:
        # per-condition run/summary tables
        rl = _read_csv_safe(os.path.join(root, "run_level.csv"))
        cs = _read_csv_safe(os.path.join(root, "condition_summary.csv"))
        # offline per-run CSV(s)
        cand = glob.glob(os.path.join(root, "**", "*offline*.csv"), recursive=True)

        cond_id = _infer_condition_id_from_root(root)

        if rl is not None:
            if "condition_id" not in rl.columns:
                rl["condition_id"] = cond_id
            rl_list.append(rl)
        if cs is not None:
            if "condition_id" not in cs.columns:
                cs["condition_id"] = cond_id
            cs_list.append(cs)
        for c in cand:
            df = _read_csv_safe(c)
            if df is not None:
                if "condition_id" not in df.columns:
                    df["condition_id"] = cond_id
                off_list.append(df)

    rl = pd.concat(rl_list, sort=False, ignore_index=True) if rl_list else None
    cs = pd.concat(cs_list, sort=False, ignore_index=True) if cs_list else None
    off = pd.concat(off_list, sort=False, ignore_index=True) if off_list else None
    return rl, cs, off, None


def _cs_to_wide(cs, value_col="mean"):
    """
    Convert tall condition_summary (columns: ['metric','mean','std',...,'condition_id'])
    into wide format with one row per condition and one column per metric.
    """
    if cs is None or not {"metric", value_col}.issubset(set(cs.columns)):
        return None
    if "condition_id" not in cs.columns:
        cs = cs.copy()
        cs["condition_id"] = "cond"  # fallback if not provided
    wide = cs.pivot_table(
        index="condition_id", columns="metric", values=value_col, aggfunc="first"
    )
    wide = wide.reset_index()
    # Ensure column names are plain strings (matplotlib friendly)
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _alpha_from_condition_id(cid):
    """
    Extract alpha from condition_id strings that contain 'symXpYY' (e.g., 'sym0p90').
    Returns float in [0,1] or None if not found.
    """
    if cid is None:
        return None
    m = re.search(r"sym(\d+)p(\d{2})", str(cid))
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    return major + minor / 100.0


# ---------------------------
# Matplotlib style utilities
# ---------------------------


def _style(ax, fontsize=12, title=None, xlabel=None, ylabel=None, legend=False):
    ax.tick_params(axis="both", labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if legend:
        leg = ax.legend(fontsize=max(8, fontsize - 2), frameon=False)
        if leg is not None:
            for l in leg.get_lines():
                l.set_linewidth(2.0)


def _savefig(fig, path):
    _ensure_dir(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("[SAVE]", path)


# ---------------------------
# Figure 1 – Convergence speed
# ---------------------------


def _gather_series(condition_roots, pattern):
    """Collect per-run timeseries matching a filename pattern under each condition root.
    Returns dict: {cond_id: [DataFrame_per_run_with_epoch_index, ...]}."""
    import glob, os
    import pandas as pd

    out = {}
    for root in condition_roots:
        cond_id = _infer_condition_id_from_root(root)
        paths = glob.glob(os.path.join(root, "run_*", pattern))
        runs = []
        for p in sorted(paths):
            df = _read_csv_safe(p)
            if df is None or "epoch" not in df.columns:
                continue
            df = df.sort_values("epoch").set_index("epoch")
            runs.append(df)
        if runs:
            out[cond_id] = runs
    return out


def _mean_sem_align(dfs, col):
    """Align on union of epochs; return (epochs, mean, sem)."""
    import pandas as pd, numpy as np

    if not dfs:
        return None, None, None
    # outer-join on epoch
    aligned = pd.concat([d[col] for d in dfs if col in d.columns], axis=1)
    aligned = aligned.sort_index()
    vals = aligned.values  # [T, R]
    mean = np.nanmean(vals, axis=1)
    sem = np.nanstd(vals, axis=1, ddof=1) / np.sqrt(
        np.sum(np.isfinite(vals), axis=1).clip(min=1)
    )
    return aligned.index.values, mean, sem


def fig1_training_dynamics(condition_roots, savepath=None, fontsize=12):
    """
    Fig 1 (2x2):
      A: loss vs epoch
      B: grad L2 (post) vs epoch
      C: spectral radius vs epoch
      D: Frobenius norm vs epoch
    """
    L = _gather_series(condition_roots, "*_loss_curve.csv")
    G = _gather_series(condition_roots, "*_grad_curve.csv")
    W = _gather_series(condition_roots, "*_wstruct_curve.csv")
    if not (L or G or W):
        print("[SKIP] fig1_training_dynamics: no per-run series found")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    (axA, axB), (axC, axD) = axs

    # Panel A: loss
    for cond, runs in L.items():
        x, m, s = _mean_sem_align(runs, "loss")
        if x is None:
            continue
        axA.plot(x, m, lw=2, label=cond)
        axA.fill_between(x, m - s, m + s, alpha=0.2)
    _style(
        axA, fontsize, title="Loss vs epoch", xlabel="epoch", ylabel="loss", legend=True
    )

    # Panel B: grad L2 (post)
    for cond, runs in G.items():
        x, m, s = _mean_sem_align(runs, "grad_L2_post")
        if x is None:
            continue
        axB.plot(x, m, lw=2, label=cond)
        axB.fill_between(x, m - s, m + s, alpha=0.2)
    _style(
        axB,
        fontsize,
        title="Grad L2 (post) vs epoch",
        xlabel="epoch",
        ylabel="‖∇‖₂",
        legend=True,
    )

    # Panel C: spectral radius
    for cond, runs in W.items():
        x, m, s = _mean_sem_align(runs, "spectral_radius")
        if x is None:
            continue
        axC.plot(x, m, lw=2, label=cond)
        axC.fill_between(x, m - s, m + s, alpha=0.15)
    _style(
        axC,
        fontsize,
        title="Spectral radius ρ(W)",
        xlabel="epoch",
        ylabel="ρ(W)",
        legend=True,
    )

    # Panel D: Frobenius norm
    for cond, runs in W.items():
        x, m, s = _mean_sem_align(runs, "fro_W")
        if x is None:
            continue
        axD.plot(x, m, lw=2, label=cond)
        axD.fill_between(x, m - s, m + s, alpha=0.15)
    _style(
        axD,
        fontsize,
        title="Frobenius ‖W‖_F",
        xlabel="epoch",
        ylabel="‖W‖_F",
        legend=True,
    )

    fig.tight_layout()
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# --------------------------------------------
# Figure 2 – Accuracy panels (eval metrics)
# --------------------------------------------


def fig2_symmetry_vs_performance(condition_df, savepath=None, fontsize=12):
    """
    Figure 2: Symmetry–performance relations (two panels, curves vs alpha, ±std).
      Panel A: mse_open (↓) vs alpha
      Panel B: best_loss (↓) vs alpha
    Expects tall condition_summary.csv with columns: ['condition_id','metric','mean','std'].
    """
    req = {"condition_id", "metric", "mean"}
    if condition_df is None or not req.issubset(condition_df.columns):
        print("[SKIP] fig2_symmetry_vs_performance: condition_summary missing columns")
        return

    # wide tables: one row per condition_id, columns are metric names
    csw_mean = _cs_to_wide(condition_df, value_col="mean")
    csw_std = _cs_to_wide(condition_df, value_col="std")
    if csw_mean is None or csw_mean.empty:
        print("[SKIP] fig2_symmetry_vs_performance: could not pivot condition_df")
        return

    # Extract alpha from condition_id; drop rows without alpha
    csw_mean["alpha"] = csw_mean["condition_id"].apply(_alpha_from_condition_id)
    if csw_std is not None and "condition_id" in csw_std.columns:
        csw_std["alpha"] = csw_std["condition_id"].apply(_alpha_from_condition_id)

    csw_mean = csw_mean.dropna(subset=["alpha"]).copy()
    csw_mean = csw_mean.sort_values("alpha")
    if csw_std is not None:
        csw_std = csw_std.dropna(subset=["alpha"]).copy()
        csw_std = (
            csw_std.set_index("condition_id")
            .reindex(csw_mean["condition_id"])
            .reset_index()
        )

    # Pick metrics (if absent, skip the panel gracefully)
    panels = [
        ("Open-loop MSE (↓) vs α", "mse_open"),
        ("Best training loss (↓) vs α", "best_loss"),
    ]

    import math

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs = np.array(axs).reshape(-1)

    plotted_any = False
    for i, (title, metric) in enumerate(panels):
        ax = axs[i]
        if metric not in csw_mean.columns:
            ax.axis("off")
            continue
        x = csw_mean["alpha"].values
        y = csw_mean[metric].values
        yerr = None
        if csw_std is not None and metric in csw_std.columns:
            yerr = (
                csw_std.set_index("condition_id")[metric]
                .reindex(csw_mean["condition_id"])
                .values
            )

        # line with errorbars
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, linewidth=1.6)
        else:
            ax.plot(x, y, "-o", linewidth=1.6)

        _style(ax, fontsize, title=title, xlabel="α (symmetry mix)", ylabel=metric)
        # Optional: invert y if you prefer “higher = better” visualization. Here we keep natural scale.

        plotted_any = True

    if not plotted_any:
        print("[SKIP] fig2_symmetry_vs_performance: no target metrics present")
        return

    fig.tight_layout()
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 3 – Emergent traveling-waves and replay dynamics
# -----------------------------------------------------------


def fig3_traveling_wave_and_polar(
    condition_dir,
    run_select="best_replay",  # "best_replay" or "best_mse"
    savepath=None,
    fontsize=12,
    max_units=100,
):
    """
    Panel A: hidden heatmaps (replay & prediction), time x unit (units sorted by peak time).
    Panel B: polar angle vs time (replay=pred only; prediction=pred + true).
    Expects per-run trace files saved by evaluate.py:
      *_replay_hidden.npy, *_prediction_hidden.npy,
      *_replay_angles.csv,  *_prediction_angles.csv
    """
    import pandas as pd

    condition_dir = Path(str(condition_dir).strip()).expanduser()

    # Try exact, then relative-with-leading-dot, then any *run_level*.csv
    candidates = [
        condition_dir / "run_level.csv",
        Path("." + str(condition_dir))
        / "run_level.csv",  # handles accidental leading '/'
    ]
    candidates += list(condition_dir.glob("*run_level*.csv"))

    # Pick the first that exists
    rpath = next((p for p in candidates if p.exists()), None)

    print("[fig3] condition_dir:", condition_dir)
    print("[fig3] run_level candidates:", [str(p) for p in candidates])
    print("[fig3] chosen run_level.csv:", rpath)

    if rpath is None:
        try:
            print("[fig3] ls:", os.listdir(str(condition_dir)))
        except Exception as e:
            print("[fig3] cannot list dir:", e)
        print(f"[SKIP] fig3: no run_level.csv in {condition_dir}")
        return

    # anchor all paths to the *actual* CSV's parent dir (avoids /runs vs ./runs mismatch)
    condition_dir = rpath.parent.resolve()

    # Decide which run to visualize, based on run_level.csv
    df = pd.read_csv(str(rpath))
    if df.empty:
        print(f"[SKIP] fig3: run_level.csv is empty in {condition_dir}")
        return

    # Choose the “best” run for the requested criterion
    if run_select == "best_replay" and "replay_r2" in df.columns:
        pick = df.sort_values("replay_r2", ascending=False).head(1)
    else:
        key = (
            "mse_open"
            if "mse_open" in df.columns
            else ("final_loss" if "final_loss" in df.columns else None)
        )
        if key is None:
            print(
                f"[SKIP] fig3: neither 'replay_r2' nor ('mse_open'/'final_loss') in run_level.csv"
            )
            return
        pick = df.sort_values(key, ascending=(key != "replay_r2")).head(1)

    if pick.empty or "run_id" not in pick.columns:
        print(f"[SKIP] fig3: cannot identify run_id from run_level.csv")
        return

    run_id = int(pick.iloc[0]["run_id"])

    # resolve stub paths inside run_XX
    run_dir = condition_dir / f"run_{int(run_id):02d}"
    # Find a stub from the checkpoint filename:
    ckpts = list(run_dir.glob("*.pth.tar"))
    if not ckpts:
        print(f"[SKIP] fig3: no checkpoint in {run_dir}")
        return

    # strip the trailing ".pth.tar" to get the stub
    ckpt_path = str(ckpts[0])
    if ckpt_path.endswith(".pth.tar"):
        stub = ckpt_path[:-8]  # drop the 8 chars ".pth.tar"
    else:
        # fallback: drop one suffix
        stub = str(Path(ckpt_path).with_suffix(""))

    # load traces (allow missing prediction files gracefully)
    rp_h = (
        np.load(stub + "_replay_hidden.npy")
        if os.path.exists(stub + "_replay_hidden.npy")
        else None
    )
    pr_h = (
        np.load(stub + "_prediction_hidden.npy")
        if os.path.exists(stub + "_prediction_hidden.npy")
        else None
    )

    def _read_angles_csv(path):
        if not os.path.exists(path):
            return None
        import csv

        rows = []
        with open(path, "r") as f:
            for i, r in enumerate(csv.reader(f)):
                if i == 0:  # header
                    header = r
                    continue
                rows.append(r)
        arr = np.array(rows, dtype=float) if rows else None
        return header, arr

    rp_hdr, rp_ang = _read_angles_csv(stub + "_replay_angles.csv")
    pr_hdr, pr_ang = _read_angles_csv(stub + "_prediction_angles.csv")

    if rp_h is None and pr_h is None:
        print(f"[SKIP] fig3: no hidden traces in {run_dir}")
        return

    # --- Panel A: heatmaps ---
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    # helper to plot one heatmap (first batch only)
    def _heatmap(ax, H, title):
        # H: [B,T,H] -> use B=0
        X = H[0]  # [T,H]
        # z-score per unit for visualization
        Xm = X - X.mean(axis=0, keepdims=True)
        Xs = X.std(axis=0, keepdims=True) + 1e-9
        Z = Xm / Xs

        # sort units by peak time to reveal wave
        peak_t = np.argmax(Z, axis=0)
        order = np.argsort(peak_t)
        if Z.shape[1] > max_units:
            order = order[:max_units]
        Zs = Z[:, order].T  # [units, time] for nicer heatmap orientation

        im = ax.imshow(Zs, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("units (sorted)")
        return im

    axA = fig.add_subplot(gs[0, 0])
    if rp_h is not None:
        _heatmap(axA, rp_h, "Replay: hidden (time × unit)")
    else:
        axA.axis("off")
        axA.set_title("Replay: (no trace)")

    axB = fig.add_subplot(gs[0, 1])
    if pr_h is not None:
        _heatmap(axB, pr_h, "Prediction: hidden (time × unit)")
    else:
        axB.axis("off")
        axB.set_title("Prediction: (no trace)")

    # --- Panel B: polar plots ---
    axC = fig.add_subplot(gs[1, :], projection="polar")

    def _plot_polar(ax, hdr, arr, label_pred, label_true=None):
        if arr is None:
            return
        cols = {name: i for i, name in enumerate(hdr)}
        t = arr[:, cols.get("t", 0)]
        tp = arr[:, cols.get("theta_pred", 1)]
        if "theta_true" in cols:
            tt = arr[:, cols["theta_true"]]
        else:
            tt = None
        ax.plot(tp, t, lw=1.5, label=label_pred)
        if tt is not None:
            ax.plot(tt, t, lw=1.0, linestyle="--", label=label_true)

    _plot_polar(axC, rp_hdr, rp_ang, label_pred="Replay θ̂(t)")
    _plot_polar(
        axC, pr_hdr, pr_ang, label_pred="Prediction θ̂(t)", label_true="Prediction θ(t)"
    )

    axC.set_title("Polar: decoded angle vs time")
    axC.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    fig.suptitle(
        f"Fig 3 — Traveling-wave & Polar (condition: {condition_dir.name}, run: {run_id})",
        fontsize=fontsize + 2,
    )

    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ------------------------------------------------
# Figure 4 – Spectral radius vs performance scatter
# ------------------------------------------------


def fig_sprad_vs_perf(
    run_level_df,
    savepath=None,
    fontsize=12,
    sprad_col="spectral_radius_last",
    perf_col="openloop_mse_last",
):
    if run_level_df is None or sprad_col not in run_level_df.columns:
        print(
            "[SKIP] fig_sprad_vs_perf: missing run_level_df or spectral radius column"
        )
        return
    if perf_col not in run_level_df.columns:
        print("[SKIP] fig_sprad_vs_perf: perf column '%s' absent" % perf_col)
        return

    df = run_level_df.dropna(subset=[sprad_col, perf_col]).copy()
    if df.empty:
        print("[SKIP] fig_sprad_vs_perf: no valid rows")
        return
    cond = df.get("condition_id", pd.Series(["?"] * len(df))).astype(str).values

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in sorted(set(cond)):
        sub = df[cond == c]
        ax.scatter(
            sub[sprad_col].values, sub[perf_col].values, s=18, label=c, alpha=0.8
        )
    _style(
        ax,
        fontsize,
        title="Spectral radius vs Performance",
        xlabel="spectral radius (last)",
        ylabel=perf_col,
        legend=True,
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 5 – Weight drift & symmetry (phase-plane per run)
# -----------------------------------------------------------


def fig_phase_plane_sym_asym(
    run_level_df,
    savepath=None,
    fontsize=12,
    sym_col="fro_S_last",
    asym_col="fro_A_last",
    color_by="final_loss",
):
    """
    Phase-plane: ||S||_F vs ||A||_F at last snapshot; color by a performance metric.
    """
    if run_level_df is None or not _has_cols(run_level_df, [sym_col, asym_col]):
        print("[SKIP] fig_phase_plane_sym_asym: missing S/A columns")
        return

    df = run_level_df.dropna(subset=[sym_col, asym_col]).copy()
    if df.empty:
        print("[SKIP] fig_phase_plane_sym_asym: no valid rows")
        return

    cvals = _col(df, color_by)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(
        df[sym_col].values,
        df[asym_col].values,
        s=18,
        c=cvals.values if cvals is not None else None,
        alpha=0.85,
    )
    if cvals is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize - 1)
        cb.set_label(color_by, fontsize=fontsize)
    _style(
        ax,
        fontsize,
        title="Sym vs Asym (Frobenius)",
        xlabel="||S||_F (last)",
        ylabel="||A||_F (last)",
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 6 – Gradient × Mixing overlay (per run)
# -----------------------------------------------------------


def fig_grad_mix_overlay(
    run_level_df,
    savepath=None,
    fontsize=12,
    grad_col="grad_global_L2_post_last",
    mix_col="mix_A_over_S_last",
):
    if run_level_df is None or not _has_cols(run_level_df, [grad_col, mix_col]):
        print("[SKIP] fig_grad_mix_overlay: missing columns")
        return
    df = run_level_df.dropna(subset=[grad_col, mix_col]).copy()
    if df.empty:
        print("[SKIP] fig_grad_mix_overlay: no valid rows")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cond = df.get("condition_id", pd.Series(["?"] * len(df))).astype(str).values
    for c in sorted(set(cond)):
        sub = df[cond == c]
        ax.scatter(sub[mix_col].values, sub[grad_col].values, s=18, alpha=0.8, label=c)
    _style(
        ax,
        fontsize,
        title="Gradient × Mixing",
        xlabel="A/S (last)",
        ylabel="global grad L2 (post)",
        legend=True,
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 7 – Spectral portrait (if offline NPZ or CSV exists)
# -----------------------------------------------------------


def fig_eigs_lines_offline(
    offline_df, savepath=None, fontsize=12, group_by="condition_id", max_curves=12
):
    """
    Plot eigenvalue magnitude curves (if stored as columns like eig_0,... or via CSV rows).
    This is a best-effort panel: if offline_df has 'spectral_radius' only, we skip.
    Expected schemas vary; we try a few reasonable patterns:
      - Wide: columns 'eig_0', 'eig_1', ... OR 'sv_0', 'sv_1', ...
      - Long:  columns 'eigval', 'rank', 'snapshot' per run_id
    """
    if offline_df is None:
        print("[SKIP] fig_eigs_lines_offline: no offline_df")
        return

    # Wide format first
    eig_cols = [c for c in offline_df.columns if c.startswith("eig_")]
    sv_cols = [c for c in offline_df.columns if c.startswith("sv_")]
    df = None
    ylabel = ""

    if eig_cols:
        df = offline_df[
            eig_cols
            + ["run_id"]
            + ([group_by] if group_by in offline_df.columns else [])
        ].copy()
        ylabel = "eigenvalue magnitude (sorted)"
        # sort columns by index
        eig_cols = sorted(eig_cols, key=lambda s: int(s.split("_")[1]))
        curves = df[eig_cols].values
        labels = df.get(group_by, pd.Series(["?"] * len(df))).astype(str).values
    elif sv_cols:
        df = offline_df[
            sv_cols
            + ["run_id"]
            + ([group_by] if group_by in offline_df.columns else [])
        ].copy()
        ylabel = "singular value (sorted)"
        sv_cols = sorted(sv_cols, key=lambda s: int(s.split("_")[1]))
        curves = df[sv_cols].values
        labels = df.get(group_by, pd.Series(["?"] * len(df))).astype(str).values
    else:
        # Long format attempt (need columns: 'rank' & 'eigval' or 'sv')
        need1 = set(["rank", "eigval"])
        need2 = set(["rank", "sv"])
        if set(offline_df.columns).issuperset(need1):
            ylabel = "eigenvalue magnitude"
            # pick up to max_curves runs to plot
            curves = []
            labels = []
            for rid, g in offline_df.groupby("run_id"):
                g2 = g.sort_values("rank")
                curves.append(g2["eigval"].values)
                labels.append(
                    str(g2[group_by].iloc[0]) if group_by in g2.columns else str(rid)
                )
                if len(curves) >= max_curves:
                    break
        elif set(offline_df.columns).issuperset(need2):
            ylabel = "singular value"
            curves = []
            labels = []
            for rid, g in offline_df.groupby("run_id"):
                g2 = g.sort_values("rank")
                curves.append(g2["sv"].values)
                labels.append(
                    str(g2[group_by].iloc[0]) if group_by in g2.columns else str(rid)
                )
                if len(curves) >= max_curves:
                    break
        else:
            print("[SKIP] fig_eigs_lines_offline: no recognizable eig/sv columns")
            return

    if curves is None or len(curves) == 0:
        print("[SKIP] fig_eigs_lines_offline: empty curves")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    kmax = max(len(c) for c in curves)
    x = np.arange(kmax)
    # plot up to max_curves with light alpha
    for i, y in enumerate(curves[:max_curves]):
        ax.plot(
            x[: len(y)],
            y,
            linewidth=1.3,
            alpha=0.7,
            label=labels[i] if i < 10 else None,
        )
    if len(curves) > 1:
        _style(
            ax,
            fontsize,
            title="Spectral lines (subset)",
            xlabel="rank",
            ylabel=ylabel,
            legend=True,
        )
    else:
        _style(ax, fontsize, title="Spectral line", xlabel="rank", ylabel=ylabel)
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ------------------------------------------
# Orchestrator: make all default main figs
# ------------------------------------------


def make_all_figures(
    agg_dir="./runs_agg",
    figdir="./figs",
    condition_regex=None,
    run_regex=None,
    fontsize=12,
):
    _ensure_dir(figdir)
    rl, cs, off, ev = load_all(agg_dir=agg_dir)

    rl = _filter_df(rl, condition_regex, run_regex)
    cs = _filter_df(cs, condition_regex, run_regex)
    off = _filter_df(off, condition_regex, run_regex)

    # 1) Accuracy and convergence speed
    fig1_training_dynamics(
        cond_roots if cond_roots else [args.agg_dir],
        os.path.join(args.figdir, "fig1_training_dynamics.png"),
        fontsize=args.fontsize,
    )

    # 2) Mixing vs performance
    fig2_symmetry_vs_performance(
        cs,
        savepath=os.path.join(args.figdir, "fig2_symmetry_vs_performance.png"),
        fontsize=args.fontsize,
    )

    # 3) Emergent traveling-waves & polar plots
    cond_for_fig3 = cond_roots[0] if cond_roots else args.agg_dir
    fig3_traveling_wave_and_polar(
        condition_dir=cond_for_fig3,
        run_select="best_replay",
        savepath=os.path.join(args.figdir, "fig3_traveling_and_polar.png"),
        fontsize=args.fontsize,
    )

    # 4) Spectral radius vs performance
    fig_sprad_vs_perf(
        rl,
        savepath=os.path.join(figdir, "fig4_sprad_vs_perf.png"),
        fontsize=fontsize,
        sprad_col=(
            "spectral_radius_last"
            if rl is not None and "spectral_radius_last" in rl.columns
            else "fro_W_last"
        ),
        perf_col=(
            "openloop_mse_last"
            if rl is not None and "openloop_mse_last" in rl.columns
            else "final_loss"
        ),
    )

    # 5) Phase-plane S vs A
    fig_phase_plane_sym_asym(
        rl,
        savepath=os.path.join(figdir, "fig5_sym_vs_asym.png"),
        fontsize=fontsize,
        sym_col=(
            "fro_S_last"
            if rl is not None and "fro_S_last" in (rl.columns if rl is not None else [])
            else "fro_W_last"
        ),
        asym_col=(
            "fro_A_last"
            if rl is not None and "fro_A_last" in (rl.columns if rl is not None else [])
            else "fro_W_last"
        ),
        color_by=(
            "final_loss"
            if rl is not None and "final_loss" in rl.columns
            else "openloop_mse_last"
        ),
    )

    # 6) Gradient × Mixing
    fig_grad_mix_overlay(
        rl,
        savepath=os.path.join(figdir, "fig6_grad_times_mix.png"),
        fontsize=fontsize,
        grad_col=(
            "grad_global_L2_post_last"
            if rl is not None and "grad_global_L2_post_last" in rl.columns
            else "grad_global_RMS_post_last"
        ),
        mix_col=(
            "mix_A_over_S_last"
            if rl is not None and "mix_A_over_S_last" in rl.columns
            else "sym_ratio_last"
        ),
    )

    # 7) Spectral lines (subset)
    fig_eigs_lines_offline(
        off,
        savepath=os.path.join(figdir, "fig7_spectral_lines.png"),
        fontsize=fontsize,
        group_by=(
            "condition_id"
            if off is not None and "condition_id" in off.columns
            else "run_id"
        ),
    )


# ---------------------------
# CLI
# ---------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate figures from aggregated metrics")
    p.add_argument(
        "--agg_dir",
        type=str,
        default="./runs_agg",
        help="Directory containing run_level.csv & condition_summary.csv",
    )
    p.add_argument("--figdir", type=str, default="./figs", help="Where to save PNGs")
    p.add_argument(
        "--fontsize", type=int, default=12, help="Base font size for all plots"
    )
    p.add_argument(
        "--condition_regex",
        type=str,
        default=None,
        help="Regex to filter conditions (applied to condition_id)",
    )
    p.add_argument(
        "--run_regex",
        type=str,
        default=None,
        help="Regex to filter runs (applied to run_id)",
    )
    p.add_argument(
        "--just",
        type=str,
        default="",
        help="Comma list of figures to render (1..7). Empty=all.",
    )
    p.add_argument(
        "--conditions",
        type=str,
        default="",
        help="Comma-separated list of condition root folders (each has run_level.csv etc.)",
    )
    p.add_argument(
        "--cond_glob",
        type=str,
        default="",
        help="Glob that expands to multiple condition roots (e.g. './runs/ElmanRNN/mh-variants/shifted-cyc/frobenius/*')",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Build list of condition roots if provided
    cond_roots = []
    if args.conditions:
        cond_roots.extend([s.strip() for s in args.conditions.split(",") if s.strip()])
    if args.cond_glob:
        cond_roots.extend(sorted(glob.glob(args.cond_glob)))

    if cond_roots:
        # Multi-condition path
        rl, cs, off, ev = load_many_conditions(cond_roots)
    else:
        # Backward-compatible single-agg-dir path
        rl, cs, off, ev = load_all(agg_dir=args.agg_dir)

    rl = _filter_df(rl, args.condition_regex, args.run_regex)
    cs = _filter_df(cs, args.condition_regex, args.run_regex)
    off = _filter_df(off, args.condition_regex, args.run_regex)

    _ensure_dir(args.figdir)
    requested = (
        set([s.strip() for s in args.just.split(",") if s.strip()])
        if args.just
        else set()
    )

    def want(k):
        return (not requested) or (str(k) in requested)

    if want(1):
        fig1_training_dynamics(
            cond_roots if cond_roots else [args.agg_dir],
            os.path.join(args.figdir, "fig1_training_dynamics.png"),
            fontsize=args.fontsize,
        )
    if want(2):
        fig2_symmetry_vs_performance(
            cs,
            savepath=os.path.join(args.figdir, "fig2_symmetry_vs_performance.png"),
            fontsize=args.fontsize,
        )
    if want(3):
        cond_for_fig3 = cond_roots[0] if cond_roots else args.agg_dir
        fig3_traveling_wave_and_polar(
            condition_dir=cond_for_fig3,
            run_select="best_replay",
            savepath=os.path.join(args.figdir, "fig3_traveling_and_polar.png"),
            fontsize=args.fontsize,
        )
    print("QUITTING AFTER FIG 3")
    quit()
    if want(4):
        fig_sprad_vs_perf(
            rl,
            os.path.join(args.figdir, "fig4_sprad_vs_perf.png"),
            fontsize=args.fontsize,
        )
    if want(5):
        fig_phase_plane_sym_asym(
            rl,
            os.path.join(args.figdir, "fig5_sym_vs_asym.png"),
            fontsize=args.fontsize,
        )
    if want(6):
        fig_grad_mix_overlay(
            rl,
            os.path.join(args.figdir, "fig6_grad_times_mix.png"),
            fontsize=args.fontsize,
        )
    if want(7):
        fig_eigs_lines_offline(
            off,
            os.path.join(args.figdir, "fig7_spectral_lines.png"),
            fontsize=args.fontsize,
        )


if __name__ == "__main__":
    main()
