#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_supp.py
Supplemental figures for the Elman RNN project.
- Pure pandas/numpy/matplotlib; Python 3.6.7-compatible.
- Works as CLI script and as an importable module for notebooks.

Inputs expected (same as make_figures.py):
- Aggregated CSVs under --agg_dir (default: ./runs_agg)
    run_level.csv           (per-run)
    condition_summary.csv   (per-condition)
    offline_all.csv         (optional; per-run long/ wide for spectra etc.)

Outputs:
- PNGs to --figdir (default: ./figs_supp)

Usage:
    python make_figures_supp.py --agg_dir ./runs_agg --figdir ./figs_supp --fontsize 12
    python make_figures_supp.py --agg_dir ./runs_agg --figdir ./figs_supp --just 1,3

Notebook:
    import make_figures_supp as ms
    rl, cs, off, ev = ms.load_all(agg_dir="./runs_agg")
    ms.fig_ecdf_convergence(rl, savepath="figs_supp/ecdf.png", fontsize=14)
"""

from __future__ import print_function
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# IO helpers (mirrors main)
# ---------------------------


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except TypeError:
            # python 3.6 fallback (exist_ok exists, but guard anyway)
            if not os.path.isdir(path):
                os.makedirs(path)


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _read_csv_safe(path):
    if path and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print("[WARN] Failed to read:", path, "->", e)
            return None
    return None


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
    agg_dir="./runs_agg", run_level_csv=None, condition_csv=None, offline_csv=None
):
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

    # Attempt to gather per-run offline CSVs if merged file missing
    if off is None:
        cands = glob.glob(os.path.join(agg_dir, "**", "*offline*.csv"), recursive=True)
        frames = []
        for c in cands:
            df = _read_csv_safe(c)
            if df is not None:
                frames.append(df)
        if frames:
            try:
                off = pd.concat(frames, sort=False, ignore_index=True)
            except Exception:
                off = None

    return rl, cs, off, None


def _style(ax, fontsize=12, title=None, xlabel=None, ylabel=None, legend=False):
    ax.tick_params(axis="both", labelsize=fontsize)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
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
                try:
                    l.set_linewidth(2.0)
                except Exception:
                    pass


def _savefig(fig, path):
    _ensure_dir(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("[SAVE]", path)


# ---------------------------------------------------------
# 1) ECDF of convergence / time-to-target across runs
# ---------------------------------------------------------


def fig_ecdf_convergence(
    run_level_df,
    savepath=None,
    fontsize=12,
    epoch_col_candidates=("convergence_epoch", "best_epoch", "epoch_to_1p1x_best"),
):
    if run_level_df is None:
        print("[SKIP] ECDF: missing run_level_df")
        return
    epoch_col = None
    for c in epoch_col_candidates:
        if c in run_level_df.columns:
            epoch_col = c
            break
    if epoch_col is None:
        print("[SKIP] ECDF: no convergence epoch column found:", epoch_col_candidates)
        return

    df = run_level_df.dropna(subset=[epoch_col]).copy()
    if df.empty:
        print("[SKIP] ECDF: empty after dropna")
        return

    vals = np.asarray(df[epoch_col].values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print("[SKIP] ECDF: no finite values")
        return
    xs = np.sort(vals)
    ys = np.arange(1, len(xs) + 1, dtype=np.float64) / float(len(xs))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(xs, ys, lw=2.0)
    _style(
        ax,
        fontsize,
        title="Convergence ECDF",
        xlabel="epoch",
        ylabel="fraction of runs ≤ epoch",
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 2) Closed-loop drift distributions per condition
# ---------------------------------------------------------


def fig_closedloop_drift_hist(
    run_level_df,
    savepath=None,
    fontsize=12,
    drift_col_candidates=("closedloop_phase_drift_last", "closedloop_phase_drift_mean"),
):
    if run_level_df is None:
        print("[SKIP] closed-loop drift hist: missing run_level_df")
        return
    dcol = None
    for c in drift_col_candidates:
        if c in run_level_df.columns:
            dcol = c
            break
    if dcol is None:
        print("[SKIP] closed-loop drift hist: no drift column", drift_col_candidates)
        return
    if "condition_id" not in run_level_df.columns:
        print("[SKIP] closed-loop drift hist: need condition_id to facet")
        return

    df = run_level_df.dropna(subset=[dcol]).copy()
    if df.empty:
        print("[SKIP] closed-loop drift hist: empty")
        return
    conds = sorted(df["condition_id"].astype(str).unique().tolist())
    K = min(6, len(conds))
    fig, axes = plt.subplots(1, K, figsize=(4 * K + 1, 3.5), squeeze=False)
    axr = axes[0]

    for i, c in enumerate(conds[:K]):
        ax = axr[i]
        vals = df.loc[df["condition_id"].astype(str) == c, dcol].values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=fontsize)
        else:
            ax.hist(vals, bins=20, alpha=0.9)
        _style(ax, fontsize, title=str(c), xlabel=dcol, ylabel="count")
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 3) Replay vs Prediction grid (per run scatter)
# ---------------------------------------------------------


def fig_replay_vs_prediction(
    run_level_df,
    savepath=None,
    fontsize=12,
    replay_col="replay_r2_last",
    pred_col="prediction_time_to_div_last",
):
    if run_level_df is None or replay_col not in run_level_df.columns:
        print("[SKIP] replay vs prediction: replay column missing")
        return
    if pred_col not in run_level_df.columns:
        print("[SKIP] replay vs prediction: prediction column missing")
        return

    df = run_level_df.dropna(subset=[replay_col, pred_col]).copy()
    if df.empty:
        print("[SKIP] replay vs prediction: empty")
        return
    cond = df.get("condition_id", pd.Series(["?"] * len(df))).astype(str).values

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in sorted(set(cond)):
        sub = df[cond == c]
        ax.scatter(
            sub[replay_col].values, sub[pred_col].values, s=18, alpha=0.8, label=c
        )
    _style(
        ax,
        fontsize,
        title="Replay R² vs Prediction TTD",
        xlabel=replay_col + " (↑)",
        ylabel=pred_col + " (↑)",
        legend=True,
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 4) Residual L2 autocorr (distribution)
# ---------------------------------------------------------


def fig_residual_autocorr(
    run_level_df,
    savepath=None,
    fontsize=12,
    col_candidates=("residual_lag1_autocorr_last", "residual_lag1_autocorr_mean"),
):
    if run_level_df is None:
        print("[SKIP] residual autocorr: missing run_level_df")
        return
    ccol = None
    for c in col_candidates:
        if c in run_level_df.columns:
            ccol = c
            break
    if ccol is None:
        print("[SKIP] residual autocorr: no column", col_candidates)
        return
    df = run_level_df.dropna(subset=[ccol]).copy()
    if df.empty:
        print("[SKIP] residual autocorr: empty")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(df[ccol].values, bins=25, alpha=0.9)
    _style(
        ax,
        fontsize,
        title="Residual L2 lag-1 autocorrelation",
        xlabel=ccol,
        ylabel="count",
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 5) Weight trace evolution (offline long format)
# ---------------------------------------------------------


def fig_weight_trace(
    off_df,
    savepath=None,
    fontsize=12,
    epoch_col="epoch",
    fro_col="fro_W",
    group_col="run_id",
):
    """
    Plots Frobenius norm of W_hh over epochs for a subset of runs.
    Requires offline long-format with columns: epoch, fro_W, run_id (at least).
    """
    if off_df is None:
        print("[SKIP] weight trace: no offline_df")
        return
    need = set([epoch_col, fro_col, group_col])
    if not set(off_df.columns).issuperset(need):
        print("[SKIP] weight trace: need columns", need)
        return

    # take up to 10 runs for readability
    runs = off_df[group_col].astype(str).unique().tolist()[:10]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for r in runs:
        g = off_df[off_df[group_col].astype(str) == r].sort_values(epoch_col)
        x = g[epoch_col].values
        y = g[fro_col].values
        if len(x) == 0:
            continue
        ax.plot(x, y, lw=1.4, alpha=0.8, label=str(r))
    _style(
        ax,
        fontsize,
        title="Weight Frobenius trace",
        xlabel="epoch",
        ylabel="||W_hh||_F",
        legend=True,
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 6) Open-loop vs Closed-loop error scatter
# ---------------------------------------------------------


def fig_open_vs_closed(
    run_level_df,
    savepath=None,
    fontsize=12,
    open_col="openloop_mse_last",
    closed_col="closedloop_phase_drift_last",
):
    if run_level_df is None:
        print("[SKIP] open vs closed: missing run_level_df")
        return
    if open_col not in run_level_df.columns or closed_col not in run_level_df.columns:
        print("[SKIP] open vs closed: need", open_col, "and", closed_col)
        return

    df = run_level_df.dropna(subset=[open_col, closed_col]).copy()
    if df.empty:
        print("[SKIP] open vs closed: empty")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cond = df.get("condition_id", pd.Series(["?"] * len(df))).astype(str).values
    for c in sorted(set(cond)):
        sub = df[cond == c]
        ax.scatter(
            sub[open_col].values, sub[closed_col].values, s=18, alpha=0.8, label=c
        )
    _style(
        ax,
        fontsize,
        title="Open-loop vs Closed-loop",
        xlabel=open_col + " (↓)",
        ylabel=closed_col + " (↓)",
        legend=True,
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# 7) Eigenvalue magnitude density (offline)
# ---------------------------------------------------------


def fig_eigmag_hist(off_df, savepath=None, fontsize=12):
    """
    Histogram of eigenvalue magnitudes if available in offline long format.
    Expected columns: 'eigval' or wide 'eig_0, eig_1, ...' (then flattened).
    """
    if off_df is None:
        print("[SKIP] eig magnitude hist: no offline_df")
        return

    vals = None
    if "eigval" in off_df.columns:
        vals = off_df["eigval"].values
    else:
        eig_cols = [c for c in off_df.columns if c.startswith("eig_")]
        if eig_cols:
            try:
                vals = off_df[eig_cols].values.reshape(-1)
            except Exception:
                vals = None

    if vals is None:
        print("[SKIP] eig magnitude hist: no eigenvalue columns")
        return

    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print("[SKIP] eig magnitude hist: empty values")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(vals, bins=40, alpha=0.9)
    _style(
        ax,
        fontsize,
        title="Eigenvalue magnitude distribution",
        xlabel="|λ|",
        ylabel="count",
    )
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# ---------------------------------------------------------
# Master: make all supplemental figures
# ---------------------------------------------------------


def make_all_supp(
    agg_dir="./runs_agg",
    figdir="./figs_supp",
    condition_regex=None,
    run_regex=None,
    fontsize=12,
):
    _ensure_dir(figdir)
    rl, cs, off, ev = load_all(agg_dir=agg_dir)
    rl = _filter_df(rl, condition_regex, run_regex)
    cs = _filter_df(cs, condition_regex, run_regex)
    off = _filter_df(off, condition_regex, run_regex)

    fig_ecdf_convergence(
        rl,
        savepath=os.path.join(figdir, "supp1_ecdf_convergence.png"),
        fontsize=fontsize,
    )
    fig_closedloop_drift_hist(
        rl,
        savepath=os.path.join(figdir, "supp2_closedloop_drift_hist.png"),
        fontsize=fontsize,
    )
    fig_replay_vs_prediction(
        rl,
        savepath=os.path.join(figdir, "supp3_replay_vs_prediction.png"),
        fontsize=fontsize,
    )
    fig_residual_autocorr(
        rl,
        savepath=os.path.join(figdir, "supp4_residual_autocorr.png"),
        fontsize=fontsize,
    )
    fig_weight_trace(
        off, savepath=os.path.join(figdir, "supp5_weight_trace.png"), fontsize=fontsize
    )
    fig_open_vs_closed(
        rl, savepath=os.path.join(figdir, "supp6_open_vs_closed.png"), fontsize=fontsize
    )
    fig_eigmag_hist(
        off,
        savepath=os.path.join(figdir, "supp7_eig_magnitude_hist.png"),
        fontsize=fontsize,
    )


# ---------------------------
# CLI
# ---------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate supplemental figures")
    p.add_argument(
        "--agg_dir",
        type=str,
        default="./runs_agg",
        help="Directory with aggregated CSVs",
    )
    p.add_argument(
        "--figdir", type=str, default="./figs_supp", help="Where to save PNGs"
    )
    p.add_argument("--fontsize", type=int, default=12, help="Base font size")
    p.add_argument(
        "--condition_regex", type=str, default=None, help="Regex to filter conditions"
    )
    p.add_argument("--run_regex", type=str, default=None, help="Regex to filter runs")
    p.add_argument(
        "--just",
        type=str,
        default="",
        help="Comma list of figures to render (1..7), empty=all",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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
        fig_ecdf_convergence(
            rl,
            os.path.join(args.figdir, "supp1_ecdf_convergence.png"),
            fontsize=args.fontsize,
        )
    if want(2):
        fig_closedloop_drift_hist(
            rl,
            os.path.join(args.figdir, "supp2_closedloop_drift_hist.png"),
            fontsize=args.fontsize,
        )
    if want(3):
        fig_replay_vs_prediction(
            rl,
            os.path.join(args.figdir, "supp3_replay_vs_prediction.png"),
            fontsize=args.fontsize,
        )
    if want(4):
        fig_residual_autocorr(
            rl,
            os.path.join(args.figdir, "supp4_residual_autocorr.png"),
            fontsize=args.fontsize,
        )
    if want(5):
        fig_weight_trace(
            off,
            os.path.join(args.figdir, "supp5_weight_trace.png"),
            fontsize=args.fontsize,
        )
    if want(6):
        fig_open_vs_closed(
            rl,
            os.path.join(args.figdir, "supp6_open_vs_closed.png"),
            fontsize=args.fontsize,
        )
    if want(7):
        fig_eigmag_hist(
            off,
            os.path.join(args.figdir, "supp7_eig_magnitude_hist.png"),
            fontsize=args.fontsize,
        )


if __name__ == "__main__":
    main()
