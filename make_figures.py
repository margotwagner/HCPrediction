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

import numpy as np
import pandas as pd
import matplotlib

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


def fig_convergence_speed(condition_df, savepath=None, fontsize=12):
    """
    Bar or dot plot of convergence metrics per condition:
    - Uses columns like: 'final_loss_mean', 'loss_auc_mean' if present.
    """
    if condition_df is None:
        print("[SKIP] fig_convergence_speed: no condition_df")
        return

    # prefer AUC if present, else final_loss
    metric_name = None
    for cand in ["loss_auc_mean", "final_loss_mean", "openloop_mse_mean"]:
        if cand in condition_df.columns:
            metric_name = cand
            break
    if metric_name is None:
        print("[SKIP] fig_convergence_speed: missing loss metrics")
        return

    df = condition_df.sort_values(metric_name)
    labels = df.get("condition_id", pd.Series(range(len(df)))).astype(str).tolist()
    y = df[metric_name].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(y)), y, width=0.7)
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ttl = "Convergence (lower better): %s" % metric_name
    _style(ax, fontsize, title=ttl, xlabel="Condition", ylabel=metric_name)
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# --------------------------------------------
# Figure 2 – Accuracy panels (eval metrics)
# --------------------------------------------


def fig_accuracy_panels(condition_df, savepath=None, fontsize=12):
    """
    Multi-panel condition-level metrics if available:
    - open-loop MSE (lower better)
    - replay R^2 (higher better)
    - prediction time-to-divergence (higher better) or phase drift (lower)
    - closed-loop: mean drift (lower) / TTD (higher)
    """
    if condition_df is None:
        print("[SKIP] fig_accuracy_panels: no condition_df")
        return

    keys = [
        ("openloop_mse_mean", "Open-loop MSE (↓)"),
        ("replay_r2_mean", "Replay R² (↑)"),
        ("prediction_time_to_div_mean", "Prediction TTD (↑)"),
        ("prediction_phase_drift_mean", "Prediction phase drift (↓)"),
        ("closedloop_time_to_div_mean", "Closed-loop TTD (↑)"),
        ("closedloop_phase_drift_mean", "Closed-loop phase drift (↓)"),
    ]
    present = [(k, lab) for (k, lab) in keys if k in condition_df.columns]
    if not present:
        print("[SKIP] fig_accuracy_panels: no accuracy columns")
        return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 2, 4), squeeze=False)
    axr = axes[0]

    df = condition_df.copy()
    labels = df.get("condition_id", pd.Series(range(len(df)))).astype(str).tolist()
    x = np.arange(len(labels))

    for i, (k, lab) in enumerate(present):
        ax = axr[i]
        y = df[k].values
        ax.bar(x, y, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        _style(ax, fontsize, title=lab, xlabel="Condition")
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 3 – Mixing ratio vs performance (scatter per run)
# -----------------------------------------------------------


def fig_mix_vs_perf(
    run_level_df,
    savepath=None,
    fontsize=12,
    mix_col="mix_A_over_S_last",
    perf_col="openloop_mse_last",
):
    """
    Scatter across runs: mixing ratio (A/S) vs performance.
    You can change perf_col to e.g. 'replay_r2_last' or 'final_loss'.
    """
    if run_level_df is None or mix_col not in run_level_df.columns:
        print("[SKIP] fig_mix_vs_perf: missing run_level_df or mix column")
        return
    if perf_col not in run_level_df.columns:
        print("[SKIP] fig_mix_vs_perf: perf column '%s' absent" % perf_col)
        return

    df = run_level_df.dropna(subset=[mix_col, perf_col]).copy()
    if df.empty:
        print("[SKIP] fig_mix_vs_perf: no valid rows")
        return
    cond = df.get("condition_id", pd.Series(["?"] * len(df))).astype(str).values

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c in sorted(set(cond)):
        sub = df[cond == c]
        ax.scatter(sub[mix_col].values, sub[perf_col].values, s=18, label=c, alpha=0.8)
    _style(
        ax,
        fontsize,
        title="Mixing (A/S) vs Performance",
        xlabel="A/S (last)",
        ylabel=perf_col,
        legend=True,
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

    # 1) Convergence speed
    fig_convergence_speed(
        cs, savepath=os.path.join(figdir, "fig1_convergence.png"), fontsize=fontsize
    )

    # 2) Accuracy panels
    fig_accuracy_panels(
        cs, savepath=os.path.join(figdir, "fig2_accuracy_panels.png"), fontsize=fontsize
    )

    # 3) Mixing vs performance (tweak perf_col if you prefer)
    fig_mix_vs_perf(
        rl,
        savepath=os.path.join(figdir, "fig3_mix_vs_perf.png"),
        fontsize=fontsize,
        mix_col="mix_A_over_S_last",
        perf_col=(
            "openloop_mse_last"
            if rl is not None and "openloop_mse_last" in rl.columns
            else "final_loss"
        ),
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
        fig_convergence_speed(
            cs,
            os.path.join(args.figdir, "fig1_convergence.png"),
            fontsize=args.fontsize,
        )
    if want(2):
        fig_accuracy_panels(
            cs,
            os.path.join(args.figdir, "fig2_accuracy_panels.png"),
            fontsize=args.fontsize,
        )
    if want(3):
        fig_mix_vs_perf(
            rl,
            os.path.join(args.figdir, "fig3_mix_vs_perf.png"),
            fontsize=args.fontsize,
        )
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
