#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
- Matplotlib-only figure suite for your Elman RNN project.
- Works both as a CLI script and as an importable module for notebooks.
- Python 3.6.7-compatible

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
import glob
import csv
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from typing import Optional
from matplotlib.lines import Line2D

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


def _short_label_from_root(root: str) -> str:
    # last path component only
    return os.path.normpath(str(root)).split(os.sep)[-1]


def _fit_slope(x, y, logx=False, logy=False):
    """
    Return slope m of a best-fit line after transforming axes
    according to logx/logy (base-10). We fit y_t = m*x_t + b where
    x_t = log10(x) if logx else x, and y_t = log10(y) if logy else y.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    xt = np.log10(x) if logx else x
    yt = np.log10(y) if logy else y
    m, b = np.polyfit(xt, yt, 1)
    return float(m)


def _pick_time_col_runlevel(df: pd.DataFrame) -> str:
    """Preferred time key for run_level: epoch > snapshot_idx > synthetic row index."""
    for c in ("epoch", "snapshot_idx"):
        if c in df.columns:
            return c
    df = df.copy()
    df["row_idx"] = np.arange(len(df))
    return "row_idx"


def _alpha_series_from_runlevel(df: pd.DataFrame):
    """
    Compute alpha = fro_S_offline / (fro_S_offline + fro_A_offline) for every row.
    Returns (time_col_name, time_values_np, alpha_values_np), or (None,None,None) if unavailable.
    """
    if df is None or df.empty:
        return None, None, None
    if not {"fro_S_offline", "fro_A_offline"}.issubset(df.columns):
        return None, None, None
    tcol = _pick_time_col_runlevel(df)
    t = pd.to_numeric(df[tcol], errors="coerce").values
    S = pd.to_numeric(df["fro_S_offline"], errors="coerce").astype(float).values
    A = pd.to_numeric(df["fro_A_offline"], errors="coerce").astype(float).values
    denom = S + A
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denom > 0, S / denom, np.nan)
    return tcol, t, alpha


def _linestyle_map_for_alphas(alpha_vals):
    """Stable map: sorted unique α0 -> a distinct linestyle."""
    LINESTYLES_ORDER = ["-", "--", "-.", ":", (0, (1, 1))]
    levels = sorted(set(float(a) for a in alpha_vals if pd.notna(a)))
    ls = {a: LINESTYLES_ORDER[i % len(LINESTYLES_ORDER)] for i, a in enumerate(levels)}
    return ls, levels


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
    # Use constrained layout if available
    try:
        fig.set_constrained_layout(True)
    except Exception:
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE]", path)


def _build_metrics_table_from_condition_summary(
    cs: pd.DataFrame, prefer: list
) -> Optional[pd.DataFrame]:
    """
    From condition_summary (long form: ['condition_id','metric','mean','std']),
    build a compact table (rows = metrics present & ordered by 'prefer';
    columns = condition_id; cells = 'mean±std' rounded).

    Returns a DataFrame of strings, or None if nothing usable.
    """
    if cs is None or cs.empty:
        return None
    need = {"condition_id", "metric", "mean", "std"}
    if not need.issubset(cs.columns):
        return None

    present = [m for m in prefer if m in cs["metric"].unique()]
    if not present:
        return None

    def _fmt(m, s):
        if pd.isna(m):
            return ""
        if pd.isna(s) or s == 0:
            return f"{float(m):.3g}"
        return f"{float(m):.3g}±{float(s):.2g}"

    rows = []
    conds = sorted(cs["condition_id"].unique())
    for m in present:
        row = {"metric": m}
        sub = cs[cs["metric"] == m]
        for c in conds:
            chunk = sub[sub["condition_id"] == c]
            if chunk.empty:
                row[c] = ""
            else:
                row[c] = _fmt(
                    chunk["mean"].iloc[0],
                    chunk["std"].iloc[0] if "std" in chunk.columns else float("nan"),
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[["metric"] + conds]


# ---------------------------
# Figure 1 – Training speed
# ---------------------------


def _gather_series(condition_roots, pattern):
    """Collect per-run timeseries matching a filename pattern under each condition root.
    Returns dict: {cond_id: [DataFrame_per_run_with_epoch_index, ...]}."""

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


def fig1_training_dynamics(
    condition_roots,
    savepath=None,
    fontsize=12,
    logxA=False,
    logyA=False,
    logxB=False,
    logyB=False,
):
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
        x_plot = (x + 1) if logxA else x
        # slope on the transformed axes requested by flags
        m_slope = _fit_slope(x_plot, m, logx=logxA, logy=logyA)
        label = f"{_short_label_from_root(cond)} (m={m_slope:.3g})"
        axA.plot(x_plot, m, lw=2, label=label)
        axA.fill_between(x_plot, m - s, m + s, alpha=0.2)
    _style(
        axA, fontsize, title="Loss vs epoch", xlabel="epoch", ylabel="loss", legend=True
    )
    if logxA:
        axA.set_xscale("log")
    if logyA:
        axA.set_yscale("log")

    # Panel B: grad L2 (post)
    for cond, runs in G.items():
        x, m, s = _mean_sem_align(runs, "grad_L2_post")
        if x is None:
            continue
        x_plot = (x + 1) if logxB else x
        m_slope = _fit_slope(x_plot, m, logx=logxB, logy=logyB)
        label = f"{_short_label_from_root(cond)} (m={m_slope:.3g})"
        axB.plot(x_plot, m, lw=2, label=label)
        axB.fill_between(x_plot, m - s, m + s, alpha=0.2)
    _style(
        axB,
        fontsize,
        title="Grad L2 (post) vs epoch",
        xlabel="epoch",
        ylabel="‖∇‖₂",
        legend=True,
    )
    if logxB:
        axB.set_xscale("log")
    if logyB:
        axB.set_yscale("log")

    # Panel C: spectral radius
    for cond, runs in W.items():
        x, m, s = _mean_sem_align(runs, "spectral_radius")
        if x is None:
            continue
        axC.plot(x, m, lw=2, label=_short_label_from_root(cond))
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
        axD.plot(x, m, lw=2, label=_short_label_from_root(cond))
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


def fig2_symmetry_vs_performance(
    condition_df, run_level_df, savepath=None, fontsize=12
):
    """
    Figure 2 (2x2):
      A: Best training loss (↓) vs α0            [condition_summary: 'best_loss']
      B: Best open-loop MSE (↓) vs α0            [condition_summary: 'mse_open']
      C: Best closed-loop MSE (↓) vs α0          [condition_summary: 'mse_free_closed' (preferred)]
      D: alpha(t) over training (median ± IQR)   [run_level.csv: fro_S_offline,fro_A_offline]
    """
    # -------------------------
    # Prepare Figure + axes
    # -------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    (axA, axB), (axC, axD) = axs

    # Capture family→color chosen by MPL in A–C so D reuses them
    FAMILY_COLOR = {}  # e.g., {"shift": "#1f77b4", ...}
    FAMILY_ORDER = []  # keep insertion order for a stable color legend

    def _record_color_from_line(fam: str, line_obj):
        if fam not in FAMILY_COLOR and line_obj is not None:
            c = getattr(line_obj, "get_color", lambda: None)()
            if c:
                FAMILY_COLOR[fam] = c
                FAMILY_ORDER.append(fam)

    # -------------------------
    # Panels A, B, C  — condition_summary
    # -------------------------
    if condition_df is None or not {"condition_id", "metric", "mean"}.issubset(
        condition_df.columns
    ):
        print(
            "[SKIP] fig2 A–C: condition_summary missing columns 'condition_id','metric','mean'"
        )
        for ax in (axA, axB, axC):
            ax.axis("off")
    else:
        df = condition_df.copy()
        df["shortname"] = df["condition_id"].apply(
            lambda cid: os.path.basename(str(cid)).split("_")[0]
        )
        families = sorted(df["shortname"].unique().tolist())
        single_family = families[0] if len(families) == 1 else None

        plotted_any = {"A": False, "B": False, "C": False}
        for fam in families:
            sub = df[df["shortname"] == fam]
            csw_mean = _cs_to_wide(sub, value_col="mean")
            csw_std = _cs_to_wide(sub, value_col="std")
            if csw_mean is None or csw_mean.empty:
                continue

            # alpha0 from condition_id
            csw_mean["alpha"] = csw_mean["condition_id"].apply(_alpha_from_condition_id)
            csw_mean = csw_mean.dropna(subset=["alpha"]).sort_values("alpha")
            if csw_std is not None and "condition_id" in csw_std.columns:
                csw_std["alpha"] = csw_std["condition_id"].apply(
                    _alpha_from_condition_id
                )
                csw_std = (
                    csw_std.set_index("condition_id")
                    .reindex(csw_mean["condition_id"])
                    .reset_index()
                )

            # Panel A: best_loss vs alpha0
            if "best_loss" in csw_mean.columns:
                x = csw_mean["alpha"].values
                y = csw_mean["best_loss"].values
                if fam in FAMILY_COLOR:
                    color = FAMILY_COLOR[fam]
                    if csw_std is not None and "best_loss" in csw_std.columns:
                        yerr = (
                            csw_std.set_index("condition_id")["best_loss"]
                            .reindex(csw_mean["condition_id"])
                            .values
                        )
                        cont = axA.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            color=color,
                            ecolor=color,
                            label=fam,
                        )
                        _record_color_from_line(
                            fam,
                            cont.lines[0]
                            if hasattr(cont, "lines") and cont.lines
                            else None,
                        )
                    else:
                        (line,) = axA.plot(
                            x, y, "-o", linewidth=1.6, color=color, label=fam
                        )
                        _record_color_from_line(fam, line)
                else:
                    # let MPL pick color, then record it
                    if csw_std is not None and "best_loss" in csw_std.columns:
                        yerr = (
                            csw_std.set_index("condition_id")["best_loss"]
                            .reindex(csw_mean["condition_id"])
                            .values
                        )
                        cont = axA.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            label=fam,
                        )
                        _record_color_from_line(
                            fam,
                            cont.lines[0]
                            if hasattr(cont, "lines") and cont.lines
                            else None,
                        )
                    else:
                        (line,) = axA.plot(x, y, "-o", linewidth=1.6, label=fam)
                        _record_color_from_line(fam, line)
                plotted_any["A"] = True
            else:
                axA.axis("off")

            # Panel B: mse_open vs alpha0
            if "mse_open" in csw_mean.columns:
                x = csw_mean["alpha"].values
                y = csw_mean["mse_open"].values
                color = FAMILY_COLOR.get(fam, None)
                if csw_std is not None and "mse_open" in csw_std.columns:
                    yerr = (
                        csw_std.set_index("condition_id")["mse_open"]
                        .reindex(csw_mean["condition_id"])
                        .values
                    )
                    cont = (
                        axB.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            color=color,
                            ecolor=color,
                            label=fam,
                        )
                        if color
                        else axB.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            label=fam,
                        )
                    )
                    _record_color_from_line(
                        fam,
                        cont.lines[0]
                        if hasattr(cont, "lines") and cont.lines
                        else None,
                    )
                else:
                    (line,) = (
                        axB.plot(x, y, "-o", linewidth=1.6, color=color, label=fam)
                        if color
                        else axB.plot(x, y, "-o", linewidth=1.6, label=fam)
                    )
                    _record_color_from_line(fam, line)
                plotted_any["B"] = True
            else:
                axB.axis("off")

            # Panel C: CLOSED — strict 'mse_free_closed'
            if "mse_free_closed" in csw_mean.columns:
                x = csw_mean["alpha"].values
                y = csw_mean["mse_free_closed"].values
                color = FAMILY_COLOR.get(fam, None)
                if csw_std is not None and "mse_free_closed" in csw_std.columns:
                    yerr = (
                        csw_std.set_index("condition_id")["mse_free_closed"]
                        .reindex(csw_mean["condition_id"])
                        .values
                    )
                    cont = (
                        axC.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            color=color,
                            ecolor=color,
                            label=fam,
                        )
                        if color
                        else axC.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            capsize=3,
                            linewidth=1.6,
                            label=fam,
                        )
                    )
                    _record_color_from_line(
                        fam,
                        cont.lines[0]
                        if hasattr(cont, "lines") and cont.lines
                        else None,
                    )
                else:
                    (line,) = (
                        axC.plot(x, y, "-o", linewidth=1.6, color=color, label=fam)
                        if color
                        else axC.plot(x, y, "-o", linewidth=1.6, label=fam)
                    )
                    _record_color_from_line(fam, line)
                plotted_any["C"] = True
            else:
                # Strict behavior: do not substitute any other column
                if not getattr(fig, "_printed_missing_mse_free_closed", False):
                    print(
                        "[SKIP] fig2C: column 'mse_free_closed' not found in condition_summary; skipping Panel C."
                    )
                    setattr(fig, "_printed_missing_mse_free_closed", True)
                axC.axis("off")

        # legends if anything plotted
        for ax_key, ax in zip(("A", "B", "C"), (axA, axB, axC)):
            if plotted_any[ax_key]:
                ax.legend(frameon=False, fontsize=max(8, fontsize - 2))

        if any(plotted_any.values()):
            if single_family:
                fig.suptitle(
                    f"Figure 2 — Performance vs α₀ (condition: {single_family})",
                    fontsize=fontsize + 2,
                )
                fig.subplots_adjust(top=0.90)
            else:
                fig.suptitle("Figure 2 — Performance vs α₀", fontsize=fontsize + 2)
                fig.subplots_adjust(top=0.90)

    # -------------------------
    # Panel D — alpha(t) from run_level.csv (reuse A–C colors; α0 -> linestyle)
    # -------------------------
    if run_level_df is None or run_level_df.empty:
        axD.axis("off")
        axD.set_title("D — alpha(t): no run_level.csv", fontsize=fontsize)
    else:
        df_rl = run_level_df.copy()
        if "condition_id" not in df_rl.columns:
            if "condition_root" in df_rl.columns:
                df_rl["condition_id"] = df_rl["condition_root"].apply(
                    _infer_condition_id_from_root
                )
            else:
                df_rl["condition_id"] = "condition"

        have_cols = {"fro_S_offline", "fro_A_offline"} <= set(df_rl.columns)
        tcol = _pick_time_col_runlevel(df_rl) if have_cols else None

        if not have_cols or tcol is None:
            axD.axis("off")
            axD.set_title(
                "D — alpha(t): missing fro_S_offline/fro_A_offline or time",
                fontsize=fontsize,
            )
        else:
            # derive family & alpha0 per condition (family must match A–C parsing logic)
            df_rl["family"] = df_rl["condition_id"].apply(
                lambda cid: os.path.basename(str(cid)).split("_")[0]
            )
            df_rl["alpha0"] = df_rl["condition_id"].apply(_alpha_from_condition_id)

            # Build linestyle map from α0 levels present (stable)
            ls_map, alpha_levels = _linestyle_map_for_alphas(df_rl["alpha0"].unique())

            # thin per-run traces (color by family from A–C; linestyle by α0)
            group_keys = ["condition_id"] + (
                ["run_id"] if "run_id" in df_rl.columns else []
            )
            for _, g in df_rl.groupby(group_keys):
                tcol_g, t, a = _alpha_series_from_runlevel(g)
                if tcol_g is None:
                    continue
                order = np.argsort(t)
                fam = g["family"].iloc[0]
                a0 = (
                    float(g["alpha0"].iloc[0])
                    if pd.notna(g["alpha0"].iloc[0])
                    else None
                )
                color = FAMILY_COLOR.get(fam, None) or "k"
                lstyle = ls_map.get(a0, "-")
                axD.plot(
                    t[order],
                    a[order],
                    lw=1.0,
                    alpha=0.30,
                    color=color,
                    linestyle=lstyle,
                )

            # per-condition median + IQR (same color & linestyle)
            for cond, cdf in df_rl.groupby("condition_id"):
                fam = cdf["family"].iloc[0]
                a0 = (
                    float(cdf["alpha0"].iloc[0])
                    if pd.notna(cdf["alpha0"].iloc[0])
                    else None
                )
                color = FAMILY_COLOR.get(fam, None) or "k"
                lstyle = ls_map.get(a0, "-")

                tcol_c, _, _ = _alpha_series_from_runlevel(cdf)
                if tcol_c is None:
                    continue
                xs = np.sort(
                    pd.to_numeric(cdf[tcol_c], errors="coerce").dropna().unique()
                )
                med, q1, q3 = [], [], []
                for x in xs:
                    sub = cdf[pd.to_numeric(cdf[tcol_c], errors="coerce") == x]
                    _, _, aa = _alpha_series_from_runlevel(sub)
                    aa = aa[np.isfinite(aa)]
                    if aa.size == 0:
                        med.append(np.nan)
                        q1.append(np.nan)
                        q3.append(np.nan)
                    else:
                        med.append(np.nanmedian(aa))
                        q1.append(np.nanpercentile(aa, 25))
                        q3.append(np.nanpercentile(aa, 75))

                xs = xs.astype(float)
                axD.plot(
                    xs, med, lw=2.2, color=color, linestyle=lstyle, label=str(cond)
                )
                axD.fill_between(xs, q1, q3, alpha=0.12, color=color)

            _style(
                axD,
                fontsize,
                title="D — alpha(t) from run_level",
                xlabel="epoch",
                ylabel=r"alpha(t) = $\|S\|_F / (\|S\|_F + \|A\|_F)$",
            )

            # Compact dual legend: colors = families (from A–C); styles = α0 levels

            fams_present = [f for f in FAMILY_ORDER if f in FAMILY_COLOR] or sorted(
                df_rl["family"].dropna().unique()
            )
            color_handles = [
                Line2D([0], [0], color=FAMILY_COLOR.get(f, "k"), lw=2.6, label=f)
                for f in fams_present
            ]
            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="k",
                    lw=2.6,
                    linestyle=ls_map[a],
                    label=f"α₀={a:.2f}",
                )
                for a in alpha_levels
            ]

            leg1 = axD.legend(
                handles=color_handles,
                title="Init type",
                frameon=False,
                fontsize=max(7, fontsize - 4),
                title_fontsize=max(8, fontsize - 3),
                loc="upper left",
            )
            axD.add_artist(leg1)
            axD.legend(
                handles=style_handles,
                title="α₀ line style",
                frameon=False,
                fontsize=max(7, fontsize - 4),
                title_fontsize=max(8, fontsize - 3),
                loc="lower left",
            )
    # Add this right after the legends for A/B/C, before Panel D starts
    _style(
        axA,
        fontsize,
        title="A — Best training loss (↓) vs α₀",
        xlabel=r"$\alpha_0$ (symmetry at $t=0$)",
        ylabel="best_loss",
    )
    _style(
        axB,
        fontsize,
        title="B — Best open-loop MSE (↓) vs α₀",
        xlabel=r"$\alpha_0$",
        ylabel="mse_open",
    )
    _style(
        axC,
        fontsize,
        title="C — Best closed-loop MSE (↓) vs α₀ [mse_free_closed]",
        xlabel=r"$\alpha_0$",
        ylabel="mse_free_closed",
    )

    # Layout & save
    try:
        fig.set_constrained_layout(True)
    except Exception:
        fig.tight_layout()
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()


# -----------------------------------------------------------
# Figure 3 – Emergent traveling-waves and replay dynamics
# -----------------------------------------------------------


def fig3_traveling_waves_and_replay(
    condition_dir: str,
    run_level_df: Optional[pd.DataFrame],
    condition_summary_df: Optional[pd.DataFrame],
    savepath: Optional[str] = None,
    fontsize: int = 12,
    max_units: int = 100,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Top-Left  (A): Replay hidden heatmap (z-scored, units sorted by peak time)
    Top-Right (B): Prediction hidden heatmap (same)
    Bottom-Left (C): Polar angle trajectories (decoded vs true if present)
    Bottom-Right (D): Metrics table from condition_summary (mean±std, only metrics that exist)

    Self-contained: selects the "best" run and loads required files inline:
      *_replay_hidden.npy, *_prediction_hidden.npy,
      *_replay_angles.csv,  *_prediction_angles.csv
    """

    cond = Path(str(condition_dir).strip()).expanduser()

    # --- Locate run_level.csv inside the condition (robust to ./runs vs runs/)
    candidates = [
        cond / "run_level.csv",
        Path("." + str(cond)) / "run_level.csv",
    ]
    candidates += list(cond.glob("*run_level*.csv"))
    rpath = next((p for p in candidates if p.exists()), None)

    print("[fig3] condition_dir:", cond)
    if rpath is None:
        try:
            print("[fig3] ls:", os.listdir(str(cond)))
        except Exception as e:
            print("[fig3] cannot list dir:", e)
        print(f"[SKIP] fig3: no run_level.csv in {cond}")
        return

    # Anchor to the actual CSV parent to avoid path mismatches
    cond = rpath.parent.resolve()

    # --- Read run_level.csv and pick a run (old inline logic, with robust fallbacks)
    df = pd.read_csv(str(rpath))
    if df.empty:
        print(f"[SKIP] fig3: run_level.csv is empty in {cond}")
        return

    def _pick_best_row(frame: pd.DataFrame):
        # Prefer highest replay metric if present; else lowest mse_open; else lowest final_loss
        if "replay_r2" in frame.columns and not frame["replay_r2"].isna().all():
            return frame.sort_values("replay_r2", ascending=False).head(1)
        key = (
            "mse_open"
            if "mse_open" in frame.columns
            else ("final_loss" if "final_loss" in frame.columns else None)
        )
        if key is None or frame[key].isna().all():
            return frame.head(1)
        return frame.sort_values(key, ascending=True).head(1)

    pick = _pick_best_row(df)
    if pick.empty or "run_id" not in pick.columns:
        print(f"[SKIP] fig3: cannot identify run_id from run_level.csv")
        return
    try:
        run_id = int(pick.iloc[0]["run_id"])
    except Exception:
        print(f"[SKIP] fig3: invalid run_id value: {pick.iloc[0]['run_id']!r}")
        return
    run_id = int(pick.iloc[0]["run_id"])

    # --- Resolve checkpoint stub to find trace/angle files
    run_dir = cond / f"run_{run_id:02d}"
    ckpts = list(run_dir.glob("*.pth.tar"))
    if not ckpts:
        print(f"[SKIP] fig3: no checkpoint in {run_dir}")
        return
    ckpt_path = str(ckpts[0])
    stub = (
        ckpt_path[:-8]
        if ckpt_path.endswith(".pth.tar")
        else str(Path(ckpt_path).with_suffix(""))
    )

    # --- Load hidden traces (if present); accept [B,T,N] or [T,N]; cap N
    def _to_TN(arr):
        if arr is None:
            return None
        A = np.asarray(arr)
        if A.ndim == 3:  # [B,T,N]
            A = A[0]
        return A if A.ndim == 2 else None

    def _load_hidden_if(path):
        return _to_TN(np.load(path)) if os.path.exists(path) else None

    rp_h = _load_hidden_if(stub + "_replay_hidden.npy")
    pr_h = _load_hidden_if(stub + "_prediction_hidden.npy")

    if rp_h is None and pr_h is None:
        print(f"[SKIP] fig3: no hidden traces in {run_dir}")
        return
    if rp_h is not None and rp_h.shape[1] > max_units:
        rp_h = rp_h[:, :max_units]
    if pr_h is not None and pr_h.shape[1] > max_units:
        pr_h = pr_h[:, :max_units]

    # --- Z-score per unit and sort by time-of-peak (wave reveal)
    def _zsort(H):
        if H is None:
            return None, None
        Z = (H - H.mean(axis=0, keepdims=True)) / (H.std(axis=0, keepdims=True) + 1e-8)
        peak_t = np.argmax(Z, axis=0)
        order = np.argsort(peak_t)
        return Z[:, order].T, order  # [units,time], permutation

    rp_z, _ = _zsort(rp_h)
    pr_z, _ = _zsort(pr_h)

    # --- Angle CSV reading (robust to simple csv)
    def _read_angles_csv(path):
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            # fallback: bare csv reader
            with open(path, "r") as f:
                rows = list(csv.reader(f))
            if not rows:
                return None
            hdr = rows[0]
            vals = rows[1:]
            try:
                arr = np.array(vals, dtype=float)
                return pd.DataFrame(arr, columns=hdr[: arr.shape[1]])
            except Exception:
                return None

    rp_ang = _read_angles_csv(stub + "_replay_angles.csv")
    pr_ang = _read_angles_csv(stub + "_prediction_angles.csv")

    def _first_angle_col(df):
        if df is None or df.empty:
            return None
        prefer = ["theta_pred", "theta_true", "theta", "angle", "phi", "ang", "phase"]
        lower = {c.lower(): c for c in df.columns}
        for k in prefer:
            if k in lower:
                return lower[k]
        # fallback: first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    # --- Build figure (2x2)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    axA = fig.add_subplot(gs[0, 0])  # replay heatmap
    axB = fig.add_subplot(gs[0, 1])  # prediction heatmap
    axC = fig.add_subplot(gs[1, 0], projection="polar")  # polar
    axD = fig.add_subplot(gs[1, 1])  # table

    plt.rcParams.update({"font.size": fontsize})

    # --- Panel A: replay heatmap
    def _heatmap(ax, Z, title):
        if Z is None:
            ax.axis("off")
            ax.set_title(f"{title} (missing)", fontsize=fontsize)
            return
        vmin_eff = np.min(Z) if vmin is None else vmin
        vmax_eff = np.max(Z) if vmax is None else vmax
        im = ax.imshow(
            Z,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("time")
        ax.set_ylabel("units (sorted)")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("activation (z)", rotation=270, labelpad=15)
        cb.ax.tick_params(labelsize=max(8, fontsize - 3))

    _heatmap(axA, rp_z, "A — Replay hidden (z; peak-sorted)")
    _heatmap(axB, pr_z, "B — Prediction hidden (z; peak-sorted)")

    # --- Panel C: polar trajectories (decoded vs true if available)
    def _plot_polar_angles(ax, df, label, style="-"):
        if df is None or df.empty:
            return None
        t = df["t"].values if "t" in df.columns else np.arange(len(df))
        # prefer explicit decoded/true; else pick first numeric column as angle
        a_pred_col = (
            "theta_pred" if "theta_pred" in df.columns else _first_angle_col(df)
        )
        if a_pred_col is None:
            return None
        (ln,) = ax.plot(
            df[a_pred_col].values, t, linestyle=style, linewidth=1.8, label=label
        )
        # overlay true in same hue if present
        if "theta_true" in df.columns:
            ax.plot(
                df["theta_true"].values,
                t,
                linestyle="--",
                linewidth=1.6,
                label=f"{label} (true)",
                color=ln.get_color(),
            )
        return ln

    any_line = False
    if _plot_polar_angles(axC, rp_ang, "Replay θ̂(t)", "-") is not None:
        any_line = True
    if _plot_polar_angles(axC, pr_ang, "Prediction θ̂(t)", "-.") is not None:
        any_line = True

    if any_line:
        axC.set_title("C — Polar angle trajectories", fontsize=fontsize)
        axC.legend(
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            frameon=False,
            fontsize=max(8, fontsize - 2),
        )
    else:
        axC.set_title("C — Polar angle trajectories (missing)", fontsize=fontsize)

    # --- Panel D: compact metrics table (reuse your existing builder)
    axD.axis("off")
    prefer_metrics = [
        # REPLAY
        "replay_r2",
        "ring_decode_R2_replay",
        "angle_error_R_replay",
        "phase_drift_per_step_replay",
        "time_to_divergence_replay",
        "residual_lag1_autocorr_replay",
        # PREDICTION
        "ring_decode_R2",
        "mse",
        "angle_error_R",
        "mean_corr",
        "time_to_divergence",
    ]
    # Filter to this condition where possible
    cs_sub = condition_summary_df
    if cs_sub is not None and "condition_id" in cs_sub.columns:
        cid = _infer_condition_id_from_root(str(cond))
        cs_sub = cs_sub[cs_sub["condition_id"] == cid]
    table_df = _build_metrics_table_from_condition_summary(cs_sub, prefer_metrics)

    if table_df is None or table_df.empty:
        axD.set_title(
            "D — Metrics table (no matching metrics found)", fontsize=fontsize
        )
    else:
        axD.set_title("D — Replay & Prediction Metrics (mean±std)", fontsize=fontsize)
        tbl = axD.table(
            cellText=table_df.values, colLabels=table_df.columns, loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(max(7, fontsize - 5))
        tbl.scale(1.0, 1.2)
        # bold header
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")

    fig.suptitle(
        f"Fig 3 — Waves & Polar (cond: {cond.name}, run: {run_id})",
        fontsize=fontsize + 2,
    )
    try:
        fig.set_constrained_layout(True)
    except Exception:
        fig.tight_layout()
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()
    return fig


# ==============================
# Figure 4 helpers (checkpoints + cache + heatmap permutation)
# ==============================


def _list_checkpoints(run_dir: str):
    pats = ["*.pth.tar", "*.pt", "*.ckpt"]
    found = []
    for p in pats:
        found.extend(glob.glob(os.path.join(run_dir, p)))

    def _key(path):
        m = re.search(r"epoch[_\-]?(\d+)", os.path.basename(path))
        return (0, int(m.group(1))) if m else (1, 10**12)

    return sorted(found, key=_key)


def _load_W_history_from_ckpt(ckpt_path: str):
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[SKIP] fig4: failed to load ckpt {ckpt_path}: {e}")
        return None, None
    weights = ckpt.get("weights", {})
    hist = weights.get("W_hh_history", None)
    epochs = ckpt.get("snapshot_epochs", None)
    if hist is None or epochs is None:
        return None, None
    Ws = []
    for w in hist:
        if hasattr(w, "detach"):
            w = w.detach().cpu().numpy()
        else:
            w = np.asarray(w)
        if w.ndim == 2 and w.shape[0] == w.shape[1]:
            Ws.append(w)
    if not Ws:
        return None, None
    return list(epochs), np.stack(Ws, axis=0)  # [T,H,H]


def _select_time_indices_generic(times_list, spec: str):
    if not times_list:
        return []
    want = [s.strip().lower() for s in (spec or "last").split(",")]
    idxs, n = set(), len(times_list)
    for w in want:
        if w == "all":
            return list(range(n))
        if w == "first":
            idxs.add(0)
        elif w == "middle":
            idxs.add(n // 2)
        elif w == "last":
            idxs.add(n - 1)
        else:
            try:
                tnum = int(w)
                if tnum in times_list:
                    idxs.add(times_list.index(tnum))
                else:
                    arr = np.asarray(times_list, dtype=float)
                    idxs.add(int(np.argmin(np.abs(arr - tnum))))
            except Exception:
                pass
    return sorted(idxs)


def _load_hidden_for_perm(run_dir: str):
    # Prefer replay hidden, then prediction hidden
    cands = []
    cands.extend(glob.glob(os.path.join(run_dir, "*replay*hidden*.npy")))
    cands.extend(glob.glob(os.path.join(run_dir, "*prediction*hidden*.npy")))
    for f in sorted(cands):
        try:
            arr = np.load(f)
            A = np.asarray(arr)
            if A.ndim == 3:  # [B,T,N]
                A = A[0]
            if A.ndim == 2:
                return A  # [T,N]
        except Exception:
            continue
    return None


def _peak_time_sort_perm(H_TN: np.ndarray):
    H = (H_TN - H_TN.mean(axis=0, keepdims=True)) / (
        H_TN.std(axis=0, keepdims=True) + 1e-8
    )
    peak_t = np.argmax(H, axis=0)  # [N]
    return np.argsort(peak_t).astype(int)


def _get_or_make_perm(run_dir: str):
    cache_dir = os.path.join(run_dir, "analysis")
    os.makedirs(cache_dir, exist_ok=True)
    fperm = os.path.join(cache_dir, "perm_peak.npy")
    if os.path.exists(fperm):
        try:
            p = np.load(fperm)
            if p.ndim == 1:
                return p.astype(int)
        except Exception:
            pass
    H = _load_hidden_for_perm(run_dir)
    if H is None:
        print(
            f"[WARN] fig4: no hidden traces found to compute permutation in {run_dir}; leaving W unsorted."
        )
        return None
    perm = _peak_time_sort_perm(H)
    np.save(fperm, perm)
    return perm


def _apply_perm(W: np.ndarray, perm: Optional[np.ndarray]):
    return W if perm is None else W[np.ix_(perm, perm)]


def _ensure_weight_cache(run_dir: str):
    p = os.path.join(run_dir, "analysis", "weights")
    os.makedirs(p, exist_ok=True)
    return p


def _find_or_make_sorted_W_snapshot(run_dir: str, epoch_val: int):
    """
    Returns path to cached sorted W for this run+epoch, creating it from checkpoint if needed.
    Cache file: <run_dir>/analysis/weights/Wsorted_epoch{epoch:06d}.npy
    """
    cache_dir = _ensure_weight_cache(run_dir)
    fout = os.path.join(cache_dir, f"Wsorted_epoch{int(epoch_val):06d}.npy")
    if os.path.exists(fout):
        return fout

    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        print(f"[SKIP] fig4: no checkpoints in {run_dir}")
        return None
    epochs, Ws = None, None
    for c in reversed(ckpts):  # latest first
        epochs, Ws = _load_W_history_from_ckpt(c)
        if epochs is not None:
            break
    if epochs is None:
        print(f"[SKIP] fig4: no W_hh_history in checkpoints for {run_dir}")
        return None

    arr = np.asarray(epochs, dtype=float)
    i = int(np.argmin(np.abs(arr - float(epoch_val))))
    W = Ws[i]
    perm = _get_or_make_perm(run_dir)
    W_sorted = _apply_perm(W, perm)

    try:
        np.save(fout, W_sorted)
    except Exception as e:
        print(f"[WARN] fig4: failed to cache {fout}: {e}")
    return fout


# -----------------------------------------------------------
# Figure 4 – Mean weight trace & eigenspectrum (per condition)
# -----------------------------------------------------------


def fig4_weights_and_spectrum_from_checkpoints(
    condition_root: str,
    time_spec: str = "last",
    savepath: Optional[str] = None,
    fontsize: int = 12,
):
    """
    Rows = selected timepoints; Columns = [Trace mean±sd, Eigenspectrum].
    Uses checkpoints as source of truth, applies SAME unit permutation as heatmaps
    (peak-time on hidden activity), caches sorted W to analysis/weights/, and reuses cache.
    Always overlays a faint eigenvalue cloud from individual runs.
    """
    run_dirs = sorted(glob.glob(os.path.join(condition_root, "run_*")))
    if not run_dirs:
        print("[SKIP] fig4: no run_* under", condition_root)
        return None

    # Discover epochs per run
    per_run_epochs = {}
    for rd in run_dirs:
        ckpts = _list_checkpoints(rd)
        epochs, Ws = (None, None)
        for c in reversed(ckpts):
            epochs, Ws = _load_W_history_from_ckpt(c)
            if epochs is not None:
                break
        if epochs is None:
            print(f"[SKIP] fig4: no W history in {rd}")
            continue
        per_run_epochs[rd] = list(epochs)
    if not per_run_epochs:
        print("[SKIP] fig4: no runs with W history in", condition_root)
        return None

    # Choose row indices using the first run as reference for labeling
    ref_run = next(iter(per_run_epochs))
    row_idxs = _select_time_indices_generic(per_run_epochs[ref_run], time_spec)
    if not row_idxs:
        print(
            "[SKIP] fig4: time_spec matched nothing; available:",
            per_run_epochs[ref_run],
        )
        return None

    fig, axes = plt.subplots(len(row_idxs), 2, figsize=(11, 3.8 * len(row_idxs)))
    if len(row_idxs) == 1:
        axes = np.array([axes])
    plt.rcParams.update({"font.size": fontsize})

    for row_i, idx in enumerate(row_idxs):
        epoch_label = (
            per_run_epochs[ref_run][idx]
            if idx < len(per_run_epochs[ref_run])
            else f"idx{idx}"
        )
        axL, axR = axes[row_i, 0], axes[row_i, 1]

        # Collect cached sorted W for each run at this row’s per-run index
        Ws_sorted = []
        for rd, epochs in per_run_epochs.items():
            if not epochs:
                continue
            use_idx = idx if idx < len(epochs) else (len(epochs) - 1)
            e = epochs[use_idx]
            fW = _find_or_make_sorted_W_snapshot(rd, e)
            if fW is None or not os.path.exists(fW):
                continue
            try:
                W = np.load(fW)
                if W.ndim == 2 and W.shape[0] == W.shape[1]:
                    Ws_sorted.append(W)
            except Exception:
                continue

        if not Ws_sorted:
            print(f"[SKIP] fig4 row {row_i}: no sorted Ws for epoch ~{epoch_label}")
            axL.axis("off")
            axR.axis("off")
            continue

        Ws_sorted = np.stack(Ws_sorted, axis=0)  # [R,H,H]
        Wm = Ws_sorted.mean(axis=0)

        # Left: diagonal-offset trace mean ± sd
        offs = np.arange(-(Wm.shape[0] - 1), Wm.shape[0], dtype=int)
        traces = np.stack(
            [
                np.array([np.trace(Ws_sorted[r], k) for k in offs], dtype=float)
                for r in range(Ws_sorted.shape[0])
            ],
            axis=0,
        )  # [R,K]
        tr_mean = traces.mean(axis=0)
        tr_sd = (
            traces.std(axis=0, ddof=1)
            if traces.shape[0] > 1
            else np.zeros_like(tr_mean)
        )

        axL.plot(offs, tr_mean, lw=2, label="mean")
        if traces.shape[0] > 1:
            axL.fill_between(
                offs, tr_mean - tr_sd, tr_mean + tr_sd, alpha=0.2, label="±1 sd"
            )
        axL.axhline(0, ls="--", lw=1, alpha=0.6, color="k")
        axL.axvline(0, ls=":", lw=1, alpha=0.6, color="k")
        _style(
            axL,
            fontsize,
            title=f"Trace (mean±sd) — epoch {epoch_label}",
            xlabel="Diagonal offset k",
            ylabel="∑ diag_k W",
        )
        axL.legend(frameon=False, fontsize=max(8, fontsize - 2))

        # Right: eigenspectrum of mean(W_sorted) + faint per-run cloud (ALWAYS on)
        eig_mean = np.linalg.eigvals(Wm)
        evs = np.concatenate(
            [np.linalg.eigvals(Ws_sorted[r]) for r in range(Ws_sorted.shape[0])]
        )
        axR.scatter(evs.real, evs.imag, s=6, alpha=0.12, label="runs")
        axR.scatter(eig_mean.real, eig_mean.imag, s=10, alpha=0.9, label="mean(W)")
        th = np.linspace(0, 2 * np.pi, 512)
        axR.plot(
            np.cos(th), np.sin(th), ls="--", lw=1, alpha=0.4, color="k", label="|λ|=1"
        )
        axR.axhline(0, color="k", lw=0.5)
        axR.axvline(0, color="k", lw=0.5)
        axR.set_aspect("equal", adjustable="box")
        rho = float(np.max(np.abs(eig_mean))) if eig_mean.size else float("nan")
        _style(
            axR,
            fontsize,
            title=f"Eigenspectrum — epoch {epoch_label} (ρ≈{rho:.3f})",
            xlabel="Re(λ)",
            ylabel="Im(λ)",
            legend=True,
        )

    try:
        fig.set_constrained_layout(True)
    except Exception:
        fig.tight_layout()
    if savepath:
        _savefig(fig, savepath)
    else:
        plt.show()
    return fig


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
        nargs="+",  # accept 1..N globs
        default=[],
        help="One or more globs for condition roots (space- or comma-separated). Example: --cond_glob './runs/.../sym*/shiftmh_*' './runs/.../sym*/shiftcycmh_*'",
    )
    p.add_argument("--fig1_logxA", action="store_true", help="Fig1A: log x-axis")
    p.add_argument("--fig1_logyA", action="store_true", help="Fig1A: log y-axis")
    p.add_argument("--fig1_logxB", action="store_true", help="Fig1B: log x-axis")
    p.add_argument("--fig1_logyB", action="store_true", help="Fig1B: log y-axis")
    p.add_argument(
        "--figtag",
        type=str,
        default="",
        help="Append this tag to output figure filenames",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for color scale in Figure 3 heatmaps",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for color scale in Figure 3 heatmaps",
    )
    p.add_argument(
        "--fig4_time",
        type=str,
        default="last",
        help="Figure 4 timepoint(s): first|middle|last|all or comma-list (e.g., 'first,last' or '100,500').",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    suffix = f"_{args.figtag}" if args.figtag else ""

    # Build list of condition roots if provided
    cond_roots = []
    if args.conditions:
        cond_roots.extend([s.strip() for s in args.conditions.split(",") if s.strip()])
    if args.cond_glob:
        # args.cond_glob is a list when provided; support comma-separated inside each
        raw_entries = (
            args.cond_glob
            if isinstance(args.cond_glob, (list, tuple))
            else [args.cond_glob]
        )
        patterns = []
        for entry in raw_entries:
            # split on commas but keep simple whitespace-only entries too
            for part in [s.strip() for s in re.split(r",", entry) if s.strip()]:
                patterns.append(part)
        for pat in patterns:
            cond_roots.extend(sorted(glob.glob(pat)))

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
        fig1_path = os.path.join(args.figdir, f"fig1_training_dynamics{suffix}.png")
        fig1_training_dynamics(
            cond_roots if cond_roots else [args.agg_dir],
            fig1_path,
            fontsize=args.fontsize,
            logxA=args.fig1_logxA,
            logyA=args.fig1_logyA,
            logxB=args.fig1_logxB,
            logyB=args.fig1_logyB,
        )
    if want(2):
        fig2_path = os.path.join(
            args.figdir, f"fig2_symmetry_vs_performance{suffix}.png"
        )
        fig2_symmetry_vs_performance(
            condition_df=cs,
            run_level_df=rl,
            savepath=fig2_path,
            fontsize=args.fontsize,
        )
    if want(3):
        roots = cond_roots if cond_roots else [args.agg_dir]
        for root in roots:
            fig3_path = os.path.join(
                args.figdir, f"fig3_{_short_label_from_root(root)}{suffix}.png"
            )
            fig3_traveling_waves_and_replay(
                condition_dir=root,
                run_level_df=rl,
                condition_summary_df=cs,
                savepath=fig3_path,
                fontsize=args.fontsize,
                max_units=100,
                vmin=args.vmin,
                vmax=args.vmax,
            )
    if want(4):
        roots = cond_roots if cond_roots else [args.agg_dir]
        for root in roots:
            short = _short_label_from_root(root)  # e.g., identity_n100_fro
            per_suffix = f"_{args.figtag}_{short}" if args.figtag else f"_{short}"
            fig4_path = os.path.join(
                args.figdir, f"fig4_weights_and_spectrum{per_suffix}.png"
            )
            fig4_weights_and_spectrum_from_checkpoints(
                condition_root=root,
                time_spec=args.fig4_time,
                savepath=fig4_path,
                fontsize=args.fontsize,
            )


if __name__ == "__main__":
    main()
