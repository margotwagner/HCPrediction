"""
Reusable plotting utilities for multirun RNN analyses.

Functions assume the CSV exports created by your AggregateMultiruns.py:
- per_run_metrics.csv
- agg_metrics.csv
- ts_bucketl.pkl

Dependency notes:
- Uses matplotlib + seaborn for convenience
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pickle


# ---------------------------
# Helpers
# ---------------------------
def _ensure_df(df: Optional[pd.DataFrame], path: Optional[Path] = None) -> pd.DataFrame:
    if df is not None:
        return df
    if path is None:
        raise ValueError("Provide a DataFrame or a path to load it.")
    return pd.read_csv(path)


def _load_ts_bucket(ts_bucket_path: Path) -> Dict[Tuple[str, str], Dict[str, List]]:
    with open(ts_bucket_path, "rb") as f:
        return pickle.load(f)


def _get_fontsizes(font_scale: float = 1.4):
    """
    Returns a dict of font sizes for titles, labels, ticks, and legends.
    font_scale multiplies a base size (good defaults for presentation/paper).
    """
    return {
        "title": int(18 * font_scale),
        "label": int(16 * font_scale),
        "ticks": int(14 * font_scale),
        "legend": int(14 * font_scale),
    }


def _filter_by_input_type_df(
    df: pd.DataFrame, input_types: Union[str, List[str], None]
):
    """Return df filtered to one or many input types (or unchanged if None)."""
    if input_types is None:
        return df
    if isinstance(input_types, str):
        input_types = [input_types]
    return df[df["input_type"].isin(input_types)].copy()


def _filter_by_hidden_init_df(
    df: pd.DataFrame, hidden_inits: Union[str, List[str], None]
):
    """Return df filtered to one or many hidden_inits (or unchanged if None)."""
    if hidden_inits is None:
        return df
    if isinstance(hidden_inits, str):
        hidden_inits = [hidden_inits]
    return df[df["hidden_init"].isin(hidden_inits)].copy()


def _cov_ellipse_from_data(x, y, n_std=1.0):
    """
    Returns (width, height, angle_degrees) for an ellipse representing n_std contours
    of the (x,y) covariance (centered at the mean). width/height are the full diameters.
    """
    cov = np.cov(x, y)
    # eigen-decomposition of covariance
    vals, vecs = np.linalg.eigh(cov)
    # sort eigenvalues (largest first)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # angle of the largest eigenvector
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # width/height are 2*n_std*sqrt(eigvals)
    width, height = 2 * n_std * np.sqrt(vals[0]), 2 * n_std * np.sqrt(vals[1])
    return width, height, angle


def _draw_group_ellipses(ax, df, xcol, ycol, color, label, draw_95=True):
    """
    Draws:
      - Mean point (x̄, ȳ)
      - 1σ ellipse (solid edge)
      - ~95% ellipse (≈ 2σ, dashed edge)
    """
    x = df[xcol].values
    y = df[ycol].values
    if len(x) < 2:
        return  # need at least 2 points for covariance

    x_mean, y_mean = np.mean(x), np.mean(y)

    # 1-sigma ellipse
    w1, h1, ang = _cov_ellipse_from_data(x, y, n_std=1.0)
    e1 = Ellipse(
        (x_mean, y_mean),
        width=w1,
        height=h1,
        angle=ang,
        fill=False,
        lw=2,
        ec=color,
        label=f"{label} (1σ)",
    )
    ax.add_patch(e1)

    # ~95% ellipse (~1.96σ). Using 2σ as a simple visual proxy.
    if draw_95:
        w2, h2, ang2 = _cov_ellipse_from_data(x, y, n_std=2.0)
        e2 = Ellipse(
            (x_mean, y_mean),
            width=w2,
            height=h2,
            angle=ang2,
            fill=False,
            lw=1.5,
            ls="--",
            ec=color,
            alpha=0.9,
            label=f"{label} (≈95%)",
        )
        ax.add_patch(e2)

    # Mean marker
    ax.scatter(
        [x_mean],
        [y_mean],
        s=120,
        marker="X",
        c=[color],
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )


def _pad_and_stack_losses(losses: List[List[float]]) -> np.ndarray:
    max_len = max(len(l) for l in losses)
    arr = np.full((len(losses), max_len), np.nan)
    for i, l in enumerate(losses):
        arr[i, : len(l)] = l
    return arr


# ---------------------------
# 1) Boxplot: Final loss distributions per run
# ---------------------------
def plot_final_loss_boxplot(
    per_run_df: Optional[pd.DataFrame] = None,
    per_run_csv: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    hue: str = "input_type",
    title: str = "Final loss distributions across runs",
    font_scale: float = 1.4,
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
) -> None:
    """
    Boxplot of final_loss across runs, grouped by hidden_init and colored by hue (default: input_type).
    """
    df = _ensure_df(per_run_df, per_run_csv)
    df = _filter_by_input_type_df(df, input_types)
    df = _filter_by_hidden_init_df(df, hidden_inits)
    fs = _get_fontsizes(font_scale)

    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df, x="hidden_init", y="final_loss", hue=hue)

    ax.set_title(title, fontsize=fs["title"])
    ax.set_ylabel("Final Loss", fontsize=fs["label"])
    ax.set_xlabel("Hidden Weight Initialization", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    ax.legend(title=hue, fontsize=fs["legend"], title_fontsize=fs["legend"])
    plt.tight_layout()
    plt.show()


# ---------------------------
# 2) Heatmap: Mean best loss by (hidden_init × input_type)
# ---------------------------
def plot_best_loss_heatmap(
    agg_df: Optional[pd.DataFrame] = None,
    agg_csv: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    value_col: str = "best_loss_mean",
    title: str = "Mean Best Loss per Setting",
    cmap: str = "RdYlGn_r",
    annot: bool = True,
    fmt: str = ".3f",
    font_scale: float = 1.4,
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
) -> None:
    """
    Heatmap of an aggregate metric (default: best_loss_mean) across hidden_init × input_type.
    """
    df = _ensure_df(agg_df, agg_csv)
    df = _filter_by_input_type_df(df, input_types)
    df = _filter_by_hidden_init_df(df, hidden_inits)
    fs = _get_fontsizes(font_scale)

    pivot = df.pivot(index="hidden_init", columns="input_type", values=value_col)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap)

    ax.set_title(title, fontsize=fs["title"])
    ax.set_ylabel("Hidden Init", fontsize=fs["label"])
    ax.set_xlabel("Input Type", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    plt.tight_layout()
    plt.show()


# ---------------------------
# 3) Loss curves: mean ± std shading from ts_bucket
# ---------------------------
def plot_loss_curves_mean_std(
    ts_bucket_path: Path,
    settings: Optional[List[Tuple[str, str]]] = None,
    figsize: Tuple[int, int] = (10, 5),
    alpha_band: float = 0.3,
    title: str = "Training Loss Curves (mean ± std)",
    legend: bool = True,
    font_scale: float = 1.4,
) -> None:
    """
    Plot mean ± std training loss curves for one or more (hidden_init, input_type) settings.
    If settings is None, plots all settings in ts_bucket.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    pairs = settings or list(ts_bucket.keys())
    fs = _get_fontsizes(font_scale)

    plt.figure(figsize=figsize)

    for h, it in pairs:
        if (h, it) not in ts_bucket:
            print(f"[WARN] Missing setting {(h, it)} in ts_bucket; skipping.")
            continue
        losses = ts_bucket[(h, it)]["losses"]
        if not losses:
            continue
        arr = _pad_and_stack_losses(losses)
        mean_curve = np.nanmean(arr, axis=0)
        std_curve = np.nanstd(arr, axis=0)
        x = np.arange(len(mean_curve))
        plt.plot(x, mean_curve, label=f"{h} / {it}")
        plt.fill_between(
            x, mean_curve - std_curve, mean_curve + std_curve, alpha=alpha_band
        )

    plt.title(title, fontsize=fs["title"])
    plt.xlabel("Epoch", fontsize=fs["label"])
    plt.ylabel("Loss", fontsize=fs["label"])
    if legend:
        plt.legend(fontsize=fs["legend"], title_fontsize=fs["legend"])
    plt.tight_layout()
    plt.show()


# ---------------------------
# 4) Barplot with error bars: loss_auc_mean ± std per setting
# ---------------------------
def plot_loss_auc_bar_with_std(
    agg_df: Optional[pd.DataFrame] = None,
    agg_csv: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    hue: str = "input_type",
    title: str = "Learning Efficiency: Loss AUC (mean ± std) per Setting",
    font_scale: float = 1.4,
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
) -> None:
    """
    Barplot of loss_auc_mean with ± loss_auc_std error bars, grouped by hidden_init and colored by hue.
    """
    df = _ensure_df(agg_df, agg_csv)
    df = _filter_by_input_type_df(df, input_types)
    df = _filter_by_hidden_init_df(df, hidden_inits)
    fs = _get_fontsizes(font_scale)

    df_plot = df[["hidden_init", "input_type", "loss_auc_mean", "loss_auc_std"]].copy()
    df_plot = df_plot.sort_values(["hidden_init", "input_type"])

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_plot, x="hidden_init", y="loss_auc_mean", hue=hue, ci=None)

    # Manual error bars to reflect *_std
    # Compute positions similar to seaborn's dodge
    categories = df_plot["hidden_init"].unique().tolist()
    hue_levels = df_plot[hue].unique().tolist()
    n_hue = len(hue_levels)
    total_width = 0.8
    step = total_width / n_hue
    start = -(total_width - step) / 2

    for i, hcat in enumerate(categories):
        sub = df_plot[df_plot["hidden_init"] == hcat]
        for j, hlev in enumerate(hue_levels):
            row = sub[sub[hue] == hlev]
            if row.empty:
                continue
            y = float(row["loss_auc_mean"].values[0])
            yerr = float(row["loss_auc_std"].values[0])
            x = i + (start + j * step)
            plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1.5)

    ax.set_title(title, fontsize=fs["title"])
    ax.set_ylabel("Loss AUC (lower is better)", fontsize=fs["label"])
    ax.set_xlabel("Hidden Weight Initialization", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        title=hue,
        fontsize=fs["legend"],
        title_fontsize=fs["legend"],
        bbox_to_anchor=(1.02, 1),
    )
    plt.tight_layout()
    plt.show()


# ---------------------------
# 5) Scatter: speed vs performance (per run)
# ---------------------------
def plot_speed_vs_best_scatter(
    per_run_df: Optional[pd.DataFrame] = None,
    per_run_csv: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "hidden_init",
    style: Optional[str] = "input_type",
    title: str = "Trade-off: Convergence Speed vs. Best Loss",
    font_scale: float = 1.4,
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
) -> None:
    """
    Scatter of time_to_110pct_best (x) vs best_loss (y), colored by hidden_init.
    Optionally style by input_type.
    """
    df = _ensure_df(per_run_df, per_run_csv)
    df = _filter_by_input_type_df(df, input_types)
    df = _filter_by_hidden_init_df(df, hidden_inits)
    df = df.dropna(subset=["time_to_110pct_best", "best_loss"])
    fs = _get_fontsizes(font_scale)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="time_to_110pct_best",
        y="best_loss",
        hue=color,
        style=style,
        s=80,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title(title, fontsize=fs["title"])
    ax.set_xlabel("Time to 110% of best loss (epochs)", fontsize=fs["label"])
    ax.set_ylabel("Best Loss", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    ax.legend(
        fontsize=fs["legend"], title_fontsize=fs["legend"], bbox_to_anchor=(1.02, 1)
    )
    plt.tight_layout()
    plt.show()


# ---------------------------
# 6) Scatter + group ellipses per hidden_init
# ---------------------------
def plot_speed_vs_best_scatter_with_ellipses(
    per_run_df: Optional[pd.DataFrame] = None,
    per_run_csv: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    draw_95: bool = True,
    title: str = "Convergence Speed vs Best Loss (mean ± std ellipses)",
    font_scale: float = 1.4,
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
) -> None:
    """
    Scatter of per-run points plus mean ± std ellipses per hidden_init
    (solid = 1σ, dashed ≈ 95% ~ 2σ). Marker style distinguishes input_type.
    """
    df = _ensure_df(per_run_df, per_run_csv)
    df = _filter_by_input_type_df(df, input_types)
    df = _filter_by_hidden_init_df(df, hidden_inits)
    df = df.dropna(subset=["time_to_110pct_best", "best_loss"])
    fs = _get_fontsizes(font_scale)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="time_to_110pct_best",
        y="best_loss",
        hue="hidden_init",
        style="input_type",
        s=70,
        alpha=0.7,
        ax=ax,
    )

    palette = sns.color_palette(n_colors=df["hidden_init"].nunique())
    hid_inits = df["hidden_init"].unique()
    color_map = {h: palette[i] for i, h in enumerate(hid_inits)}

    for h in hid_inits:
        sub = df[df["hidden_init"] == h]
        if sub.shape[0] < 2:
            continue
        x = sub["time_to_110pct_best"].to_numpy()
        y = sub["best_loss"].to_numpy()
        x_mean, y_mean = x.mean(), y.mean()
        # 1σ
        w1, h1, ang = _cov_ellipse_from_data(x, y, n_std=1.0)
        e1 = Ellipse(
            (x_mean, y_mean),
            width=w1,
            height=h1,
            angle=ang,
            fill=False,
            lw=2,
            ec=color_map[h],
            label=f"{h} (1σ)",
        )
        ax.add_patch(e1)
        # ~95% (≈2σ)
        if draw_95:
            w2, h2, ang2 = _cov_ellipse_from_data(x, y, n_std=2.0)
            e2 = Ellipse(
                (x_mean, y_mean),
                width=w2,
                height=h2,
                angle=ang2,
                fill=False,
                lw=1.5,
                ls="--",
                ec=color_map[h],
                alpha=0.9,
                label=f"{h} (≈95%)",
            )
            ax.add_patch(e2)
        # mean marker
        ax.scatter(
            [x_mean],
            [y_mean],
            s=120,
            marker="X",
            c=[color_map[h]],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_title(title, fontsize=fs["title"])
    ax.set_xlabel("Time to 110% of best loss (epochs)", fontsize=fs["label"])
    ax.set_ylabel("Best Loss", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    ax.legend(
        fontsize=fs["legend"], title_fontsize=fs["legend"], bbox_to_anchor=(1.02, 1)
    )
    plt.tight_layout()
    plt.show()


# ---------------------------
# Radar for a SINGLE input type (overlay per-init, or average across inits)
# ---------------------------
def plot_radar_metrics_for_input_type(
    agg_df: Optional[pd.DataFrame] = None,
    agg_csv: Optional[Path] = None,
    input_type: str = "gaussian",
    hidden_inits: Optional[List[str]] = None,  # subset of inits; None = all available
    metrics: Optional[List[str]] = None,
    overlay_per_init: bool = True,  # True: one polygon per init; False: average across inits -> single polygon
    normalize: str = "minmax",  # {"minmax","zscore",None}
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    show_legend: bool = True,
    radial_ticks: Optional[List[float]] = None,
    font_scale: float = 1.4,
) -> None:
    """
    Radar plot for a SINGLE input_type. Either:
      - overlay one polygon per hidden_init (overlay_per_init=True), or
      - draw a single polygon averaged over the selected hidden_inits (overlay_per_init=False).
    """
    df = _ensure_df(agg_df, agg_csv)
    fs = _get_fontsizes(font_scale)

    # Default metric set (keep only existing)
    default_metrics = [
        "final_loss_mean",
        "best_loss_mean",
        "spectral_norm_mean",
        "orth_err_mean",
        "cond_num_mean",
        "loss_auc_mean",
    ]
    if metrics is None:
        metrics = [m for m in default_metrics if m in df.columns]
    if not metrics:
        raise ValueError(
            "No valid metric columns found to plot. Provide 'metrics=' with existing *_mean columns."
        )

    # Filter rows: single input_type
    sub = df[df["input_type"] == input_type].copy()
    if sub.empty:
        raise ValueError(f"No rows found for input_type='{input_type}' in agg data.")

    # Optional: restrict hidden_inits
    if hidden_inits is not None:
        sub = sub[sub["hidden_init"].isin(hidden_inits)]
        if sub.empty:
            raise ValueError(
                f"No rows for input_type='{input_type}' with hidden_inits={hidden_inits}."
            )

    # Build table: rows = init(s); columns = metrics
    if overlay_per_init:
        plot_rows = (
            sub[["hidden_init"] + metrics]
            .dropna(subset=metrics, how="all")
            .set_index("hidden_init")
            .sort_index()
        )
        if plot_rows.empty:
            raise ValueError("Nothing to plot after filtering.")
    else:
        # Single averaged polygon across the chosen inits
        agg_row = sub[metrics].mean(numeric_only=True)
        plot_rows = pd.DataFrame([agg_row], index=[f"{input_type} (avg over inits)"])

    # ---- Normalize to comparable 0..1 scale (optional) ----
    vals = plot_rows.copy()
    if normalize == "minmax":
        for m in metrics:
            col = vals[m].astype(float)
            vmin, vmax = col.min(), col.max()
            vals[m] = (
                (col - vmin) / (vmax - vmin)
                if np.isfinite(vmax) and vmax > vmin
                else 0.5
            )
        rticks = (
            radial_ticks if radial_ticks is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    elif normalize == "zscore":
        for m in metrics:
            col = vals[m].astype(float)
            mu, sd = col.mean(), col.std(ddof=1)
            if np.isfinite(sd) and sd > 0:
                z = (col - mu) / sd
                vals[m] = np.clip((z + 2.0) / 4.0, 0.0, 1.0)
            else:
                vals[m] = 0.5
        rticks = (
            radial_ticks if radial_ticks is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    else:
        rticks = radial_ticks  # None => matplotlib auto

    # ---- Radar geometry ----
    categories = metrics
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=figsize)

    for row_name, row_vals in vals.iterrows():
        data = row_vals[categories].astype(float).values
        data = np.concatenate([data, data[:1]])
        ax.plot(angles, data, linewidth=2, label=str(row_name))
        ax.fill(angles, data, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    if rticks is not None:
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=9)
        ax.set_ylim(min(rticks), max(rticks) if len(rticks) else 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)

    ttl = title or (
        f"Radar: {input_type} — "
        + ("per init" if overlay_per_init else "avg across inits")
    )
    ax.set_title(ttl, va="bottom", y=1.08)
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    ax.set_xticklabels(categories, fontsize=fs["label"])

    ax.tick_params(axis="y", labelsize=fs["ticks"])
    ax.set_title(ttl, fontsize=fs["title"], y=1.08)
    if show_legend:
        ax.legend(
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            bbox_to_anchor=(1.25, 1.1),
        )

    plt.tight_layout()
    plt.show()


# ---------------------------
# Convenience: single polygon averaged over multiple hidden inits (one input type)
# ---------------------------
def plot_radar_metrics_avg_over_inits(
    agg_df: Optional[pd.DataFrame] = None,
    agg_csv: Optional[Path] = None,
    input_type: str = "gaussian",
    hidden_inits: Optional[List[str]] = None,  # choose which inits to average
    metrics: Optional[List[str]] = None,
    normalize: str = "minmax",
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    show_legend: bool = False,
    radial_ticks: Optional[List[float]] = None,
) -> None:
    """
    Thin wrapper around plot_radar_metrics_for_input_type(..., overlay_per_init=False)
    to draw a SINGLE polygon averaged over the provided hidden_inits for a given input_type.
    """
    return plot_radar_metrics_for_input_type(
        agg_df=agg_df,
        agg_csv=agg_csv,
        input_type=input_type,
        hidden_inits=hidden_inits,
        metrics=metrics,
        overlay_per_init=False,  # <---- single averaged polygon
        normalize=normalize,
        figsize=figsize,
        title=title
        or f"Radar: {input_type} — avg over {len(hidden_inits) if hidden_inits else 'all'} inits",
        show_legend=show_legend,
        radial_ticks=radial_ticks,
    )


# ---------------------------
# Learning curves: mean ± std per (hidden_init, input_type)
# ---------------------------
def _get_available_types(ts_bucket: Dict[Tuple[str, str], Dict[str, list]]):
    inits = sorted({h for (h, _) in ts_bucket.keys()})
    inputs = sorted({it for (_, it) in ts_bucket.keys()})
    return inits, inputs


def _loss_stats_for_pair(ts_bucket, h, it):
    """Return (x, mean_curve, std_curve, runs_matrix or None)."""
    pair = (h, it)
    if pair not in ts_bucket:
        return None
    losses = ts_bucket[pair].get("losses", [])
    if not losses:
        return None
    arr = _pad_and_stack_losses(losses)  # shape: (runs, max_len) with NaNs
    mean_curve = np.nanmean(arr, axis=0)
    std_curve = np.nanstd(arr, axis=0)
    x = np.arange(len(mean_curve))
    return x, mean_curve, std_curve, arr


# ---------------------------
# Learning curves: mean ± std per (hidden_init, input_type) with bigger fonts
# ---------------------------


def plot_learning_curves_for_input_type(
    ts_bucket_path: Path,
    input_type: str,
    hidden_inits: Optional[List[str]] = None,
    show_individual: bool = False,
    alpha_band: float = 0.25,
    alpha_runs: float = 0.15,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    legend_outside: bool = True,
    font_scale: float = 1.4,  # scale factor for fonts (1.0 = default matplotlib)
) -> None:
    """
    Plot learning curves (mean ± std) for a single input_type.
    All fonts scaled up for presentation/paper readability.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    all_inits, all_inputs = _get_available_types(ts_bucket)

    if input_type not in all_inputs:
        raise ValueError(
            f"input_type='{input_type}' not found in ts_bucket. Available: {all_inputs}"
        )

    if hidden_inits is None:
        hidden_inits = all_inits

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for h in hidden_inits:
        stats = _loss_stats_for_pair(ts_bucket, h, input_type)
        if stats is None:
            continue
        x, mean_curve, std_curve, arr = stats

        if show_individual and arr is not None:
            for r in range(arr.shape[0]):
                ax.plot(x, arr[r, :], lw=0.8, alpha=alpha_runs, color="gray")

        ax.plot(x, mean_curve, lw=2, label=h)
        ax.fill_between(
            x, mean_curve - std_curve, mean_curve + std_curve, alpha=alpha_band
        )

    # Font adjustments
    fs_title = int(18 * font_scale)
    fs_label = int(16 * font_scale)
    fs_ticks = int(14 * font_scale)
    fs_legend = int(14 * font_scale)

    ax.set_title(title or f"Learning Curves — {input_type}", fontsize=fs_title)
    ax.set_xlabel("Epoch", fontsize=fs_label)
    ax.set_ylabel("Loss", fontsize=fs_label)
    ax.tick_params(axis="both", labelsize=fs_ticks)
    ax.grid(True, linestyle=":", alpha=0.5)

    if legend_outside:
        ax.legend(
            title="hidden_init",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=fs_legend,
            title_fontsize=fs_legend,
        )
    else:
        ax.legend(fontsize=fs_legend, title_fontsize=fs_legend)

    plt.tight_layout()
    plt.show()


def plot_learning_curves_all_inputs(
    ts_bucket_path: Path,
    input_types: Optional[List[str]] = None,
    hidden_inits: Optional[List[str]] = None,
    show_individual: bool = False,
    alpha_band: float = 0.25,
    alpha_runs: float = 0.12,
    layout: str = "grid",  # {"grid","separate"}
    ncols: int = 2,
    figsize_each: Tuple[int, int] = (10, 6),
    sharey: bool = True,
    font_scale: float = 1.4,
) -> None:
    """
    Produce learning-curve plots (mean ± std) for all input types, with larger fonts.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    all_inits, all_inputs = _get_available_types(ts_bucket)

    if input_types is None:
        input_types = all_inputs

    if hidden_inits is None:
        hidden_inits = all_inits

    fs_title = int(18 * font_scale)
    fs_label = int(16 * font_scale)
    fs_ticks = int(14 * font_scale)
    fs_legend = int(14 * font_scale)

    if layout == "separate":
        for it in input_types:
            plot_learning_curves_for_input_type(
                ts_bucket_path=ts_bucket_path,
                input_type=it,
                hidden_inits=hidden_inits,
                show_individual=show_individual,
                alpha_band=alpha_band,
                alpha_runs=alpha_runs,
                figsize=figsize_each,
                title=f"Learning Curves — {it}",
                legend_outside=True,
                font_scale=font_scale,
            )
        return

    # Grid layout
    n = len(input_types)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_each[0] * ncols * 0.9, figsize_each[1] * nrows * 0.9),
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()

    for idx, it in enumerate(input_types):
        ax = axes[idx]
        ax.set_title(f"{it}", fontsize=fs_title)
        for h in hidden_inits:
            stats = _loss_stats_for_pair(ts_bucket, h, it)
            if stats is None:
                continue
            x, mean_curve, std_curve, arr = stats

            if show_individual and arr is not None:
                for r in range(arr.shape[0]):
                    ax.plot(x, arr[r, :], lw=0.6, alpha=alpha_runs, color="gray")

            ax.plot(x, mean_curve, lw=1.8, label=h)
            ax.fill_between(
                x, mean_curve - std_curve, mean_curve + std_curve, alpha=alpha_band
            )

        ax.set_xlabel("Epoch", fontsize=fs_label)
        ax.set_ylabel("Loss", fontsize=fs_label)
        ax.tick_params(axis="both", labelsize=fs_ticks)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="hidden_init",
            loc="upper center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, 1.05),
            fontsize=fs_legend,
            title_fontsize=fs_legend,
        )

    plt.tight_layout()
    plt.show()
