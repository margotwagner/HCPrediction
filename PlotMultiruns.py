"""
Reusable plotting utilities for multirun RNN analyses.

Functions assume the CSV exports created by your AggregateMultiruns.py:
- per_run_metrics.csv
- agg_metrics.csv
- ts_bucket.pkl

Dependency notes:
- Uses matplotlib + seaborn for convenience
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Iterable
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


def _to_list(x: Union[str, Iterable, None]) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _union_epochs(dfs: List[pd.DataFrame], epoch_col: str) -> np.ndarray:
    """Collect the union of all epoch values present across a list of DataFrames."""
    epochs = set()
    for df in dfs:
        if epoch_col in df.columns:
            epochs.update(df[epoch_col].astype(int).tolist())
    return np.array(sorted(epochs), dtype=int)


def _mean_std_by_epoch(
    dfs: List[pd.DataFrame], metric: str, epoch_col: str = "epoch"
) -> pd.DataFrame:
    """
    Given multiple run-level DataFrames containing (epoch, metric), return a DataFrame with
    columns: epoch, mean, std computed across runs at each epoch (NaNs ignored).
    """
    if not dfs:
        return pd.DataFrame(columns=[epoch_col, "mean", "std"])
    all_epochs = _union_epochs(dfs, epoch_col)
    if all_epochs.size == 0:
        return pd.DataFrame(columns=[epoch_col, "mean", "std"])

    # Build matrix (runs x epochs) aligned on union epochs
    mat = np.full((len(dfs), len(all_epochs)), np.nan, dtype=float)
    for i, df in enumerate(dfs):
        if metric not in df.columns or epoch_col not in df.columns:
            continue
        m = df[[epoch_col, metric]].dropna()
        if m.empty:
            continue
        # map epochs to indices
        idx = np.searchsorted(all_epochs, m[epoch_col].astype(int).values)
        # only keep those that match exact positions (defensive)
        valid = (
            (idx >= 0)
            & (idx < len(all_epochs))
            & (all_epochs[idx] == m[epoch_col].astype(int).values)
        )
        mat[i, idx[valid]] = m[metric].astype(float).values[valid]

    mean_curve = np.nanmean(mat, axis=0)
    std_curve = np.nanstd(mat, axis=0)
    return pd.DataFrame({epoch_col: all_epochs, "mean": mean_curve, "std": std_curve})


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


def plot_metric_trajectories(
    ts_bucket_path: Path,
    metrics: Union[str, List[str]] = ("spectral_radius", "frob", "cond_num"),
    input_types: Union[str, List[str], None] = None,  # filter: encodings to include
    hidden_inits: Union[str, List[str], None] = None,  # filter: inits to include
    show_individual: bool = False,  # faint per-run lines
    show_mean_band: bool = True,  # mean ± std shaded band
    rolling: Optional[int] = None,  # e.g., 5 -> rolling mean window (epochs)
    facet_by: str = "input_type",  # {"input_type", "hidden_init"} which dimension gets separate subplots
    ncols: int = 2,  # grid columns for facets
    figsize_each: Tuple[int, int] = (9, 5),  # size per subplot
    alpha_runs: float = 0.2,
    alpha_band: float = 0.25,
    lw_mean: float = 2.0,
    font_scale: float = 1.4,
    title: Optional[str] = None,
) -> None:
    """
    Plot per-epoch trajectories for selected metrics, optionally filtered by input type(s) and hidden init(s).

    For each facet (e.g., each input_type), this overlays one curve per hidden_init:
      - optional faint individual run lines
      - mean ± std band across runs
      - supports multiple metrics -> draws a separate figure per metric

    Parameters
    ----------
    ts_bucket_path : Path
        Path to ts_bucket_global.pkl
    metrics : str | list[str]
        Which time-series metrics in metrics_df_list to plot (e.g., "spectral_radius", "frob", "cond_num").
    input_types, hidden_inits : str | list[str] | None
        Filters; None = all available.
    show_individual : bool
        Draw every run's trajectory (faint gray) behind the aggregate.
    show_mean_band : bool
        Overlay mean curve and ±1 std shaded band.
    rolling : int | None
        Rolling mean window (in epochs) applied to both individual and aggregate curves.
    facet_by : {"input_type","hidden_init"}
        Which dimension to facet on (subplots). The other dimension becomes the line color.
    ncols : int
        Subplot columns (rows computed automatically).
    figsize_each : tuple
        Size of each subplot (width, height).
    """

    ts_bucket = _load_ts_bucket(ts_bucket_path)
    metrics = _to_list(metrics)
    filt_inputs = _to_list(input_types)
    filt_inits = _to_list(hidden_inits)
    fs = _get_fontsizes(font_scale)

    # Collect available labels
    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})

    use_inputs = [it for it in (filt_inputs or all_inputs) if it in all_inputs]
    use_inits = [h for h in (filt_inits or all_inits) if h in all_inits]

    if facet_by not in ("input_type", "hidden_init"):
        raise ValueError("facet_by must be 'input_type' or 'hidden_init'.")

    # Build a nested dict: data[(hidden_init, input_type)] = list of run metrics DFs
    # Each run metrics DF should include an 'epoch' column and your target metric columns.
    def _runs_for_pair(h, it) -> List[pd.DataFrame]:
        entry = ts_bucket.get((h, it), {})
        mlist = entry.get("metrics_df_list", [])
        # Defensive: ensure DataFrame and has 'epoch'
        out = []
        for df in mlist:
            if isinstance(df, pd.DataFrame) and "epoch" in df.columns and not df.empty:
                out.append(df.copy())
        return out

    # Which categories will be faceted? (rows/cols)
    facets = use_inputs if facet_by == "input_type" else use_inits
    color_dim = "hidden_init" if facet_by == "input_type" else "input_type"
    color_values = use_inits if facet_by == "input_type" else use_inputs

    # Color cycle
    import seaborn as sns

    palette = sns.color_palette(n_colors=len(color_values))
    color_map = {lab: palette[i] for i, lab in enumerate(color_values)}

    for metric in metrics:
        n = len(facets)
        ncols_eff = max(1, ncols)
        nrows = int(np.ceil(n / ncols_eff))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols_eff,
            figsize=(figsize_each[0] * ncols_eff, figsize_each[1] * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for idx, facet in enumerate(facets):
            ax = axes[idx]
            ax.set_title(
                f"{facet_by} = {facet}   •   metric = {metric}", fontsize=fs["title"]
            )

            # For this facet, overlay one line/band per category in color_dim
            for color_label in color_values:
                h, it = (
                    (color_label, facet)
                    if facet_by == "input_type"
                    else (facet, color_label)
                )
                if h not in use_inits or it not in use_inputs:
                    continue

                run_dfs = _runs_for_pair(h, it)
                if not run_dfs:
                    continue

                # Optional smoothing on individual runs
                if rolling and rolling > 1:
                    smoothed = []
                    for df in run_dfs:
                        df2 = df.sort_values("epoch").copy()
                        if metric in df2:
                            df2[metric] = (
                                df2[metric]
                                .rolling(rolling, min_periods=1, center=False)
                                .mean()
                            )
                        smoothed.append(df2)
                    run_dfs = smoothed

                # Show individual runs (faint)
                if show_individual:
                    for df in run_dfs:
                        if metric not in df.columns:
                            continue
                        ax.plot(
                            df["epoch"].astype(int).values,
                            df[metric].astype(float).values,
                            lw=0.8,
                            alpha=alpha_runs,
                            color=color_map[color_label],
                        )

                # Aggregate mean ± std over runs at each epoch
                agg = _mean_std_by_epoch(run_dfs, metric, epoch_col="epoch")
                if agg.empty:
                    continue

                if show_mean_band:
                    ax.plot(
                        agg["epoch"].values,
                        agg["mean"].values,
                        lw=lw_mean,
                        color=color_map[color_label],
                        label=str(color_label),
                    )
                    ax.fill_between(
                        agg["epoch"].values,
                        agg["mean"].values - agg["std"].values,
                        agg["mean"].values + agg["std"].values,
                        alpha=alpha_band,
                        color=color_map[color_label],
                    )
                else:
                    ax.plot(
                        agg["epoch"].values,
                        agg["mean"].values,
                        lw=lw_mean,
                        color=color_map[color_label],
                        label=str(color_label),
                    )

            ax.set_xlabel("Epoch", fontsize=fs["label"])
            ax.set_ylabel(metric, fontsize=fs["label"])
            ax.tick_params(axis="both", labelsize=fs["ticks"])
            ax.grid(True, linestyle=":", alpha=0.5)

        # Remove any unused axes
        for j in range(len(facets), len(axes)):
            fig.delaxes(axes[j])

        # Single legend for the figure
        handles, labels = [], []
        for a in axes[: max(1, len(facets))]:
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            fig.legend(
                handles,
                labels,
                title=color_dim,
                fontsize=fs["legend"],
                title_fontsize=fs["legend"],
                loc="upper center",
                ncol=min(4, len(color_values)),
                bbox_to_anchor=(0.5, 1.02),
            )

        fig.suptitle(title or "Metric trajectories over training", fontsize=fs["title"])
        plt.tight_layout()
        plt.show()


def _pick_grad_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first gradient column present in df, given a priority-ordered list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try any column that contains 'grad' and 'l2' and 'mean'
    for c in df.columns:
        lc = c.lower()
        if ("grad" in lc) and ("l2" in lc) and ("mean" in lc):
            return c
    return None


def plot_gradient_dynamics(
    ts_bucket_path: Path,
    input_types: Union[
        str, List[str], None
    ] = None,  # filter encodings (e.g., "gaussian")
    hidden_inits: Union[
        str, List[str], None
    ] = None,  # filter inits (e.g., ["he","orthog"])
    grad_col_candidates: List[str] = None,  # priority list of col names to try
    show_individual: bool = False,  # faint per-run lines
    show_mean_band: bool = True,  # mean ± std shaded band
    rolling: Optional[int] = None,  # e.g., 5 -> rolling mean window (epochs)
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    ncols: int = 2,
    figsize_each: Tuple[int, int] = (9, 5),
    alpha_runs: float = 0.2,
    alpha_band: float = 0.25,
    lw_mean: float = 2.0,
    font_scale: float = 1.6,
    title: Optional[str] = None,
) -> None:
    """
    Gradient dynamics across training:
      - Concatenates grad_df_list across runs (for each (hidden_init, input_type))
      - Plots grad L2 magnitude trajectory over epochs
      - Overlays mean ± std bands across runs
      - Optional per-run traces (faint) and rolling smoothing

    Expected columns in each grad_df (best effort, flexible):
      - 'epoch' (int)
      - Gradient magnitude column, typically 'grad_l2_norm_mean'
        (we try multiple candidates; see `grad_col_candidates`).
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)

    # Defaults for gradient column names (first match wins)
    if grad_col_candidates is None:
        grad_col_candidates = [
            "grad_l2_norm_mean",  # our recommended export
            "grad_l2_norm",  # sometimes used
            "grad_norm_mean",  # fallback naming
            "grad_norm",  # very generic fallback
        ]

    fs = _get_fontsizes(font_scale)

    # Collect available labels
    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})

    use_inputs = [
        it for it in (_to_list(input_types) or all_inputs) if it in all_inputs
    ]
    use_inits = [h for h in (_to_list(hidden_inits) or all_inits) if h in all_inits]

    if facet_by not in ("input_type", "hidden_init"):
        raise ValueError("facet_by must be 'input_type' or 'hidden_init'.")

    # Pull grad_df_list for a pair
    def _runs_grad_for_pair(h: str, it: str) -> List[pd.DataFrame]:
        entry = ts_bucket.get((h, it), {})
        glist = entry.get("grad_df_list", [])
        out = []
        for df in glist:
            if isinstance(df, pd.DataFrame) and "epoch" in df.columns and not df.empty:
                # choose a gradient column and filter to (epoch, grad_col)
                gcol = _pick_grad_col(df, grad_col_candidates)
                if gcol is None:
                    continue
                # keep only (epoch, chosen_grad) and clean Nans
                d = df[["epoch", gcol]].copy()
                d = d.dropna(subset=["epoch", gcol])
                if d.empty:
                    continue
                d.rename(columns={gcol: "grad"}, inplace=True)
                d = d.sort_values("epoch")
                # optional rolling smoothing
                if rolling and rolling > 1:
                    d["grad"] = d["grad"].rolling(rolling, min_periods=1).mean()

                # drop any NaNs introduced at head
                d = d.dropna(subset=["grad"])
                if d.empty:
                    continue
                out.append(d)
        return out

    # Mean ± std across runs aligned by epoch
    def _mean_std_grad_by_epoch(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame(columns=["epoch", "mean", "std"])
        # union of epochs
        epochs = sorted(set(int(e) for df in dfs for e in df["epoch"].values))
        if not epochs:
            return pd.DataFrame(columns=["epoch", "mean", "std"])
        mat = np.full((len(dfs), len(epochs)), np.nan, dtype=float)
        e2i = {e: i for i, e in enumerate(epochs)}
        for r, df in enumerate(dfs):
            for e, v in zip(
                df["epoch"].astype(int).values, df["grad"].astype(float).values
            ):
                if e in e2i:
                    mat[r, e2i[e]] = v
        mean_curve = np.nanmean(mat, axis=0)
        std_curve = np.nanstd(mat, axis=0)
        return pd.DataFrame(
            {"epoch": np.array(epochs, dtype=int), "mean": mean_curve, "std": std_curve}
        )

    # Facets & color dimension
    facets = use_inputs if facet_by == "input_type" else use_inits
    color_dim = "hidden_init" if facet_by == "input_type" else "input_type"
    color_values = use_inits if facet_by == "input_type" else use_inputs

    import seaborn as sns

    palette = sns.color_palette(n_colors=len(color_values))
    color_map = {lab: palette[i] for i, lab in enumerate(color_values)}

    n = len(facets)
    ncols_eff = max(1, ncols)
    nrows = int(np.ceil(n / ncols_eff))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_eff,
        figsize=(figsize_each[0] * ncols_eff, figsize_each[1] * nrows),
        squeeze=False,
    )
    axes = axes.ravel()
    any_plotted = False  # track if anything was actually plotted

    for idx, facet in enumerate(facets):
        ax = axes[idx]
        ax.set_title(
            f"{facet_by} = {facet}   •   gradient = L2 norm (mean over params)",
            fontsize=fs["title"],
        )

        panel_plotted = False

        for color_label in color_values:
            h, it = (
                (color_label, facet)
                if facet_by == "input_type"
                else (facet, color_label)
            )
            if h not in use_inits or it not in use_inputs:
                continue

            run_dfs = _runs_grad_for_pair(h, it)
            if not run_dfs:
                print(
                    f"[INFO] No grad data for (hidden_init={h}, input_type={it}); skipping."
                )
                continue

            # individual runs
            if show_individual:
                for d in run_dfs:
                    ax.plot(
                        d["epoch"].values,
                        d["grad"].values,
                        lw=0.8,
                        alpha=alpha_runs,
                        color=color_map[color_label],
                    )
                    panel_plotted = True
                    any_plotted = True

            # aggregate band
            agg = _mean_std_grad_by_epoch(run_dfs)
            if not agg.empty:
                ax.plot(
                    agg["epoch"].values,
                    agg["mean"].values,
                    lw=lw_mean,
                    color=color_map[color_label],
                    label=str(color_label),
                )

                if show_mean_band:
                    ax.fill_between(
                        agg["epoch"].values,
                        agg["mean"].values - agg["std"].values,
                        agg["mean"].values + agg["std"].values,
                        alpha=alpha_band,
                        color=color_map[color_label],
                    )
                panel_plotted = True
                any_plotted = True

        ax.set_xlabel("Epoch", fontsize=fs["label"])
        ax.set_ylabel("Gradient L2 (per-epoch mean over params)", fontsize=fs["label"])
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.grid(True, linestyle=":", alpha=0.5)

        if not panel_plotted:
            ax.text(
                0.5,
                0.5,
                "No gradient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=fs["ticks"],
                alpha=0.7,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # remove any unused axes
    for j in range(len(facets), len(axes)):
        fig.delaxes(axes[j])

    # figure-level legend
    handles, labels = [], []
    for a in axes[: max(1, len(facets))]:
        h, l = a.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            title=color_dim,
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            loc="upper center",
            ncol=min(4, len(color_values)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(title or "Gradient dynamics over training", fontsize=fs["title"])
    plt.tight_layout()
    plt.show()

    if not any_plotted:
        print(
            "[INFO] No gradient data to plot after cleaning; check RecordEp/sample_batch_idx and aggregator outputs."
        )


def plot_lambda_dynamics(
    ts_bucket_path: Path,
    input_types=None,
    hidden_inits=None,
    facet_by="input_type",
    ncols=2,
    figsize_each=(9, 5),
    font_scale=1.6,
):
    """
    Plot lambda (symmetric vs antisymmetric weight) over training.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    fs = _get_fontsizes(font_scale)

    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})

    use_inputs = [
        it for it in (_to_list(input_types) or all_inputs) if it in all_inputs
    ]
    use_inits = [h for h in (_to_list(hidden_inits) or all_inits) if h in all_inits]

    facets = use_inputs if facet_by == "input_type" else use_inits
    color_dim = "hidden_init" if facet_by == "input_type" else "input_type"
    color_values = use_inits if facet_by == "input_type" else use_inputs

    import seaborn as sns

    palette = sns.color_palette(n_colors=len(color_values))
    color_map = {lab: palette[i] for i, lab in enumerate(color_values)}

    n = len(facets)
    ncols_eff = max(1, ncols)
    nrows = int(np.ceil(n / ncols_eff))
    fig, axes = plt.subplots(
        nrows,
        ncols_eff,
        figsize=(figsize_each[0] * ncols_eff, figsize_each[1] * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for idx, facet in enumerate(facets):
        ax = axes[idx]
        ax.set_title(
            f"{facet_by} = {facet} • λ (sym/antisym mix)", fontsize=fs["title"]
        )

        for color_label in color_values:
            h, it = (
                (color_label, facet)
                if facet_by == "input_type"
                else (facet, color_label)
            )
            entry = ts_bucket.get((h, it), {})
            dfs = entry.get("metrics_df_list", [])
            for df in dfs:
                if "lambda" not in df.columns:
                    continue
                ax.plot(
                    df["epoch"],
                    df["lambda"],
                    lw=2,
                    alpha=0.8,
                    color=color_map[color_label],
                    label=color_label,
                )

        ax.set_xlabel("Epoch", fontsize=fs["label"])
        ax.set_ylabel("λ", fontsize=fs["label"])
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.grid(True, linestyle=":", alpha=0.5)

    # collapse legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title=color_dim,
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            loc="upper center",
            ncol=min(4, len(color_values)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle("Lambda dynamics", fontsize=fs["title"])
    plt.tight_layout()
    plt.show()


def plot_history_metric(
    ts_bucket_path: Path,
    metric: str = "grad_norm",  # e.g., "grad_norm", "loss", "custom_col"
    metric_candidates: Optional[
        List[str]
    ] = None,  # priority fallback names, if metric not found
    input_types: Union[
        str, List[str], None
    ] = None,  # filter encodings (e.g., "gaussian")
    hidden_inits: Union[
        str, List[str], None
    ] = None,  # filter inits (e.g., ["he","orthog"])
    show_individual: bool = False,  # faint per-run lines
    show_mean_band: bool = True,  # mean ± std shaded band
    rolling: Optional[int] = None,  # e.g., 5 -> rolling mean window (epochs)
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    ncols: int = 2,
    figsize_each: Tuple[int, int] = (9, 5),
    alpha_runs: float = 0.2,
    alpha_band: float = 0.25,
    lw_mean: float = 2.0,
    font_scale: float = 1.6,
    title: Optional[str] = None,
    y_log: bool = False,  # log-scale y-axis (useful for exploding/vanishing)
) -> None:
    """
    Plot a history metric (default: grad_norm) vs epoch using history_df_list for the selected settings.

    Looks for the requested `metric` column in each run's history DataFrame.
    If not found and `metric_candidates` is provided, tries those names in order.

    Facets by `input_type` (color by hidden_init) or vice versa.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    fs = _get_fontsizes(font_scale)

    # Default fallbacks: if user asks for "grad_norm" or "loss", try common variations
    default_candidates = {
        "grad_norm": [
            "grad_norm",
            "grad_l2_norm",
            "grad_norm_mean",
            "grad_l2_norm_mean",
        ],
        "loss": ["loss", "train_loss", "epoch_loss", "loss_batch_mean"],
    }
    if metric_candidates is None:
        metric_candidates = default_candidates.get(metric, [metric])

    def _norm_list(x: Union[str, Iterable, None]) -> Optional[List[str]]:
        if x is None:
            return None
        if isinstance(x, str):
            return [x]
        return list(x)

    def _pick_col(
        df: pd.DataFrame, target: str, candidates: List[str]
    ) -> Optional[str]:
        # exact matches first
        if target in df.columns:
            return target
        for c in candidates:
            if c in df.columns:
                return c
        # fuzzy fallback: any column that contains target substring
        target_l = target.lower()
        for c in df.columns:
            if target_l in c.lower():
                return c
        return None

    def _runs_hist_for_pair(h: str, it: str) -> List[pd.DataFrame]:
        """Return a list of DataFrames with columns ['epoch','val'] for a (hidden_init,input_type) pair."""
        entry = ts_bucket.get((h, it), {})
        hlist = entry.get("history_df_list", [])
        out = []
        for df in hlist:
            if not (
                isinstance(df, pd.DataFrame) and "epoch" in df.columns and not df.empty
            ):
                continue
            mcol = _pick_col(df, metric, metric_candidates)
            if mcol is None:
                continue
            d = df[["epoch", mcol]].copy().sort_values("epoch")
            d.rename(columns={mcol: "val"}, inplace=True)
            if rolling and rolling > 1:
                d["val"] = d["val"].rolling(rolling, min_periods=1).mean()
            out.append(d)
        return out

    def _mean_std_by_epoch(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Align on union of epochs; return DataFrame with columns ['epoch','mean','std']."""
        if not dfs:
            return pd.DataFrame(columns=["epoch", "mean", "std"])
        epochs = sorted(set(int(e) for df in dfs for e in df["epoch"].values))
        if not epochs:
            return pd.DataFrame(columns=["epoch", "mean", "std"])
        mat = np.full((len(dfs), len(epochs)), np.nan, dtype=float)
        e2i = {e: i for i, e in enumerate(epochs)}
        for r, df in enumerate(dfs):
            ev = df["epoch"].astype(int).values
            vv = df["val"].astype(float).values
            for e, v in zip(ev, vv):
                j = e2i.get(e)
                if j is not None:
                    mat[r, j] = v
        return pd.DataFrame(
            {
                "epoch": np.array(epochs, dtype=int),
                "mean": np.nanmean(mat, axis=0),
                "std": np.nanstd(mat, axis=0),
            }
        )

    # Available labels
    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})

    use_inputs = [
        it for it in (_norm_list(input_types) or all_inputs) if it in all_inputs
    ]
    use_inits = [h for h in (_norm_list(hidden_inits) or all_inits) if h in all_inits]

    if facet_by not in ("input_type", "hidden_init"):
        raise ValueError("facet_by must be 'input_type' or 'hidden_init'.")

    # Facets & color mapping
    facets = use_inputs if facet_by == "input_type" else use_inits
    color_dim = "hidden_init" if facet_by == "input_type" else "input_type"
    color_values = use_inits if facet_by == "input_type" else use_inputs

    import seaborn as sns

    palette = sns.color_palette(n_colors=len(color_values))
    color_map = {lab: palette[i] for i, lab in enumerate(color_values)}

    # Figure layout
    n = len(facets)
    ncols_eff = max(1, ncols)
    nrows = int(np.ceil(n / ncols_eff))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_eff,
        figsize=(figsize_each[0] * ncols_eff, figsize_each[1] * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for idx, facet in enumerate(facets):
        ax = axes[idx]
        ax.set_title(
            f"{facet_by} = {facet}   •   {metric} vs epoch", fontsize=fs["title"]
        )

        for color_label in color_values:
            h, it = (
                (color_label, facet)
                if facet_by == "input_type"
                else (facet, color_label)
            )
            if h not in use_inits or it not in use_inputs:
                continue

            runs = _runs_hist_for_pair(h, it)
            if not runs:
                continue

            # Optional: individual runs
            if show_individual:
                for d in runs:
                    ax.plot(
                        d["epoch"].values,
                        d["val"].values,
                        lw=0.8,
                        alpha=alpha_runs,
                        color=color_map[color_label],
                    )

            # Aggregate mean ± std band
            agg = _mean_std_by_epoch(runs)
            if not agg.empty and show_mean_band:
                ax.plot(
                    agg["epoch"].values,
                    agg["mean"].values,
                    lw=lw_mean,
                    color=color_map[color_label],
                    label=str(color_label),
                )
                ax.fill_between(
                    agg["epoch"].values,
                    agg["mean"].values - agg["std"].values,
                    agg["mean"].values + agg["std"].values,
                    alpha=alpha_band,
                    color=color_map[color_label],
                )
            elif not agg.empty:
                ax.plot(
                    agg["epoch"].values,
                    agg["mean"].values,
                    lw=lw_mean,
                    color=color_map[color_label],
                    label=str(color_label),
                )

        ax.set_xlabel("Epoch", fontsize=fs["label"])
        ax.set_ylabel(metric, fontsize=fs["label"])
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.grid(True, linestyle=":", alpha=0.5)
        if y_log:
            ax.set_yscale("log")

    # Remove unused axes
    for j in range(len(facets), len(axes)):
        fig.delaxes(axes[j])

    # Figure-level legend
    handles, labels = [], []
    for a in axes[: max(1, len(facets))]:
        hnd, lbl = a.get_legend_handles_labels()
        if hnd:
            handles, labels = hnd, lbl
            break
    if handles:
        fig.legend(
            handles,
            labels,
            title=color_dim,
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            loc="upper center",
            ncol=min(4, len(color_values)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        title or f"History curves: {metric} over training", fontsize=fs["title"]
    )
    plt.tight_layout()
    plt.show()


def _first_epoch_within_threshold(
    loss: List[float], mult: float = 1.10
) -> Optional[int]:
    """
    Return the first epoch index where loss <= mult * best_loss for this run.
    If never reached, return None (censored).
    """
    if not loss:
        return None
    arr = np.asarray(loss, dtype=float)
    best = np.nanmin(arr)
    thr = mult * best
    hits = np.where(arr <= thr)[0]
    return int(hits[0]) if hits.size > 0 else None


def _ecdf_over_time(times: List[Optional[int]], max_epoch: int) -> np.ndarray:
    """
    Given per-run 'times to threshold' (epoch indices, or None if never),
    return cumulative fraction achieved by each epoch 0..max_epoch (inclusive).
    Denominator = total number of runs (so plateaus < 1.0 if some never reach).
    """
    n = len(times)
    if n == 0:
        return np.zeros(max_epoch + 1, dtype=float)
    finite = np.array([t for t in times if t is not None], dtype=float)
    # counts of successes by epoch
    counts = np.zeros(max_epoch + 1, dtype=int)
    for t in finite:
        if 0 <= t <= max_epoch:
            counts[int(t) :] += 1  # after first hit, remain “converged”
    return counts / float(n)


def plot_convergence_speed_curves(
    ts_bucket_path: Path,
    input_types: Union[str, List[str], None] = None,  # filters
    hidden_inits: Union[str, List[str], None] = None,
    threshold_mult: float = 1.10,  # k in “within k × best”
    aggregate_over_inputs: bool = True,  # True: 1 panel, per-init curves aggregating selected inputs
    # False: facet by input_type; each panel shows per-init curves
    show_quantiles: Optional[List[float]] = (
        0.5,
    ),  # e.g., [0.5, 0.8]; None/[] to disable
    rolling: Optional[int] = None,  # smooth ECDF with rolling mean over epochs
    percent: bool = True,  # y-axis as percent (0..100)
    figsize: Tuple[int, int] = (11, 7),
    ncols: int = 2,  # used when aggregate_over_inputs=False (facets)
    alpha_curves: float = 0.95,
    lw: float = 2.2,
    font_scale: float = 1.6,
    title: Optional[str] = None,
) -> None:
    """
    Convergence speed curves:
      For each hidden init, plot the cumulative fraction of runs that have
      reached within (threshold_mult × best_loss) by each epoch.

    If aggregate_over_inputs=True (default):
      - One figure, one axis, one curve per init (aggregated across selected input types).
    Else:
      - Facet by input_type: grid of panels; each panel shows per-init curves
        using only runs from that input type.

    Notes:
      - Runs that never reach the threshold remain censored; curves can plateau < 1.0.
      - Denominator is total runs (including censored) for honest comparison.
    """
    ts_bucket = _load_ts_bucket(ts_bucket_path)
    fs = _get_fontsizes(font_scale)

    # Available labels
    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})

    use_inputs = [
        it for it in (_to_list(input_types) or all_inputs) if it in all_inputs
    ]
    use_inits = [h for h in (_to_list(hidden_inits) or all_inits) if h in all_inits]
    if len(use_inputs) == 0 or len(use_inits) == 0:
        raise ValueError("No matching input_types or hidden_inits found in ts_bucket.")

    # Helper: collect all loss sequences per (init, input)
    def _loss_lists_for_pair(h: str, it: str) -> List[List[float]]:
        entry = ts_bucket.get((h, it), {})
        losses = entry.get("losses", [])
        # keep only non-empty lists
        return [l for l in losses if isinstance(l, (list, tuple)) and len(l) > 0]

    # Prepare plotting
    import seaborn as sns

    palette = sns.color_palette(n_colors=len(use_inits))
    color_map = {h: palette[i] for i, h in enumerate(use_inits)}

    if aggregate_over_inputs:
        fig, ax = plt.subplots(figsize=figsize)

        # Combine selected input types per init
        for h in use_inits:
            run_losses = []
            for it in use_inputs:
                run_losses.extend(_loss_lists_for_pair(h, it))
            if not run_losses:
                continue

            # Compute time-to-threshold per run + the global epoch limit
            t_list = []
            max_epoch = 0
            for loss in run_losses:
                t = _first_epoch_within_threshold(loss, mult=threshold_mult)
                t_list.append(t)
                if len(loss) - 1 > max_epoch:
                    max_epoch = len(loss) - 1

            ecdf = _ecdf_over_time(t_list, max_epoch)
            if rolling and rolling > 1:
                s = pd.Series(ecdf)
                ecdf = s.rolling(rolling, min_periods=1).mean().to_numpy()

            x = np.arange(len(ecdf))
            y = ecdf * (100.0 if percent else 1.0)
            ax.plot(x, y, lw=lw, alpha=alpha_curves, color=color_map[h], label=h)

            # Optional quantile markers (e.g., median)
            if show_quantiles:
                for q in show_quantiles:
                    if not (0.0 <= q <= 1.0):
                        continue
                    # first epoch where ecdf >= q
                    idx = np.argmax(ecdf >= q)
                    if ecdf[idx] >= q:
                        ax.axvline(idx, color=color_map[h], ls="--", lw=1.2, alpha=0.7)
                        ax.text(
                            idx,
                            (q * (100.0 if percent else 1.0)),
                            f"{h}  q={int(q*100)}% @ {idx}",
                            fontsize=max(10, int(fs["ticks"] * 0.9)),
                            color=color_map[h],
                            ha="left",
                            va="bottom",
                            rotation=90,
                            alpha=0.9,
                        )

        ax.set_title(
            title
            or f"Convergence speed (≤ {int(threshold_mult*100)}% of run best loss)",
            fontsize=fs["title"],
        )
        ax.set_xlabel("Epoch", fontsize=fs["label"])
        ax.set_ylabel(
            "Cumulative " + ("%" if percent else "fraction") + " of runs",
            fontsize=fs["label"],
        )
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.set_ylim(0, 100 if percent else 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(
            title="hidden_init",
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout()
        plt.show()
        return

    # Facet by input_type: one panel per input; per-init curves inside each
    n = len(use_inputs)
    ncols_eff = max(1, ncols)
    nrows = int(np.ceil(n / ncols_eff))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_eff,
        figsize=(figsize[0], max(figsize[1], 0.9 * figsize[1] * nrows)),
        squeeze=False,
    )
    axes = axes.ravel()

    for idx, it in enumerate(use_inputs):
        ax = axes[idx]
        ax.set_title(
            f"input_type = {it}  •  threshold ≤ {int(threshold_mult*100)}%",
            fontsize=fs["title"],
        )

        for h in use_inits:
            run_losses = _loss_lists_for_pair(h, it)
            if not run_losses:
                continue

            t_list = []
            max_epoch = 0
            for loss in run_losses:
                t = _first_epoch_within_threshold(loss, mult=threshold_mult)
                t_list.append(t)
                if len(loss) - 1 > max_epoch:
                    max_epoch = len(loss) - 1

            ecdf = _ecdf_over_time(t_list, max_epoch)
            if rolling and rolling > 1:
                s = pd.Series(ecdf)
                ecdf = s.rolling(rolling, min_periods=1).mean().to_numpy()

            x = np.arange(len(ecdf))
            y = ecdf * (100.0 if percent else 1.0)
            ax.plot(x, y, lw=lw, alpha=alpha_curves, color=color_map[h], label=h)

            if show_quantiles:
                for q in show_quantiles:
                    if not (0.0 <= q <= 1.0):
                        continue
                    idx_q = np.argmax(ecdf >= q)
                    if ecdf[idx_q] >= q:
                        ax.axvline(
                            idx_q, color=color_map[h], ls="--", lw=1.2, alpha=0.7
                        )

        ax.set_xlabel("Epoch", fontsize=fs["label"])
        ax.set_ylabel(
            "Cumulative " + ("%" if percent else "fraction") + " of runs",
            fontsize=fs["label"],
        )
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.set_ylim(0, 100 if percent else 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)

    # remove any unused axes
    for j in range(len(use_inputs), len(axes)):
        fig.delaxes(axes[j])

    # single legend
    handles, labels = [], []
    for a in axes[: max(1, len(use_inputs))]:
        hnd, lbl = a.get_legend_handles_labels()
        if hnd:
            handles, labels = hnd, lbl
            break
    if handles:
        fig.legend(
            handles,
            labels,
            title="hidden_init",
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            loc="upper center",
            ncol=min(4, len(use_inits)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        title or "Convergence speed curves (survival-style)", fontsize=fs["title"]
    )
    plt.tight_layout()
    plt.show()


def plot_stability_vs_accuracy(
    per_run_df: Optional[pd.DataFrame] = None,
    per_run_csv: Optional[Path] = None,
    input_types: Union[str, List[str], None] = None,  # filter encodings
    hidden_inits: Union[str, List[str], None] = None,  # filter inits
    aggregate_over_inputs: bool = True,  # True: one point per init (aggregate selected inputs)
    min_runs: int = 2,  # require at least N runs to compute stats
    figsize: Tuple[int, int] = (10, 7),
    font_scale: float = 1.6,
    title: Optional[str] = None,
    annotate: bool = True,  # label points with init (or init/input)
    percent_y: bool = False,  # show CV as %
    show_quadrants: bool = True,  # draw medians to guide “robust & good”
) -> None:
    """
    Stability vs. accuracy:
      - x-axis: mean(best_loss)  (lower is better -> accuracy)
      - y-axis: coefficient of variation (std/mean) of best_loss across runs  (lower is better -> stability)

    When aggregate_over_inputs=True:
      - One point per hidden_init (aggregated over all selected input types)

    When aggregate_over_inputs=False:
      - Points per (hidden_init, input_type); colored by hidden_init, faceted by input_type.

    Bubble size encodes number of runs used for that point.
    """
    # Load data
    df = _ensure_df(per_run_df, per_run_csv)
    fs = _get_fontsizes(font_scale)

    # Filters
    df = _filter_by_input_type_df(df, input_types) if "input_type" in df.columns else df
    df = (
        _filter_by_hidden_init_df(df, hidden_inits)
        if "hidden_init" in df.columns
        else df
    )

    # Drop NAs and keep relevant columns
    needed = {"hidden_init", "input_type", "best_loss"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"per_run_df is missing required columns: {missing}")
    df = df.dropna(subset=["best_loss"]).copy()

    # Grouping
    if aggregate_over_inputs:
        group_cols = ["hidden_init"]
    else:
        group_cols = ["hidden_init", "input_type"]

    def _cv(x: pd.Series) -> float:
        m = float(np.nanmean(x))
        s = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
        return (s / m) if (np.isfinite(m) and m != 0.0) else np.nan

    stats = (
        df.groupby(group_cols, dropna=False)["best_loss"]
        .agg(
            best_loss_mean=lambda x: float(np.nanmean(x)),
            best_loss_std=lambda x: float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0,
            num_runs="count",
        )
        .reset_index()
    )
    stats["cv_best_loss"] = stats["best_loss_std"] / stats["best_loss_mean"]
    stats = stats[stats["num_runs"] >= max(1, min_runs)].copy()

    if stats.empty:
        raise ValueError(
            "No groups with sufficient runs to compute stability (check filters/min_runs)."
        )

    # Plotting
    if aggregate_over_inputs:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        # color by hidden_init; one point per init
        palette = sns.color_palette(n_colors=stats["hidden_init"].nunique())
        cmap = {
            h: palette[i] for i, h in enumerate(sorted(stats["hidden_init"].unique()))
        }

        x = stats["best_loss_mean"].values
        y = stats["cv_best_loss"].values * (100.0 if percent_y else 1.0)
        sizes = 40 + 10 * stats["num_runs"].values  # bubble size ~ #runs

        for _, row in stats.iterrows():
            xi = row["best_loss_mean"]
            yi = row["cv_best_loss"] * (100.0 if percent_y else 1.0)
            hi = row["hidden_init"]
            ax.scatter(
                xi,
                yi,
                s=40 + 10 * row["num_runs"],
                color=cmap[hi],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                label=hi,
            )

        # de-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(
            uniq.values(),
            uniq.keys(),
            title="hidden_init",
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        # Annotations
        if annotate:
            for _, r in stats.iterrows():
                ax.annotate(
                    r["hidden_init"],
                    (
                        r["best_loss_mean"],
                        r["cv_best_loss"] * (100.0 if percent_y else 1.0),
                    ),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=max(10, int(fs["ticks"] * 0.9)),
                )

        # Quadrant medians (guide)
        if show_quadrants and len(stats) >= 2:
            vx = np.median(stats["best_loss_mean"])
            vy = np.median(stats["cv_best_loss"]) * (100.0 if percent_y else 1.0)
            ax.axvline(vx, ls="--", lw=1.2, color="gray", alpha=0.7)
            ax.axhline(vy, ls="--", lw=1.2, color="gray", alpha=0.7)

        ax.set_title(
            title or "Stability vs. Accuracy (aggregated over inputs)",
            fontsize=fs["title"],
        )
        ax.set_xlabel("Mean Best Loss (lower is better)", fontsize=fs["label"])
        ax.set_ylabel(
            ("CV of Best Loss" + " (%)" if percent_y else "CV of Best Loss"),
            fontsize=fs["label"],
        )
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.show()
        return

    # Faceted version: points per (hidden_init, input_type), color by hidden_init, facet by input_type
    inputs = sorted(stats["input_type"].unique().tolist())
    n = len(inputs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0], max(figsize[1], 0.9 * figsize[1] * nrows)),
        squeeze=False,
    )
    axes = axes.ravel()

    palette = sns.color_palette(n_colors=stats["hidden_init"].nunique())
    cmap = {h: palette[i] for i, h in enumerate(sorted(stats["hidden_init"].unique()))}

    for idx, it in enumerate(inputs):
        ax = axes[idx]
        sub = stats[stats["input_type"] == it]
        ax.set_title(f"{it}", fontsize=fs["title"])

        for _, r in sub.iterrows():
            xi = r["best_loss_mean"]
            yi = r["cv_best_loss"] * (100.0 if percent_y else 1.0)
            hi = r["hidden_init"]
            ax.scatter(
                xi,
                yi,
                s=40 + 10 * r["num_runs"],
                color=cmap[hi],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                label=hi,
            )
            if annotate:
                ax.annotate(
                    hi,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=max(10, int(fs["ticks"] * 0.9)),
                )

        ax.set_xlabel("Mean Best Loss", fontsize=fs["label"])
        ax.set_ylabel(("CV (%)" if percent_y else "CV"), fontsize=fs["label"])
        ax.tick_params(axis="both", labelsize=fs["ticks"])
        ax.grid(True, linestyle=":", alpha=0.5)

    # remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # shared legend
    handles, labels = [], []
    for a in axes[: max(1, n)]:
        h, l = a.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        # dedupe
        uniq = dict(zip(labels, handles))
        fig.legend(
            uniq.values(),
            uniq.keys(),
            title="hidden_init",
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            loc="upper center",
            ncol=min(4, len(uniq)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(title or "Stability vs. Accuracy by input type", fontsize=fs["title"])
    plt.tight_layout()
    plt.show()


# --- t-SNE of per-run training dynamics from metrics_df_list ---
def _build_epoch_grid(
    dfs: List[pd.DataFrame], epoch_col: str, epoch_max: Optional[int], epoch_step: int
) -> np.ndarray:
    """Choose a global epoch grid to interpolate onto."""
    max_e = 0
    for df in dfs:
        if epoch_col in df.columns and not df.empty:
            me = int(np.nanmax(df[epoch_col].values))
            max_e = max(max_e, me)
    if epoch_max is not None:
        max_e = min(max_e, int(epoch_max))
    if max_e <= 0:
        return np.array([], dtype=int)
    return np.arange(0, max_e + 1, max(1, int(epoch_step)), dtype=int)


def _interp_series_to_grid(
    epoch: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linear interpolate values(x=epoch) onto grid; returns NaNs where interpolation is impossible."""
    epoch = epoch.astype(float)
    values = values.astype(float)
    if epoch.size < 2:
        # not enough points to interpolate -> constant fill if single point, else NaNs
        if epoch.size == 1:
            return np.full(grid.shape, float(values[0]))
        return np.full(grid.shape, np.nan)
    # Ensure strictly increasing x for interp
    order = np.argsort(epoch)
    x = epoch[order]
    y = values[order]
    # Interpolate within range; outside -> NaN
    y_interp = np.interp(np.clip(grid, x[0], x[-1]), x, y)
    y_interp[(grid < x[0]) | (grid > x[-1])] = np.nan
    return y_interp


def tsne_training_dynamics(
    ts_bucket_path: Path,
    metrics: Union[str, List[str]] = ("loss", "spectral_radius", "frob", "cond_num"),
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
    epoch_max: Optional[int] = None,
    epoch_step: int = 1,
    rolling: Optional[int] = None,
    normalize: str = "zscore",
    pca_components: Optional[int] = 20,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (9, 7),
    font_scale: float = 1.6,
    annotate: bool = False,
    return_data: bool = False,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Robust t-SNE over per-run metric trajectories from metrics_df_list.
    Enforces numeric dtypes and sanitizes NaNs/strings to avoid sklearn dtype errors.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns

    ts_bucket = _load_ts_bucket(ts_bucket_path)
    fs = _get_fontsizes(font_scale)

    # --- helpers ---
    def _as_list_opt(x):
        if x is None:
            return None
        if isinstance(x, str):
            return [x]
        return list(x)

    def _build_epoch_grid(
        dfs: List[pd.DataFrame],
        epoch_col: str,
        epoch_max: Optional[int],
        epoch_step: int,
    ) -> np.ndarray:
        max_e = 0
        for df in dfs:
            if epoch_col in df.columns and not df.empty:
                try:
                    me = pd.to_numeric(df[epoch_col], errors="coerce").dropna()
                    if not me.empty:
                        max_e = max(max_e, int(me.max()))
                except Exception:
                    pass
        if epoch_max is not None:
            max_e = min(max_e, int(epoch_max))
        if max_e <= 0:
            return np.array([], dtype=int)
        return np.arange(0, max_e + 1, max(1, int(epoch_step)), dtype=int)

    def _interp_series_to_grid(
        epoch: np.ndarray, values: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        epoch = epoch.astype(float)
        values = values.astype(float)
        # drop non-finite
        mask = np.isfinite(epoch) & np.isfinite(values)
        epoch = epoch[mask]
        values = values[mask]
        if epoch.size == 0:
            return np.full_like(grid, np.nan, dtype=float)
        if epoch.size == 1:
            return np.full_like(grid, float(values[0]), dtype=float)
        order = np.argsort(epoch)
        x = epoch[order]
        y = values[order]
        y_interp = np.interp(np.clip(grid, x[0], x[-1]), x, y)
        y_interp[(grid < x[0]) | (grid > x[-1])] = np.nan
        return y_interp

    # --- select settings ---
    metrics = _as_list_opt(metrics)
    if not metrics:
        raise ValueError("Provide at least one metric to use from metrics_df_list.")
    all_inits = sorted({h for (h, it) in ts_bucket.keys()})
    all_inputs = sorted({it for (h, it) in ts_bucket.keys()})
    use_inputs = [
        it for it in (_as_list_opt(input_types) or all_inputs) if it in all_inputs
    ]
    use_inits = [h for h in (_as_list_opt(hidden_inits) or all_inits) if h in all_inits]

    # --- gather per-run frames ---
    run_frames: List[pd.DataFrame] = []
    run_meta: List[Tuple[str, str, str]] = []  # (hidden_init, input_type, run_id)
    bad_cols_examples: List[
        Tuple[str, str, str, str]
    ] = []  # (init, input, metric, example_value)

    for h in use_inits:
        for it in use_inputs:
            entry = ts_bucket.get((h, it), {})
            mlist = entry.get("metrics_df_list", [])
            for df in mlist:
                if (
                    not isinstance(df, pd.DataFrame)
                    or df.empty
                    or "epoch" not in df.columns
                ):
                    continue
                # force numeric epoch
                df2 = df.copy()
                df2["epoch"] = pd.to_numeric(df2["epoch"], errors="coerce")
                df2 = df2.dropna(subset=["epoch"])
                if df2.empty:
                    continue
                # coerce requested metric cols to numeric if present
                for m in metrics:
                    if m in df2.columns:
                        before_non_numeric = df2[m].dtype
                        df2[m] = pd.to_numeric(df2[m], errors="coerce")
                        # collect an example of non-numeric if we coerced many to NaN
                        if str(before_non_numeric) == "object":
                            ex = (
                                df[m].iloc[0]
                                if m in df.columns and len(df)
                                else "object"
                            )
                            bad_cols_examples.append((h, it, m, str(ex)))
                # identify run_id
                rid = str(df2.get("run_id", [f"{h}_{it}_{len(run_frames)}"]).iloc[0])
                run_frames.append(df2)
                run_meta.append((h, it, rid))

    if not run_frames:
        raise ValueError(
            "No metrics time-series found in ts_bucket for the selected filters."
        )

    # available metrics across frames (after coercion)
    available = set()
    for df in run_frames:
        for m in metrics:
            if m in df.columns:
                if pd.to_numeric(df[m], errors="coerce").notna().any():
                    available.add(m)
    used_metrics = [m for m in metrics if m in available]
    if not used_metrics:
        raise ValueError(
            f"None of the requested metrics {metrics} have numeric values in metrics_df_list."
        )

    # epoch grid
    grid = _build_epoch_grid(
        run_frames, "epoch", epoch_max=epoch_max, epoch_step=epoch_step
    )
    if grid.size == 0:
        raise ValueError(
            "Could not determine a valid epoch grid from the available metrics (check epochs)."
        )

    # --- feature matrix ---
    feat_rows: List[np.ndarray] = []
    valid_indices: List[int] = []

    for i, df in enumerate(run_frames):
        df2 = df.sort_values("epoch").copy()
        series_list = []
        for m in used_metrics:
            if m not in df2.columns:
                series_list.append(np.full(grid.shape, np.nan, dtype=float))
                continue
            y = pd.to_numeric(df2[m], errors="coerce").to_numpy(dtype=float)
            x = pd.to_numeric(df2["epoch"], errors="coerce").to_numpy(dtype=float)
            vec = _interp_series_to_grid(x, y, grid)
            if rolling and rolling > 1:
                vec = pd.Series(vec).rolling(rolling, min_periods=1).mean().to_numpy()
            series_list.append(vec.astype(float))
        if series_list:
            feat = np.concatenate(series_list, axis=0).astype(np.float64)
            feat_rows.append(feat)
            valid_indices.append(i)

    if not feat_rows:
        raise ValueError("Failed to construct any numeric feature vectors (all NaNs?).")

    X = np.vstack(feat_rows).astype(np.float64)  # ensure float
    meta = [run_meta[i] for i in valid_indices]

    # normalize
    if normalize == "zscore":
        col_mean = np.nanmean(X, axis=0)
        col_std = np.nanstd(X, axis=0)
        safe_std = np.where(col_std > 0, col_std, 1.0)
        X = (X - col_mean) / safe_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # Optional PCA (safe bounds)
    if pca_components and pca_components > 0:
        k = min(int(pca_components), X.shape[1], max(2, X.shape[0] - 1))
        X_pca = (
            PCA(n_components=k, random_state=random_state).fit_transform(X)
            if k >= 2
            else X
        )
    else:
        X_pca = X

    # Perplexity must be < #samples
    n_samples = X_pca.shape[0]
    eff_perp = float(min(perplexity, max(5, n_samples - 1)))
    if eff_perp >= n_samples:
        eff_perp = max(5.0, n_samples / 3.0)

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perp,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate=200,
        metric="euclidean",
        verbose=0,
    )
    # --- this is where your error occurred previously ---
    Z = tsne.fit_transform(X_pca.astype(np.float64))

    # embedding df
    emb = pd.DataFrame(Z, columns=["tsne_1", "tsne_2"])
    emb["hidden_init"] = [h for (h, _, _) in meta]
    emb["input_type"] = [it for (_, it, _) in meta]
    emb["run_id"] = [rid for (_, _, rid) in meta]

    # plot
    import seaborn as sns

    plt.figure(figsize=figsize)
    ax = plt.gca()
    sns.scatterplot(
        data=emb,
        x="tsne_1",
        y="tsne_2",
        hue="hidden_init",
        style="input_type",
        s=80,
        alpha=0.9,
        ax=ax,
    )
    ax.set_title(
        "t-SNE of training dynamics (metrics time-series)", fontsize=fs["title"]
    )
    ax.set_xlabel("t-SNE 1", fontsize=fs["label"])
    ax.set_ylabel("t-SNE 2", fontsize=fs["label"])
    ax.tick_params(axis="both", labelsize=fs["ticks"])
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        title="hidden_init",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=fs["legend"],
        title_fontsize=fs["legend"],
    )

    if annotate:
        for _, r in emb.iterrows():
            ax.annotate(
                str(r["run_id"]),
                (r["tsne_1"], r["tsne_2"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=max(9, int(fs["ticks"] * 0.8)),
                alpha=0.9,
            )
    plt.tight_layout()
    plt.show()

    # Helpful diagnostics if we saw non-numeric columns
    if bad_cols_examples:
        uniq = {(h, it, m) for (h, it, m, _) in bad_cols_examples}
        print(
            "[INFO] Coerced non-numeric metric values to NaN in the following (showing one example each):"
        )
        for h, it, m in list(uniq)[:6]:
            ex = next(
                (
                    ex
                    for (_h, _it, _m, ex) in bad_cols_examples
                    if (_h, _it, _m) == (h, it, m)
                ),
                "object",
            )
            print(f"  - ({h}, {it}) metric='{m}' example value: {ex!r}")

    if return_data:
        feat_cols = []
        for m in used_metrics:
            for t in range(len(grid)):
                feat_cols.append(f"{m}@t{int(grid[t])}")
        feature_df = pd.DataFrame(X, columns=feat_cols)
        feature_df["hidden_init"] = emb["hidden_init"].values
        feature_df["input_type"] = emb["input_type"].values
        feature_df["run_id"] = emb["run_id"].values
        return emb, feature_df
    return None


def plot_parallel_coordinates(
    per_run_df: Optional[pd.DataFrame] = None,
    per_run_csv: Optional[Path] = None,
    metrics: Optional[
        List[str]
    ] = None,  # e.g., ["final_loss","final_spectral_radius","final_orth_err","final_frob"]
    color_by: str = "hidden_init",  # {"hidden_init","input_type"}
    input_types: Union[str, List[str], None] = None,
    hidden_inits: Union[str, List[str], None] = None,
    normalize: str = "minmax",  # {"minmax","zscore","none"}
    drop_na: bool = True,  # drop rows with any NA in selected metrics
    max_runs_per_group: Optional[
        int
    ] = 200,  # subsample within each color group to reduce clutter
    show_group_means: bool = True,  # thicker mean line per group over normalized axes
    alpha: float = 0.25,  # line transparency for runs
    lw: float = 1.2,  # line width for runs
    lw_mean: float = 3.0,  # line width for group mean
    figsize: Tuple[int, int] = (12, 7),
    font_scale: float = 1.6,
    title: Optional[str] = "Parallel Coordinates: per-run scalar metrics",
) -> None:
    """
    Parallel coordinates plot of per-run scalar metrics.
    Each polyline = one run. Colors = groups (hidden_init or input_type).
    """

    df = _ensure_df(per_run_df, per_run_csv)
    fs = _get_fontsizes(font_scale)

    # Basic columns checks
    needed_group_cols = {"hidden_init", "input_type"}
    missing = [c for c in needed_group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"per_run_df is missing required columns: {missing}")

    # Filters
    if input_types is not None:
        df = _filter_by_input_type_df(df, input_types)
    if hidden_inits is not None:
        df = _filter_by_hidden_init_df(df, hidden_inits)

    # Default metrics: prefer final_* if present; otherwise some common scalars
    if metrics is None:
        preferred = [
            "final_loss",
            "best_loss",
            "final_spectral_radius",
            "final_spectral_norm",
            "final_orth_err",
            "final_frob",
            "final_cond_num",
            "final_w_max_abs",
            "final_w_sparsity",
            "final_act_mean",
            "final_tanh_sat",
        ]
        metrics = [m for m in preferred if m in df.columns]
        if len(metrics) < 2:
            # fallback to whatever scalar columns exist (excluding ids)
            exclude = {"hidden_init", "input_type", "run_kind", "run_id", "path"}
            candidates = [
                c
                for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            metrics = candidates[:6]  # cap to keep it readable
    if len(metrics) < 2:
        raise ValueError("Need at least 2 metric columns to draw parallel coordinates.")

    # Keep just the columns we need
    use_cols = ["hidden_init", "input_type", "run_id"] + metrics
    use_cols = [c for c in use_cols if c in df.columns]
    df = df[use_cols].copy()

    if drop_na:
        df = df.dropna(subset=metrics)
    if df.empty:
        raise ValueError(
            "No rows left after filtering/dropping NA for the selected metrics."
        )

    # Optional subsample to reduce clutter, stratified by color group
    if max_runs_per_group is not None and max_runs_per_group > 0:
        group_col = color_by
        df = (
            df.groupby(group_col, group_keys=False)
            .apply(
                lambda g: g.sample(n=min(len(g), max_runs_per_group), random_state=17)
            )
            .reset_index(drop=True)
        )

    # Normalize metrics (column-wise on the filtered data)
    df_norm = df.copy()
    if normalize == "minmax":
        for m in metrics:
            col = df_norm[m].astype(float)
            vmin, vmax = np.nanmin(col.values), np.nanmax(col.values)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                df_norm[m] = (col - vmin) / (vmax - vmin)
            else:
                df_norm[m] = 0.5  # degenerate constant column
    elif normalize == "zscore":
        for m in metrics:
            col = df_norm[m].astype(float)
            mu, sd = np.nanmean(col.values), np.nanstd(col.values, ddof=1)
            if np.isfinite(sd) and sd > 0:
                z = (col - mu) / sd
                # map roughly to 0..1 using +/- 2σ clamp
                df_norm[m] = np.clip((z + 2.0) / 4.0, 0.0, 1.0)
            else:
                df_norm[m] = 0.5
    elif normalize == "none":
        # still plot; axes will be on raw scales (harder to read across metrics)
        pass
    else:
        raise ValueError("normalize must be one of {'minmax','zscore','none'}")

    # Build figure & palette
    plt.figure(figsize=figsize)
    ax = plt.gca()
    groups = df_norm[color_by].unique().tolist()
    palette = sns.color_palette(n_colors=len(groups))
    color_map = {g: palette[i] for i, g in enumerate(groups)}

    # X positions for axes
    x_coords = np.arange(len(metrics))

    # Draw each run
    for _, row in df_norm.iterrows():
        vals = row[metrics].astype(float).values
        color = color_map[row[color_by]]
        ax.plot(x_coords, vals, color=color, alpha=alpha, lw=lw)

    # Optional group means (thicker, labeled)
    if show_group_means:
        for g in groups:
            sub = df_norm[df_norm[color_by] == g]
            if sub.empty:
                continue
            mean_vals = sub[metrics].astype(float).mean().values
            ax.plot(
                x_coords,
                mean_vals,
                color=color_map[g],
                lw=lw_mean,
                alpha=0.95,
                label=str(g),
            )
            ax.scatter(
                x_coords,
                mean_vals,
                color=color_map[g],
                s=30,
                zorder=3,
                edgecolor="black",
                linewidth=0.5,
            )

    # Axis labels/formatting
    ax.set_xticks(x_coords)
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=fs["label"])
    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(0, 1 if normalize in ("minmax", "zscore") else None)
    ax.set_ylabel(
        "Normalized value" if normalize in ("minmax", "zscore") else "Raw value",
        fontsize=fs["label"],
    )
    ax.set_title(
        title or "Parallel Coordinates: per-run scalar metrics", fontsize=fs["title"]
    )
    ax.tick_params(axis="y", labelsize=fs["ticks"])
    ax.grid(True, linestyle=":", alpha=0.5)

    # Legend (group means only, to avoid clutter)
    if show_group_means:
        ax.legend(
            title=color_by,
            fontsize=fs["legend"],
            title_fontsize=fs["legend"],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    plt.tight_layout()
    plt.show()


#########################################################################################################################################################################################       Weight Traces       ########################### ################################################################################
################################################################################

import torch
import torch.nn as nn
from RNN_Class import ElmanRNN_pytorch_module_v2 as Elman
import math

# -------------------
# Project layout
# -------------------
try:
    MODULE_DIR = Path(__file__).resolve().parent
except NameError:
    # e.g., running inside a notebook cell
    MODULE_DIR = Path.cwd()
DATA_DIR = MODULE_DIR / "data" / "Ns100_SeqN100"
MODEL_ROOT = MODULE_DIR / "Elman_SGD" / "Remap_predloss" / "N100T100"
MULTIRUNS_DIR = "multiruns"
RUN_PREFIX = "run_"
MODEL_FNAME = "Ns100_SeqN100_predloss_full.pth.tar"
HIDDEN_WEIGHTS_SUBDIR = "hidden-weights"


def _style_axes(ax, font_scale: float = 1.0):
    ax.title.set_fontsize(14 * font_scale)
    ax.xaxis.label.set_fontsize(12 * font_scale)
    ax.yaxis.label.set_fontsize(12 * font_scale)
    ax.tick_params(axis="both", which="major", labelsize=11 * font_scale)
    ax.tick_params(axis="both", which="minor", labelsize=10 * font_scale)
    # legend (if present)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(11 * font_scale)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(12 * font_scale)


def _style_cbar(cbar, font_scale: float = 1.0):
    if cbar is None:
        return
    cbar.ax.tick_params(labelsize=11 * font_scale)
    label = cbar.ax.get_ylabel()
    if label:
        cbar.ax.set_ylabel(label, fontsize=12 * font_scale)


# -------------------
# I/O
# -------------------
def _load_dataset_Xmini(N=100, SeqN=100, data_dir: Path = DATA_DIR) -> torch.Tensor:
    pkg = torch.load(data_dir / f"Ns{N}_SeqN{SeqN}_1.pth.tar", map_location="cpu")
    return pkg["X_mini"][:, :-1, :]  # (1, SeqN-1, N)


def _load_ckpt(path: Union[str, Path], map_location="cpu") -> Dict:
    return torch.load(Path(path), map_location=map_location)


def _find_hidden_W(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    if "hidden_linear.weight" in state_dict:
        W = state_dict["hidden_linear.weight"]
    elif "rnn.weight_hh_l0" in state_dict:
        W = state_dict["rnn.weight_hh_l0"]
    else:
        raise KeyError("Hidden weight matrix not found in state_dict.")
    return W.detach().cpu().numpy()


# -------------------
# Peak-time selection (argmax OR rolling-mean argmax)
# -------------------
def peak_time_from_argmax(hidden_seq: np.ndarray) -> np.ndarray:
    """hidden_seq: (HN, T)"""
    return np.argmax(hidden_seq, axis=1).astype(int)


def peak_time_from_rolling_mean(hidden_seq: np.ndarray, window: int = 5) -> np.ndarray:
    """hidden_seq: (HN, T). Centered moving average with 'same' conv."""
    HN, T = hidden_seq.shape
    kernel = np.ones(window, dtype=np.float32) / float(window)
    peak_t = np.empty(HN, dtype=int)
    for i in range(HN):
        sm = np.convolve(hidden_seq[i], kernel, mode="same")
        peak_t[i] = int(np.argmax(sm))
    return peak_t


def compute_peak_times(
    hidden_seq: np.ndarray,
    use_rolling: bool = False,
    rolling_window: int = 5,
) -> np.ndarray:
    """Return per-neuron peak time indices according to the chosen method."""
    if use_rolling:
        # recommend odd window (5/7/9) for centered smoothing
        return peak_time_from_rolling_mean(hidden_seq, window=rolling_window)
    return peak_time_from_argmax(hidden_seq)


# -------------------
# Hidden sequence (forward pass over dataset)
# -------------------
def hidden_sequence_from_dataset(
    state_dict: Dict[str, torch.Tensor],
    N: int,
    SeqN: int,
    HN: int,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    data_dir: Path = DATA_DIR,
) -> np.ndarray:
    X_in = _load_dataset_Xmini(N=N, SeqN=SeqN, data_dir=data_dir).to(device)  # (1,T,N)
    model = Elman(N, HN, N).to(device)
    if activation is not None:
        model.act = activation
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    T = X_in.shape[1]
    h = torch.zeros(1, 1, HN, device=device)
    hidden_seq = torch.empty(HN, T, device=device)

    with torch.no_grad():
        for t in range(T):
            _, h = model(X_in[:, t : t + 1, :], h)
            hidden_seq[:, t] = h.squeeze(0).squeeze(0)
    return hidden_seq.detach().cpu().numpy()


# -------------------
# Single-run weight trace
# -------------------
def weight_trace_single_run(
    model_path: Union[str, Path],
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    wh_type: Optional[str] = None,
    i_type: Optional[str] = None,
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: (offsets, trace_vals, sort_idx)
    """
    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]

    W = _find_hidden_W(state_dict)  # (HN, HN)
    hidden_seq = hidden_sequence_from_dataset(
        state_dict, N=N, SeqN=SeqN, HN=HN, activation=activation, device=device
    )

    # peak times: argmax OR rolling-mean argmax
    peak_t = compute_peak_times(
        hidden_seq, use_rolling=use_rolling, rolling_window=rolling_window
    )
    sort_idx = np.argsort(peak_t)

    W_sorted = W[sort_idx, :][:, sort_idx]
    offsets = np.arange(-(HN - 1), HN)
    trace_vals = np.array([np.trace(W_sorted, offset=k) for k in offsets])

    if not plot:
        return offsets, trace_vals, sort_idx

    # plotting
    c_max = float(np.abs(W_sorted).max()) if W_sorted.size else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.imshow(
        W_sorted, cmap="RdBu_r", vmin=-1.001 * c_max, vmax=+1.001 * c_max, aspect="auto"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-c_max, 0.0, c_max])
    cbar.ax.set_yticklabels([f"{-c_max:.3f}", "0.000", f"{c_max:.3f}"])
    ax.set_xlabel("Presynaptic (sorted)")
    ax.set_ylabel("Postsynaptic (sorted)")
    title_tag = "rolling" if use_rolling else "argmax"
    ax.set_title(
        f"W sorted ({title_tag}) — {wh_type}, {i_type}"
        if wh_type and i_type
        else f"W sorted ({title_tag})"
    )

    ax = axes[1]
    ax.plot(offsets, trace_vals, label="sum diag(W_sorted, k)")
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Diagonal offset (k)")
    ax.set_ylabel("Weight sum")
    ax.set_title(f"Diagonal-offset trace ({title_tag})")
    ax.legend()

    plt.tight_layout()
    plt.show()
    return offsets, trace_vals, sort_idx


# -------------------
# Multi-run helpers
# -------------------
def _run_dir(hidden_init: str, input_type: str) -> Path:
    return MODEL_ROOT / hidden_init / input_type / MULTIRUNS_DIR


def _find_run_paths(hidden_init: str, input_type: str) -> List[Tuple[str, Path]]:
    base = _run_dir(hidden_init, input_type)
    if not base.exists():
        return []
    out = []
    for d in sorted(base.glob(f"{RUN_PREFIX}*")):
        p = d / MODEL_FNAME
        if p.exists():
            rid = d.name.replace(RUN_PREFIX, "", 1)
            out.append((rid, p))
    return out


def compute_trace_for_run(
    model_path: Union[str, Path],
    N=100,
    SeqN=100,
    HN=100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    plot=False,
) -> Tuple[np.ndarray, np.ndarray]:
    offsets, trace_vals, _ = weight_trace_single_run(
        model_path,
        N=N,
        SeqN=SeqN,
        HN=HN,
        activation=activation,
        device=device,
        use_rolling=use_rolling,
        rolling_window=rolling_window,
        plot=plot,
    )
    return offsets, trace_vals


def compute_traces_for_setting(
    hidden_init: str,
    input_type: str,
    mode: str = "mean",  # {"single", "mean"}
    run_id: Optional[str] = None,
    N=100,
    SeqN=100,
    HN=100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    show_individual: bool = False,
):
    paths = _find_run_paths(hidden_init, input_type)
    if not paths:
        return None

    if mode == "single":
        rid, p = (
            next(((r, path) for r, path in paths if r == run_id), paths[0])
            if run_id is not None
            else paths[0]
        )
        offs, tr = compute_trace_for_run(
            p,
            N=N,
            SeqN=SeqN,
            HN=HN,
            activation=activation,
            device=device,
            use_rolling=use_rolling,
            rolling_window=rolling_window,
        )
        return {
            "offsets": offs,
            "trace_mean": tr,
            "trace_std": None,
            "per_run": [(rid, tr)] if show_individual else None,
        }

    per_run = []
    offs_ref = None
    for rid, p in paths:
        offs, tr = compute_trace_for_run(
            p,
            N=N,
            SeqN=SeqN,
            HN=HN,
            activation=activation,
            device=device,
            use_rolling=use_rolling,
            rolling_window=rolling_window,
        )
        if offs_ref is None:
            offs_ref = offs
        per_run.append((rid, tr))

    stack = np.vstack([tr for _, tr in per_run])
    return {
        "offsets": offs_ref,
        "trace_mean": stack.mean(axis=0),
        "trace_std": stack.std(axis=0, ddof=1)
        if stack.shape[0] > 1
        else np.zeros_like(stack[0]),
        "per_run": per_run if show_individual else None,
    }


# -------------------
# Faceted plotting
# -------------------
def plot_weight_traces(
    hidden_inits: Union[str, List[str]],
    input_types: Union[str, List[str]],
    mode: str = "mean",
    run_id: Optional[str] = None,
    facet_by: str = "input_type",
    ncols: Optional[int] = 2,
    nrows: Optional[int] = None,
    show_individual: bool = False,
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[object] = None,  # or nn.Module
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    figsize_each: Tuple[int, int] = (8, 4),
    alpha_runs: float = 0.18,
    alpha_band: float = 0.25,
    lw_mean: float = 2.0,
    font_scale: float = 1.0,
    # Legend controls (3.8-safe)
    legend: bool = True,
    legend_outside: bool = False,
    legend_loc: str = "best",
) -> None:
    import math
    from matplotlib.lines import Line2D

    if isinstance(hidden_inits, str):
        hidden_inits = [hidden_inits]
    if isinstance(input_types, str):
        input_types = [input_types]
    assert facet_by in {"input_type", "hidden_init"}

    overlay_dim = "hidden_init" if facet_by == "input_type" else "input_type"
    facets = input_types if facet_by == "input_type" else hidden_inits
    n = len(facets)

    # grid
    if ncols is None and nrows is None:
        ncols = 2
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
        squeeze=False,
    )

    method_tag = "rolling" if use_rolling else "argmax"

    # if using a single legend for the whole figure, collect items here
    global_legend: dict[str, Line2D] = {}

    for idx, facet_val in enumerate(facets):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        overlays = hidden_inits if overlay_dim == "hidden_init" else input_types

        # local legend items (if per-subplot legend)
        local_legend: dict[str, Line2D] = {}

        for ov in overlays:
            h, i = (ov, facet_val) if overlay_dim == "hidden_init" else (facet_val, ov)
            res = compute_traces_for_setting(
                h,
                i,
                mode=mode,
                run_id=run_id,
                show_individual=show_individual,
                N=N,
                SeqN=SeqN,
                HN=HN,
                activation=activation,
                device=device,
                use_rolling=use_rolling,
                rolling_window=rolling_window,
            )
            if res is None:
                continue

            offs, meanv, stdv = res["offsets"], res["trace_mean"], res["trace_std"]
            # Keep labels simple: overlay name only
            label = f"{h}" if facet_by == "input_type" else f"{i}"
            # draw mean line
            (line,) = ax.plot(offs, meanv, lw=lw_mean, label=label)
            # keep a single handle per label
            local_legend.setdefault(label, line)
            global_legend.setdefault(label, line)

            if mode == "mean" and stdv is not None:
                ax.fill_between(
                    offs, meanv - stdv, meanv + stdv, alpha=alpha_band, linewidth=0
                )
                if show_individual and res["per_run"]:
                    for _, tr in res["per_run"]:
                        ax.plot(offs, tr, alpha=alpha_runs, linewidth=1)

        ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Diagonal offset (k)")
        ax.set_ylabel("Weight sum")
        ax.set_title(
            (f"input_type = {facet_val}")
            if facet_by == "input_type"
            else (f"hidden_init = {facet_val}")
        )
        _style_axes(ax, font_scale=font_scale)

        if legend and not legend_outside:
            if local_legend:
                ax.legend(
                    list(local_legend.values()),
                    list(local_legend.keys()),
                    loc=legend_loc,
                    fontsize=11 * font_scale,
                )

    # hide empties
    total = nrows * ncols
    for j in range(n, total):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # single global legend
    if legend and legend_outside and global_legend:
        fig.legend(
            list(global_legend.values()),
            list(global_legend.keys()),
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=11 * font_scale,
            frameon=False,
        )
        fig.subplots_adjust(right=0.85)

    fig.tight_layout()
    plt.show()


def compute_sorted_W_for_run(
    model_path: Union[str, Path],
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    data_dir: Optional[Union[str, Path]] = DATA_DIR,
) -> np.ndarray:
    """
    Returns W_sorted (HN, HN) for a single run.
    """
    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]

    W = _find_hidden_W(state_dict)  # (HN, HN)
    hidden_seq = hidden_sequence_from_dataset(
        state_dict,
        N=N,
        SeqN=SeqN,
        HN=HN,
        activation=activation,
        device=device,
        data_dir=data_dir,
    )
    peak_t = compute_peak_times(
        hidden_seq, use_rolling=use_rolling, rolling_window=rolling_window
    )
    sort_idx = np.argsort(peak_t)
    W_sorted = W[sort_idx, :][:, sort_idx]
    return W_sorted


def collect_sorted_Ws_for_setting(
    hidden_init: str,
    input_type: str,
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    data_dir: Optional[Union[str, Path]] = DATA_DIR,
) -> List[np.ndarray]:
    """
    Returns a list of W_sorted (each HN x HN), one per run found.
    """
    W_list = []
    for rid, p in _find_run_paths(hidden_init, input_type):
        try:
            W_sorted = compute_sorted_W_for_run(
                p,
                N=N,
                SeqN=SeqN,
                HN=HN,
                activation=activation,
                device=device,
                use_rolling=use_rolling,
                rolling_window=rolling_window,
                data_dir=data_dir,
            )
            W_list.append(W_sorted)
        except Exception as e:
            print(f"[WARN] Skipping {hidden_init}/{input_type} run {rid}: {e}")
    return W_list


def aggregate_sorted_Ws(W_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elementwise mean and std across runs (stack W_sorted along axis=0).
    """
    if not W_list:
        raise ValueError("No W_sorted matrices provided.")
    stack = np.stack(W_list, axis=0)  # [n_runs, HN, HN]
    mean_W = stack.mean(axis=0)
    std_W = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
    return mean_W, std_W


def _compute_facet_grid(
    n: int, ncols: Optional[int], nrows: Optional[int]
) -> Tuple[int, int]:
    import math

    if ncols is None and nrows is None:
        ncols = 2
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_sorted_W_aggregates(
    hidden_inits: Union[str, List[str]],
    input_types: Union[str, List[str]],
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    show: str = "both",  # {"mean","std","both"}
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    use_rolling: bool = False,
    rolling_window: int = 5,
    data_dir: Optional[Union[str, Path]] = None,
    ncols: Optional[int] = 2,
    nrows: Optional[int] = None,
    figsize_each: Tuple[int, int] = (4, 4),
    cmap_mean: str = "RdBu_r",
    cmap_std: str = "viridis",
    font_scale: float = 1.0,
):
    if isinstance(hidden_inits, str):
        hidden_inits = [hidden_inits]
    if isinstance(input_types, str):
        input_types = [input_types]
    assert facet_by in {"input_type", "hidden_init"}
    assert show in {"mean", "std", "both"}

    facets = input_types if facet_by == "input_type" else hidden_inits

    # gather facet aggregates
    agg = {}  # facet -> (mean_W, std_W)
    for facet_val in facets:
        W_all = []
        if facet_by == "input_type":
            for hinit in hidden_inits:
                W_list = collect_sorted_Ws_for_setting(
                    hinit,
                    facet_val,
                    N=N,
                    SeqN=SeqN,
                    HN=HN,
                    activation=activation,
                    device=device,
                    use_rolling=use_rolling,
                    rolling_window=rolling_window,
                    data_dir=DATA_DIR,
                )
                if W_list:
                    mean_W, _ = aggregate_sorted_Ws(W_list)
                    W_all.append(mean_W)
        else:
            for it in input_types:
                W_list = collect_sorted_Ws_for_setting(
                    facet_val,
                    it,
                    N=N,
                    SeqN=SeqN,
                    HN=HN,
                    activation=activation,
                    device=device,
                    use_rolling=use_rolling,
                    rolling_window=rolling_window,
                    data_dir=DATA_DIR,
                )
                if W_list:
                    mean_W, _ = aggregate_sorted_Ws(W_list)
                    W_all.append(mean_W)

        if not W_all:
            print(f"[WARN] no runs found for facet={facet_val}; skipping.")
            continue

        stack = np.stack(W_all, axis=0)
        mean_of_means = stack.mean(axis=0)
        std_of_means = (
            stack.std(axis=0, ddof=1)
            if stack.shape[0] > 1
            else np.zeros_like(mean_of_means)
        )
        agg[facet_val] = (mean_of_means, std_of_means)

    if not agg:
        print("[INFO] Nothing to plot.")
        return

    means = [m for (m, s) in agg.values()]
    stds = [s for (m, s) in agg.values()]
    max_abs_mean = max(float(np.abs(m).max()) for m in means)
    max_std = max(float(s.max()) for s in stds)

    # grid dims
    import math

    n = len(agg)
    if ncols is None and nrows is None:
        ncols = 2
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)

    # MEAN figure
    if show in {"mean", "both"}:
        fig_m, axes_m = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        vmin_m, vmax_m = -max_abs_mean, +max_abs_mean
        for idx, facet_val in enumerate(facets):
            if facet_val not in agg:
                continue
            r, c = divmod(idx, ncols)
            ax = axes_m[r][c]
            mean_W, _ = agg[facet_val]
            im = ax.imshow(
                mean_W, cmap=cmap_mean, vmin=vmin_m, vmax=vmax_m, aspect="auto"
            )
            ax.set_title(f"{facet_by} = {facet_val} — mean")
            ax.set_xlabel("Presynaptic (sorted)")
            ax.set_ylabel("Postsynaptic (sorted)")
            _style_axes(ax, font_scale=font_scale)  # <--- apply scaling
            if c == ncols - 1:
                cbar = fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)  # <--- apply scaling
        # hide empties
        total = nrows * ncols
        for j in range(n, total):
            r, c = divmod(j, ncols)
            axes_m[r][c].axis("off")
        fig_m.tight_layout()
        plt.show()

    # STD figure
    if show in {"std", "both"}:
        fig_s, axes_s = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        vmin_s, vmax_s = 0.0, max_std if max_std > 0 else 1e-12
        for idx, facet_val in enumerate(facets):
            if facet_val not in agg:
                continue
            r, c = divmod(idx, ncols)
            ax = axes_s[r][c]
            _, std_W = agg[facet_val]
            im = ax.imshow(
                std_W, cmap=cmap_std, vmin=vmin_s, vmax=vmax_s, aspect="auto"
            )
            ax.set_title(f"{facet_by} = {facet_val} — std")
            ax.set_xlabel("Presynaptic (sorted)")
            ax.set_ylabel("Postsynaptic (sorted)")
            _style_axes(ax, font_scale=font_scale)  # <--- apply scaling
            if c == ncols - 1:
                cbar = fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)  # <--- apply scaling
        total = nrows * ncols
        for j in range(n, total):
            r, c = divmod(j, ncols)
            axes_s[r][c].axis("off")
        fig_s.tight_layout()
        plt.show()


######### Eigenspectrum
def _list_hidden_weight_files(run_dir_or_file: Path) -> List[Path]:
    """
    Accept either a run directory or a file within it; return sorted hidden-weight files.
    """
    run_root = run_dir_or_file
    if run_root.is_file():
        run_root = run_root.parent  # e.g., was ".../run_03/Ns...pth.tar"
    hw_dir = run_root / HIDDEN_WEIGHTS_SUBDIR
    if not hw_dir.exists():
        return []
    # be liberal in file patterns: handle .pt, .pth, .pth.tar, etc.
    files = [p for p in hw_dir.iterdir() if p.is_file()]
    files.sort()  # assumes names encode epoch order
    return files


def _load_hidden_W_from_ckpt(ckpt_path: Path) -> torch.Tensor:
    pkg = torch.load(ckpt_path, map_location="cpu")
    sd = pkg.get("state_dict", pkg)
    # common keys for your codebase
    for k in ("rnn.weight_hh_l0", "hidden_linear.weight"):
        if k in sd and torch.is_tensor(sd[k]):
            W = sd[k]
            break
    else:
        # try suffix match to be safer across variants
        k = next(
            (
                k
                for k in sd.keys()
                if k.endswith("rnn.weight_hh_l0") or k.endswith("hidden_linear.weight")
            ),
            None,
        )
        if k is None:
            raise KeyError(
                "No hidden weight key found in checkpoint: %s" % list(sd.keys())[:10]
            )
        W = sd[k]
    if W.dim() != 2:
        W = W.view(W.shape[0], -1)
    return W


def _pick_snapshot(paths: List[Path], snapshot: str) -> Optional[Path]:
    if not paths:
        return None
    snapshot = snapshot.lower()
    if snapshot == "first":
        return paths[0]
    if snapshot == "middle":
        return paths[len(paths) // 2]
    # default
    return paths[-1]


def _load_weight_tensor(p: Path) -> torch.Tensor:
    obj = torch.load(p, map_location="cpu")
    if torch.is_tensor(obj):
        W = obj
    elif isinstance(obj, dict):
        # try common keys
        for k in ("weight", "hidden_linear.weight", "rnn.weight_hh_l0"):
            if k in obj and torch.is_tensor(obj[k]):
                W = obj[k]
                break
        else:
            # if it's a plain state_dict
            for k in obj.keys():
                if k.endswith("hidden_linear.weight") or k.endswith("rnn.weight_hh_l0"):
                    W = obj[k]
                    break
            else:
                raise ValueError(
                    f"Unrecognized dict payload in {p.name}: keys={list(obj.keys())[:5]}"
                )
    else:
        raise ValueError(f"Unsupported weight file type at {p}")
    # ensure 2D
    if W.dim() != 2:
        W = W.view(W.shape[0], -1)
    return W


def eigvals_compat(W: torch.Tensor) -> np.ndarray:
    try:
        ev = torch.linalg.eigvals(W)
        return ev.detach().cpu().numpy()
    except Exception:
        reim, _ = torch.eig(W, eigenvectors=False)
        return reim[:, 0].numpy() + 1j * reim[:, 1].numpy()


def _find_run_entries(
    hidden_init: str, input_type: str
) -> List[Tuple[str, Path, Path]]:
    """
    Returns a list of (run_id, run_dir, ckpt_path) for runs that have the checkpoint.
    """
    base = MODEL_ROOT / hidden_init / input_type / MULTIRUNS_DIR
    if not base.exists():
        return []
    out = []
    for d in sorted(base.glob(f"{RUN_PREFIX}*")):
        ckpt = d / MODEL_FNAME
        if ckpt.exists():
            rid = d.name.replace(RUN_PREFIX, "", 1)
            out.append((rid, d, ckpt))
    return out


def plot_eigs_single_run(
    hidden_init: str,
    input_type: str,
    run_id: str = None,  # e.g., "03"
    snapshot: str = "last",  # {"first","middle","last"}
    font_scale: float = 1.0,
) -> None:
    runs = _find_run_entries(hidden_init, input_type)
    if not runs:
        print("[WARN] No runs for %s/%s" % (hidden_init, input_type))
        return

    if run_id is None:
        rid, run_dir, ckpt = runs[0]
    else:
        match = [x for x in runs if x[0] == run_id]
        rid, run_dir, ckpt = match[0] if match else runs[0]

    files = _list_hidden_weight_files(run_dir)
    if files:
        p = _pick_snapshot(files, snapshot)
        W = _load_weight_tensor(p)
        subtitle = "%s" % p.name
    else:
        # fallback to the checkpoint's hidden W
        W = _load_hidden_W_from_ckpt(ckpt)
        subtitle = "%s (ckpt)" % ckpt.name

    W = _load_weight_tensor(p)
    # Heatmap
    vmax = float(torch.max(torch.abs(W)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im = axes[0].imshow(W, origin="upper", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"W ({hidden_init}, {input_type}, run {rid}, {snapshot})")
    axes[0].set_xlabel("pre")
    axes[0].set_ylabel("post")
    _style_axes(axes[0], font_scale=font_scale)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Eigs
    ev = eigvals_compat(W)
    r = float(np.max(np.abs(ev))) if ev.size else 0.0
    axes[1].scatter(ev.real, ev.imag, s=8)
    circ = plt.Circle((0, 0), r, fill=False, ls="--")
    axes[1].add_artist(circ)
    axes[1].axhline(0, lw=0.5, c="k")
    axes[1].axvline(0, lw=0.5, c="k")
    axes[1].set_aspect("equal", "box")
    axes[1].set_title(f"Eigvals (radius≈{r:.3f})")
    axes[1].set_xlabel("Re(λ)")
    axes[1].set_ylabel("Im(λ)")
    _style_axes(axes[1], font_scale=font_scale)

    fig.tight_layout()
    plt.show()


def _radial_hist(
    ev: np.ndarray, rmax: Optional[float], nbins: int
) -> Tuple[np.ndarray, np.ndarray]:
    radii = np.abs(ev)
    if rmax is None:
        rmax = float(radii.max()) if radii.size else 1.0
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist, _ = np.histogram(radii, bins=edges, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def _collect_eigs_for_setting(
    hidden_init: str, input_type: str, snapshot: str
) -> List[np.ndarray]:
    out = []
    for rid, run_dir, ckpt in _find_run_entries(hidden_init, input_type):
        files = _list_hidden_weight_files(run_dir)
        try:
            if files:
                p = _pick_snapshot(files, snapshot)
                W = _load_weight_tensor(p)
            else:
                W = _load_hidden_W_from_ckpt(ckpt)  # fallback
            ev = eigvals_compat(W)
            out.append(ev)
        except Exception as e:
            print(
                "[WARN] Skipping %s/%s run %s at %s: %s: %s"
                % (
                    hidden_init,
                    input_type,
                    rid,
                    (files[-1] if files else ckpt),
                    type(e).__name__,
                    e,
                )
            )
    return out


def plot_eigs_aggregates(
    hidden_inits: Union[str, List[str]],
    input_types: Union[str, List[str]],
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    snapshot: str = "last",  # {"first","middle","last"}
    nbins: int = 40,
    ncols: Optional[int] = 2,
    nrows: Optional[int] = None,
    figsize_each: Tuple[int, int] = (6, 4),
    font_scale: float = 1.0,
    show_radius: bool = True,
) -> None:
    """
    Each facet = one (hidden_init, input_type) setting (fixed snapshot).
    Within each facet: mean ± std (across runs) of the radial eigendensity p(|λ|).
    """
    # Build the cartesian list of settings as facets (one panel per setting)
    pairs = []
    labels = []
    H = hidden_inits if isinstance(hidden_inits, list) else [hidden_inits]
    I = input_types if isinstance(input_types, list) else [input_types]
    if facet_by == "hidden_init" and len(I) == 1:
        for h in H:
            pairs.append((h, I[0]))
            labels.append(h)
    elif facet_by == "input_type" and len(H) == 1:
        for it in I:
            pairs.append((H[0], it))
            labels.append(it)
    else:
        # cartesian when multiple of both (label both dims)
        for h in H:
            for it in I:
                pairs.append((h, it))
                labels.append(f"{h} | {it}")

    # Collect eigs and spectral radii per setting
    per_setting = []  # (label, centers, mean_hist, std_hist, mean_radius)
    # common rmax across runs in a setting so hist bins align; choose per-setting rmax as max radius across its runs
    for (h, it), lab in zip(pairs, labels):
        all_eigs = _collect_eigs_for_setting(h, it, snapshot=snapshot)
        if not all_eigs:
            print(f"[WARN] No eigs for {h}/{it}; skipping.")
            continue
        # choose rmax
        rmax = max(float(np.max(np.abs(ev))) for ev in all_eigs if ev.size)
        # build per-run histograms with same edges
        Hs = []
        for ev in all_eigs:
            centers, hist = _radial_hist(ev, rmax=rmax, nbins=nbins)
            Hs.append(hist)
        Hs = np.stack(Hs, axis=0)  # [runs, nbins]
        mean_h = Hs.mean(axis=0)
        std_h = Hs.std(axis=0, ddof=1) if Hs.shape[0] > 1 else np.zeros_like(mean_h)
        mean_r = np.mean([float(np.max(np.abs(ev))) for ev in all_eigs])
        per_setting.append((lab, centers, mean_h, std_h, mean_r, (h, it)))

    if not per_setting:
        print("[INFO] Nothing to plot.")
        return

    # Layout
    n = len(per_setting)
    nrows_eff, ncols_eff = _compute_grid(n, ncols, nrows)
    fig, axes = plt.subplots(
        nrows_eff,
        ncols_eff,
        figsize=(figsize_each[0] * ncols_eff, figsize_each[1] * nrows_eff),
        squeeze=False,
    )

    # Plot each facet
    for idx, (lab, centers, mean_h, std_h, mean_r, (h, it)) in enumerate(per_setting):
        r, c = divmod(idx, ncols_eff)
        ax = axes[r][c]
        ax.plot(centers, mean_h, lw=2.0, label="mean density")
        ax.fill_between(
            centers,
            mean_h - std_h,
            mean_h + std_h,
            alpha=0.25,
            linewidth=0,
            label="±1 SD",
        )
        if show_radius:
            ax.axvline(
                mean_r, color="r", ls="--", lw=1.0, label=f"mean radius={mean_r:.3f}"
            )
        ax.set_xlabel("|λ|")
        ax.set_ylabel("density")
        ax.set_title(f"{lab}  [{h}, {it}, {snapshot}]")
        _style_axes(ax, font_scale=font_scale)
        ax.legend(fontsize=10 * font_scale)

    # Hide empties
    total = nrows_eff * ncols_eff
    for j in range(n, total):
        r, c = divmod(j, ncols_eff)
        axes[r][c].axis("off")

    fig.tight_layout()
    plt.show()


#########################################################################################################################################################################################       Replay      ################################### ################################################################################
################################################################################
# -------------------
# I/O
# -------------------
_INPUT_SUFFIX = {
    "gaussian": "1",
    "small-gaussian": "5gauss",
    "onehot": "1hot",
    "khot": "5hot",
}


def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)


def _dataset_filename(N: int, SeqN: int, input_type: str) -> str:
    key = input_type.lower()
    if key not in _INPUT_SUFFIX:
        raise ValueError(
            "Unknown input_type %r. Expected one of: %s"
            % (input_type, ", ".join(_INPUT_SUFFIX.keys()))
        )
    suf = _INPUT_SUFFIX[key]
    return "Ns%d_SeqN%d_%s.pth.tar" % (N, SeqN, suf)


def _load_dataset_X_and_Y(
    N: int = 100,
    SeqN: int = 100,
    input_type: str = "gaussian",
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the dataset that matches `input_type` and returns:
      X_in:  (1, T, N) where T = SeqN-1
      Y_true:(1, T, N) = Target[:, 1:, :]
    """
    base = _as_path(data_dir) or DATA_DIR
    fname = base / _dataset_filename(N, SeqN, input_type)
    if not fname.exists():
        raise FileNotFoundError(
            "Dataset not found at %s (module_dir=%s)" % (str(fname), str(MODULE_DIR))
        )
    pkg = torch.load(fname, map_location="cpu")
    X_mini = pkg["X_mini"]  # (1, SeqN, N)
    Target = pkg["Target_mini"]  # (1, SeqN, N)
    X_in = X_mini[:, :-1, :]  # (1, T, N)
    Y_true = Target[:, 1:, :]  # (1, T, N)
    return X_in, Y_true


def _facet_pairs(hidden_inits, input_types, facet_by):
    """
    Returns a list of (h, i) settings to plot, and the facet order.
    Each facet corresponds to exactly one (hidden_init, input_type) setting.
    """
    H = _to_list(hidden_inits)
    I = _to_list(input_types)
    assert facet_by in {"hidden_init", "input_type"}

    pairs = []
    facet_labels = []

    if facet_by == "hidden_init":
        # Prefer: multiple hidden_inits, single input_type
        if len(I) == 1:
            for h in H:
                pairs.append((h, I[0]))
                facet_labels.append(h)
        else:
            # Cartesian (warn informally); each panel is one (h,i)
            for h, it in product(H, I):
                pairs.append((h, it))
                facet_labels.append("%s | %s" % (h, it))
    else:  # facet_by == "input_type"
        if len(H) == 1:
            for it in I:
                pairs.append((H[0], it))
                facet_labels.append(it)
        else:
            for h, it in product(H, I):
                pairs.append((h, it))
                facet_labels.append("%s | %s" % (it, h))

    return pairs, facet_labels


# -------------------
# Core replay primitives
# -------------------
def make_noise_like(
    X_in: torch.Tensor,
    scale: float = 0.01,
    seed: Optional[int] = 0,
) -> torch.Tensor:
    """
    X_in: (1, T, N) tensor; noise std = scale * max(|X_in|)
    """
    rng = np.random.default_rng(seed)
    T, N = X_in.shape[1], X_in.shape[2]
    sigma = float(torch.as_tensor(X_in).abs().max().item()) * scale
    Z = rng.normal(0.0, sigma, size=(1, T, N)).astype(np.float32)
    return torch.from_numpy(Z)


def forward_sequence(
    state_dict: Dict[str, torch.Tensor],
    X: torch.Tensor,  # (1, T, N)
    N: int,
    HN: int,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run Elman over sequence X (no grad). Returns outputs (1, T, N).
    """
    model = Elman(N, HN, N).to(device)
    if activation is not None:
        model.act = activation
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    X = X.to(device)
    T = X.shape[1]
    h = torch.zeros(1, 1, HN, device=device)
    outs = torch.empty(1, T, N, device=device)
    with torch.no_grad():
        for t in range(T):
            o, h = model(X[:, t : t + 1, :], h)
            outs[:, t, :] = o
    return outs.detach().cpu()


# -------------------
# Single-run replay (optionally plot)
# -------------------
def replay_single_run(
    model_path: Union[str, Path],
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    input_type: str = "gaussian",
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    *,
    data_dir: Optional[Union[str, Path]] = DATA_DIR,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
    plot: bool = True,
    font_scale: float = 1.0,
    title_tag: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_noise: (N, T), Y_true: (N, T), Y_out: (N, T) as numpy arrays.
    """
    X_in, Y_true = _load_dataset_X_and_Y(
        N=N, SeqN=SeqN, input_type=input_type, data_dir=data_dir
    )
    X_noise = make_noise_like(X_in, scale=noise_scale, seed=noise_seed)

    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    Y_out = forward_sequence(
        state_dict, X_noise, N=N, HN=HN, activation=activation, device=device
    )

    # (1,T,N) -> (N,T) for plotting (neurons x time)
    Xn = X_noise[0].T.numpy()
    Yt = Y_true[0].T.numpy()
    Yo = Y_out[0].T.numpy()

    if plot:
        data = [Xn, Yt, Yo]
        titles = [
            "Input noise ($x_t$)",
            "Target ($y_t$: %s)" % input_type,
            "Output on noise ($\\hat{y}_t$)",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        vmaxs = [np.abs(d).max() for d in data]
        for i, ax in enumerate(axes):
            im = ax.imshow(data[i], aspect="auto")
            ax.set_title(titles[i])
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _style_cbar(cbar, font_scale=font_scale)
        supt = title_tag if title_tag else "Replay"
        fig.suptitle(supt, fontsize=16 * font_scale)
        fig.tight_layout()
        plt.show()

    return Xn, Yt, Yo


# -------------------
# Multirun aggregation: mean/std of replay OUTPUT on the same noise
# -------------------
def compute_replay_for_run(
    model_path: Union[str, Path],
    X_noise: torch.Tensor,  # (1, T, N)
    N=100,
    HN=100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
) -> np.ndarray:
    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    Y_out = forward_sequence(
        state_dict, X_noise, N=N, HN=HN, activation=activation, device=device
    )  # (1,T,N)
    return Y_out[0].T.numpy()  # (N,T)


def collect_replay_outputs_for_setting(
    hidden_init: str,
    input_type: str,
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    data_dir: Optional[Union[str, Path]] = DATA_DIR,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xn: (N,T), Yt: (N,T), list_of_Yo: List[(run_id, (N,T))]
    All runs use the SAME noise for comparability.
    """
    # Prepare deterministic noise & target
    X_in, Y_true = _load_dataset_X_and_Y(
        N=N, SeqN=SeqN, input_type=input_type, data_dir=data_dir
    )
    X_noise = make_noise_like(X_in, scale=noise_scale, seed=noise_seed)

    paths = _find_run_paths(hidden_init, input_type)
    outs = []
    for rid, p in paths:
        try:
            Yo = compute_replay_for_run(
                p, X_noise, N=N, HN=HN, activation=activation, device=device
            )
            outs.append((rid, Yo))
        except Exception as e:
            print(f"[WARN] Skipping {hidden_init}/{input_type} run {rid}: {e}")
    if not outs:
        return X_noise[0].T.numpy(), Y_true[0].T.numpy(), []

    return X_noise[0].T.numpy(), Y_true[0].T.numpy(), outs


def aggregate_replay_outputs(outs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    outs: list of (N,T) arrays
    Returns: (mean_out, std_out) both (N,T)
    """
    stack = np.stack(outs, axis=0)  # [n_runs, N, T]
    mean_o = stack.mean(axis=0)
    std_o = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
    return mean_o, std_o


# -------------------
# Faceted plotting of aggregated replay outputs (mean/std)
# -------------------
def _compute_grid(
    n: int, ncols: Optional[int], nrows: Optional[int]
) -> Tuple[int, int]:
    if ncols is None and nrows is None:
        ncols, nrows = 2, math.ceil(n / 2)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_replay_aggregates(
    hidden_inits: Union[str, List[str]],
    input_types: Union[str, List[str]],
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    show: str = "both",  # {"mean","std","both"}
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    data_dir: Optional[Union[str, Path]] = None,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
    ncols: Optional[int] = 2,
    nrows: Optional[int] = None,
    figsize_each: Tuple[int, int] = (5, 4),
    cmap_mean: str = "viridis",
    cmap_std: str = "magma",
    font_scale: float = 1.0,
) -> None:
    """
    Each facet = one (hidden_init, input_type) **setting**.
    Inside each facet: aggregate **across runs only** (mean/std) for that setting.
    No averaging across different hidden_inits or input_types.
    """
    # Build (h,i) settings per facet
    pairs, facet_labels = _facet_pairs(hidden_inits, input_types, facet_by)

    # Collect stats per setting
    setting_stats = {}  # (h,i) -> (mean_out (N,T), std_out (N,T))
    for (h, it), label in zip(pairs, facet_labels):
        Xn, Yt, outs = collect_replay_outputs_for_setting(
            h,
            it,
            N=N,
            SeqN=SeqN,
            HN=HN,
            activation=activation,
            device=device,
            data_dir=data_dir,
            noise_scale=noise_scale,
            noise_seed=noise_seed,
        )
        if not outs:
            print("[WARN] No runs for setting %s/%s; skipping." % (h, it))
            continue
        mean_o, std_o = aggregate_replay_outputs([o for _, o in outs])
        setting_stats[(h, it)] = (mean_o, std_o)

    if not setting_stats:
        print("[INFO] Nothing to plot.")
        return

    # Global color scales across panels
    means = [m for (m, s) in setting_stats.values()]
    stds = [s for (m, s) in setting_stats.values()]
    max_abs_mean = max(float(np.abs(m).max()) for m in means)
    max_std = max(float(s.max()) for s in stds)
    vmin_m, vmax_m = -max_abs_mean, +max_abs_mean
    vmin_s, vmax_s = 0.0, max_std if max_std > 0 else 1e-12

    # Grid dims
    import math

    n = len(setting_stats)
    if ncols is None and nrows is None:
        ncols, nrows = 2, math.ceil(n / 2)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)

    # Order facets per the labels list
    ordered_keys = []
    for (h, it), label in zip(pairs, facet_labels):
        if (h, it) in setting_stats:
            ordered_keys.append(((h, it), label))

    # MEAN figure
    if show in {"mean", "both"}:
        fig_m, axes_m = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        for idx, ((h, it), label) in enumerate(ordered_keys):
            r, c = divmod(idx, ncols)
            ax = axes_m[r][c]
            mean_o, _ = setting_stats[(h, it)]
            im = ax.imshow(mean_o, cmap=cmap_mean, vmin=0, vmax=1, aspect="auto")
            ax.set_title("%s = %s — mean" % (facet_by, label))
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            if c == ncols - 1:
                cbar = fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)
        # hide empties
        total = nrows * ncols
        for j in range(len(ordered_keys), total):
            r, c = divmod(j, ncols)
            axes_m[r][c].axis("off")
        fig_m.tight_layout()
        plt.show()

    # STD figure
    if show in {"std", "both"}:
        fig_s, axes_s = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        for idx, ((h, it), label) in enumerate(ordered_keys):
            r, c = divmod(idx, ncols)
            ax = axes_s[r][c]
            _, std_o = setting_stats[(h, it)]
            im = ax.imshow(
                std_o, cmap=cmap_std, vmin=vmin_s, vmax=vmax_s, aspect="auto"
            )
            ax.set_title("%s = %s — std" % (facet_by, label))
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            if c == ncols - 1:
                cbar = fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)
        total = nrows * ncols
        for j in range(len(ordered_keys), total):
            r, c = divmod(j, ncols)
            axes_s[r][c].axis("off")
        fig_s.tight_layout()
        plt.show()


#########################################################################################################################################################################################       Prediction          ########################### ################################################################################
################################################################################


# -------------------
# Core primitives
# -------------------
def make_noise_like(
    X_in: torch.Tensor,
    scale: float = 0.01,
    seed: Optional[int] = 0,
) -> torch.Tensor:
    """
    X_in: (1, T, N) tensor; noise std = scale * max(|X_in|)
    """
    rng = np.random.default_rng(seed)
    T, N = X_in.shape[1], X_in.shape[2]
    sigma = float(torch.as_tensor(X_in).abs().max().item()) * scale
    Z = rng.normal(0.0, sigma, size=(1, T, N)).astype(np.float32)
    return torch.from_numpy(Z)


def forward_sequence(
    state_dict: Dict[str, torch.Tensor],
    X: torch.Tensor,  # (1, T, N)
    N: int,
    HN: int,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run Elman over sequence X (no grad). Returns outputs (1, T, N).
    """
    model = Elman(N, HN, N).to(device)
    if activation is not None:
        model.act = activation
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    X = X.to(device)
    T = X.shape[1]
    h = torch.zeros(1, 1, HN, device=device)
    outs = torch.empty(1, T, N, device=device)
    with torch.no_grad():
        for t in range(T):
            o, h = model(X[:, t : t + 1, :], h)
            outs[:, t, :] = o
    return outs.detach().cpu()


# -------------------
# Single-run prediction (optionally plot)
# -------------------
def prediction_single_run(
    model_path: Union[str, Path],
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    input_type: str = "gaussian",
    prefix_T: int = 10,  # how many initial timesteps use true inputs
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    *,
    data_dir: Optional[Union[str, Path]] = None,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
    plot: bool = True,
    font_scale: float = 1.0,
    title_tag: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xpred: (N, T), Y_true: (N, T), Y_out: (N, T) as numpy arrays.
    """
    # Load dataset matching the encoding
    X_in, Y_true = _load_dataset_X_and_Y(
        N=N, SeqN=SeqN, input_type=input_type, data_dir=data_dir
    )
    T = X_in.shape[1]
    prefix_T = max(0, min(prefix_T, T))

    # Build prediction input: noise with a real prefix
    X_noise = make_noise_like(X_in, scale=noise_scale, seed=noise_seed)
    X_pred = X_noise.clone()
    if prefix_T > 0:
        X_pred[:, :prefix_T, :] = X_in[:, :prefix_T, :]

    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    Y_out = forward_sequence(
        state_dict, X_pred, N=N, HN=HN, activation=activation, device=device
    )

    # (1,T,N) -> (N,T)
    Xin = X_pred[0].T.numpy()
    Yt = Y_true[0].T.numpy()
    Yo = Y_out[0].T.numpy()

    if plot:
        data = [Xin, Yt, Yo]
        titles = [
            "Prediction input ($x_t$)",
            "Target (%s)" % input_type,
            "Output on prediction input ($\\hat{y}_t$)",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for i, ax in enumerate(axes):
            im = ax.imshow(data[i], aspect="auto")
            ax.set_title(titles[i])
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _style_cbar(cbar, font_scale=font_scale)
        supt = (
            title_tag
            if title_tag
            else "Prediction (%s, prefix_T=%d)" % (input_type, prefix_T)
        )
        fig.suptitle(supt, fontsize=16 * font_scale)
        fig.tight_layout()
        plt.show()

    return Xin, Yt, Yo


# -------------------
# Multirun aggregation: mean/std of prediction OUTPUT on shared X_pred
# -------------------
def compute_prediction_for_run(
    model_path: Union[str, Path],
    X_pred: torch.Tensor,  # (1, T, N)
    N=100,
    HN=100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
) -> np.ndarray:
    ckpt = _load_ckpt(model_path, map_location=device)
    state_dict = ckpt["state_dict"]
    Y_out = forward_sequence(
        state_dict, X_pred, N=N, HN=HN, activation=activation, device=device
    )  # (1,T,N)
    return Y_out[0].T.numpy()  # (N,T)


def collect_prediction_outputs_for_setting(
    hidden_init: str,
    input_type: str,
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    data_dir: Optional[Union[str, Path]] = None,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
    prefix_T: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Returns:
      Xpred: (N,T), Y_true: (N,T), list_of_Yo: List[(run_id, (N,T))]
    All runs use the SAME X_pred for comparability.
    """
    X_in, Y_true = _load_dataset_X_and_Y(
        N=N, SeqN=SeqN, input_type=input_type, data_dir=data_dir
    )
    T = X_in.shape[1]
    prefix_T = max(0, min(prefix_T, T))

    X_noise = make_noise_like(X_in, scale=noise_scale, seed=noise_seed)
    X_pred = X_noise.clone()
    if prefix_T > 0:
        X_pred[:, :prefix_T, :] = X_in[:, :prefix_T, :]

    paths = _find_run_paths(hidden_init, input_type)
    outs: List[Tuple[str, np.ndarray]] = []
    for rid, p in paths:
        try:
            Yo = compute_prediction_for_run(
                p, X_pred, N=N, HN=HN, activation=activation, device=device
            )
            outs.append((rid, Yo))
        except Exception as e:
            print(
                f"[WARN] Skipping {hidden_init}/{input_type} run {rid} at {p}: {type(e).__name__}: {e}"
            )
    return X_pred[0].T.numpy(), Y_true[0].T.numpy(), outs


def aggregate_prediction_outputs(
    outs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    outs: list of (N,T) arrays
    Returns: (mean_out, std_out) both (N,T)
    """
    if not outs:
        raise ValueError("No outputs to aggregate.")
    stack = np.stack(outs, axis=0)  # [n_runs, N, T]
    mean_o = stack.mean(axis=0)
    std_o = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
    return mean_o, std_o


# -------------------
# Faceted plotting of aggregated prediction outputs (mean/std)
# -------------------
def _compute_grid(
    n: int, ncols: Optional[int], nrows: Optional[int]
) -> Tuple[int, int]:
    if ncols is None and nrows is None:
        ncols, nrows = 2, math.ceil(n / 2)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_prediction_aggregates(
    hidden_inits: Union[str, List[str]],
    input_types: Union[str, List[str]],
    facet_by: str = "input_type",  # {"input_type","hidden_init"}
    show: str = "both",  # {"mean","std","both"}
    N: int = 100,
    SeqN: int = 100,
    HN: int = 100,
    activation: Optional[nn.Module] = nn.Sigmoid(),
    device: str = "cpu",
    data_dir: Optional[Union[str, Path]] = None,
    noise_scale: float = 0.01,
    noise_seed: Optional[int] = 0,
    prefix_T: int = 10,
    ncols: Optional[int] = 2,
    nrows: Optional[int] = None,
    figsize_each: Tuple[int, int] = (5, 4),
    cmap_mean: str = "viridis",
    cmap_std: str = "magma",
    font_scale: float = 1.0,
) -> None:
    """
    Each facet = one (hidden_init, input_type) setting.
    Inside each facet: aggregate across runs only (mean/std) on a shared X_pred.
    """
    pairs, facet_labels = _facet_pairs(hidden_inits, input_types, facet_by)

    setting_stats = {}  # (h,i) -> (mean_out, std_out)
    for (h, it), label in zip(pairs, facet_labels):
        Xpred, Yt, outs = collect_prediction_outputs_for_setting(
            h,
            it,
            N=N,
            SeqN=SeqN,
            HN=HN,
            activation=activation,
            device=device,
            data_dir=data_dir,
            noise_scale=noise_scale,
            noise_seed=noise_seed,
            prefix_T=prefix_T,
        )
        if not outs:
            print("[WARN] No runs for setting %s/%s; skipping." % (h, it))
            continue
        mean_o, std_o = aggregate_prediction_outputs([o for _, o in outs])
        setting_stats[(h, it)] = (mean_o, std_o)

    if not setting_stats:
        print("[INFO] Nothing to plot.")
        return

    means = [m for (m, s) in setting_stats.values()]
    stds = [s for (m, s) in setting_stats.values()]
    max_abs_mean = max(float(np.abs(m).max()) for m in means)
    max_std = max(float(s.max()) for s in stds)
    vmin_m, vmax_m = -max_abs_mean, +max_abs_mean
    vmin_s, vmax_s = 0.0, max_std if max_std > 0 else 1e-12

    import math

    n = len(setting_stats)
    if ncols is None and nrows is None:
        ncols, nrows = 2, math.ceil(n / 2)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    else:
        if ncols * nrows < n:
            nrows = math.ceil(n / ncols)

    ordered_keys = []
    for (h, it), label in zip(pairs, facet_labels):
        if (h, it) in setting_stats:
            ordered_keys.append(((h, it), label))

    # MEAN
    if show in {"mean", "both"}:
        fig_m, axes_m = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        for idx, ((h, it), label) in enumerate(ordered_keys):
            r, c = divmod(idx, ncols)
            ax = axes_m[r][c]
            mean_o, _ = setting_stats[(h, it)]
            im = ax.imshow(mean_o, cmap=cmap_mean, vmin=0, vmax=1, aspect="auto")
            ax.set_title(
                "%s = %s — output mean (prefix_T=%d)" % (facet_by, label, prefix_T)
            )
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            if c == ncols - 1:
                cbar = fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)
        total = nrows * ncols
        for j in range(len(ordered_keys), total):
            r, c = divmod(j, ncols)
            axes_m[r][c].axis("off")
        fig_m.tight_layout()
        plt.show()

    # STD
    if show in {"std", "both"}:
        fig_s, axes_s = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_each[0] * ncols, figsize_each[1] * nrows),
            squeeze=False,
        )
        for idx, ((h, it), label) in enumerate(ordered_keys):
            r, c = divmod(idx, ncols)
            ax = axes_s[r][c]
            _, std_o = setting_stats[(h, it)]
            im = ax.imshow(
                std_o, cmap=cmap_std, vmin=vmin_s, vmax=vmax_s, aspect="auto"
            )
            ax.set_title(
                "%s = %s — output std (prefix_T=%d)" % (facet_by, label, prefix_T)
            )
            ax.set_xlabel("time t")
            ax.set_ylabel("neurons")
            _style_axes(ax, font_scale=font_scale)
            if c == ncols - 1:
                cbar = fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _style_cbar(cbar, font_scale=font_scale)
        total = nrows * ncols
        for j in range(len(ordered_keys), total):
            r, c = divmod(j, ncols)
            axes_s[r][c].axis("off")
        fig_s.tight_layout()
        plt.show()
