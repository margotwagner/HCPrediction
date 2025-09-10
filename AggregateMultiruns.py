# ============================================================
# Multirun analysis: combine and save metrics
# ============================================================

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import pandas as pd

# -------------------
# CONFIG
# -------------------
data_dir = Path("../data/Ns100_SeqN100/")
model_root = Path("../Elman_SGD/Remap_predloss/N100T100/")

hidden_weight_inits = [
    "he",
    "cyclic-shift",
    "shift",
    "cmh",
    "mh",
    "ctridiag",
    "tridiag",
    "orthog",
]
input_types = ["gaussian", "onehot", "khot", "small-gaussian"]

SINGLE_DIR = "single-run"
MULTIRUNS_DIR = "multiruns"
RUN_PREFIX = "run_"
MODEL_FNAME = "Ns100_SeqN100_predloss_full.pth.tar"
HIDDEN_WEIGHTS_SUBDIR = "hidden-weights"

# Output dir for CSVs
CSV_ROOT = model_root / "csv"


# -------------------
# I/O helpers
# -------------------


def _load_torch(p):
    """Load torch file; returns None if missing or corrupt."""
    try:
        return torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Could not load {p}: {e}")
        return None


def _extract_loss_series(ckpt) -> Optional[List[float]]:
    """Extract loss series from checkpoint dict. Returns None if not found."""
    if ckpt is None:
        return None
    if "loss" in ckpt:
        return [float(x) for x in ckpt["loss"]]
    else:
        print("[WARN] No loss series found in checkpoint.")
        return None


def _extract_metrics_list(ckpt) -> Optional[List[Dict]]:
    """Extract metrics list from checkpoint dict. Returns None if not found."""
    if ckpt is None:
        return None
    # save metric as list of dicts (per recorded epoch)
    m = ckpt.get("metrics", None)
    if isinstance(m, list) and (len(m) == 0 or isinstance(m[0], dict)):
        return m
    return None


def _extract_grad_list(ckpt) -> Optional[List[Dict]]:
    """Extract gradient norms list from checkpoint dict. Returns None if not found."""
    if ckpt is None:
        return None
    g = ckpt.get("grad_list", None)
    if isinstance(g, list):
        return g
    return None


def _extract_history(ckpt, keep=("epoch", "loss", "grad_norm"), summarize=False):
    """
    Return a compact history dict with only scalar series that align with 'epoch'.

    If summarize=True, include light summaries for large tensors (means/stds), computed per epoch to a *few* numbers.
    """
    if ckpt is None:
        return None
    history = ckpt.get("history")
    if not history or "epoch" not in history or not isinstance(history["epoch"], list):
        return None
    L = len(history["epoch"])
    out = {"epoch": [int(e) for e in history["epoch"]]}

    # keep scalar lists
    for k in keep:
        if k == "epoch":
            continue
        v = history.get(k)
        if isinstance(v, list) and len(v) == L:
            # allow Python floats/ints or 0-dim tensors
            cleaned = []
            for x in v:
                if isinstance(x, (float, int)):
                    cleaned.append(float(x))
                elif torch.is_tensor(x) and x.numel() == 1:
                    cleaned.append(float(x.item()))
                else:
                    # skip non-scalar entries
                    cleaned.append(float("nan"))
            out[k] = cleaned

    if summarize:
        # Example: summarize hidden and y_hat by mean
        def _summarize_tensor_list(tlist, name):
            if not isinstance(tlist, list) or len(tlist) != L:
                return
            means = []
            stds = []
            for t in tlist:
                if torch.is_tensor(t):
                    means.append(float(t.mean().item()))
                    stds.append(float(t.std(unbiased=False).item()))
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))
            out[f"{name}_mean"] = means
            out[f"{name}_std"] = stds

        _summarize_tensor_list(history.get("hidden"), "hidden")
        _summarize_tensor_list(history.get("y_hat"), "y_hat")
    return out


def _iter_multirun_files(base_dir: Path):
    """Yield (run_id, path) pairs for each run file found under multiruns/run_XX."""
    multiruns_dir = base_dir / MULTIRUNS_DIR
    if not multiruns_dir.exists():
        print(f"[WARN] Multirun dir does not exist: {multiruns_dir}")
        return
    for run_dir in sorted(multiruns_dir.glob(f"{RUN_PREFIX}*")):
        path = run_dir / MODEL_FNAME
        if path.exists():
            run_id = run_dir.name.replace(
                RUN_PREFIX, "", 1
            )  # Extract run ID ('00' from 'run_00')
            yield run_id, path


def _attach_epoch_to_list(lst: List[Dict], epoch_list=None) -> Optional[pd.DataFrame]:
    """Given a list of dicts (e.g. grad snapshots), attach epoch number if available."""
    if not lst:
        return None
    df = pd.DataFrame(lst)
    if epoch_list and len(epoch_list) == len(lst):
        df["epoch"] = epoch_list
    else:
        df["epoch"] = range(len(lst))
    return df


# -------------------
# Core metrics utilities
# -------------------
def _metrics_from_loss(loss: Optional[List[float]]) -> Optional[Dict[str, float]]:
    """Given a loss list, compute final loss, best loss and best_epoch (index)."""
    if not loss:
        return None
    final_loss = float(loss[-1])
    best_epoch = int(np.argmin(loss))
    best_loss = float(loss[best_epoch])
    # auc (lower is better); trapezoidal rule
    auc = float(np.trapz(loss, dx=1.0))
    # time-to-110% of best (how fast it gets close to best)
    threshold = 1.1 * best_loss
    t110 = int(next((i for i, v in enumerate(loss) if v <= threshold), len(loss) - 1))
    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "loss_auc": auc,
        "time_to_110pct_best": t110,
    }


def _metrics_df_from_list(metrics_list: List[Dict], run_id: str) -> pd.DataFrame:
    """Convert list of metrics dicts to a DataFrame, adding run_id column."""
    if not metrics_list:
        return pd.DataFrame()
    df = pd.DataFrame(metrics_list)
    df["run_id"] = run_id
    return df


def _summarize_metrics(
    metrics_df: pd.DataFrame, loss_series: Optional[List[float]]
) -> Dict[str, float]:
    """Summarize metrics DataFrame (over epochs) into per-run scalars.
    Returns a flat dict like: {'final_frob': ...}
    """
    if metrics_df is None or metrics_df.empty:
        return {}

    # Define which metrics to summarize if present
    metrics_cols = [
        "loss",
        "loss_batch_mean",
        "loss_batch_std",
        "frob",
        "drift_from_init",
        "spectral_radius",
        "spectral_norm",
        "min_singular",
        "cond_num",
        "orth_err",
        "w_max_abs",
        "w_sparsity",
        "act_mean",
        "act_std",
        "tanh_sat",
    ]
    metric_cols = [c for c in metrics_cols if c in metrics_df.columns]

    out = {}

    # final = last row
    last = metrics_df.iloc[-1]
    for c in metric_cols:
        out[f"final_{c}"] = float(last[c])

    # best = row at best loss epoch
    if loss_series and len(loss_series) == len(metrics_df):
        best_epoch = int(np.argmin(loss_series))
        best_row = metrics_df.iloc[best_epoch]
        for c in metric_cols:
            out[f"best_{c}"] = float(best_row[c])

    # global summaries
    for c in metric_cols:
        out[f"{c}_mean"] = float(metrics_df[c].mean())
        out[f"{c}_std"] = (
            float(metrics_df[c].std(ddof=1)) if len(metrics_df) > 1 else 0.0
        )
        out[f"{c}_max"] = float(metrics_df[c].max())
        out[f"{c}_min"] = float(metrics_df[c].min())

    return out


def _reduce_grad_snapshot_paramwise(
    snap: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Reduce nested gradient snapshot into flat scalars.
    Expects:
        snap["per_param"] : dict(param -> stats dict with keys: mean, std, etc)snap["per_group_norm] : dict(group -> float) (optional)
    """
    if not isinstance(snap, dict) or "per_param" not in snap:
        return ValueError("Gradient snapshot must be a dict with 'per_param' key.")

    per_param = snap.get("per_param", {})

    keys = ["mean", "std", "l2_norm", "mean_sq", "max_abs", "sparsity"]
    out = {f"grad_{k}_sum": 0.0 for k in keys}
    out.update({f"grad_{k}_max": float("-inf") for k in keys})

    count = 0
    for stats in per_param.values():
        count += 1
        for k in keys:
            v = float(stats.get(k, 0.0))
            out[f"grad_{k}_sum"] += v
            out[f"grad_{k}_max"] = max(out[f"grad_{k}_max"], v)

    # finalize means
    for k in keys:
        out[f"grad_{k}_mean"] = out[f"grad_{k}_sum"] / count if count > 0 else 0.0

    # surface per-group L2 if provided
    for grp, val in snap.get("per_group_norm", {}).items():
        out[f"grad_group_{grp}_l2_norm"] = float(val)

    return out


# -------------------
# Collection for one (hidden_init, input_type)
# -------------------
def collect_for_setting(hidden_init: str, input_type: str):
    """Collect data for a given (hidden_init, input_type) setting.

    Returns:
        per_run_rows: List of dicts with per-run summary metrics
        per_run_timeseries: Dict with keys:
            "losses": List of loss series (list of lists)
            "metrics_df_list": List of DataFrames with metrics time series
            "grad_df_list": List of DataFrames with gradient norms time series
            "history_df_list": List of DataFrames with history time series
    """
    base = model_root / hidden_init / input_type
    per_run_rows = []
    losses_all = []
    metrics_df_list = []
    grad_df_list = []
    history_df_list = []

    for run_id, p in _iter_multirun_files(base):
        print(f"Run {run_id}: {p}")
        # Load checkpoint
        ckpt = _load_torch(p)

        # Extract loss series
        loss_series = _extract_loss_series(ckpt)
        if loss_series:
            losses_all.append(loss_series)

        # Load loss metrics from loss_series
        m = _metrics_from_loss(loss_series)

        # Get metrics time series (over recorded epochs)
        mlist = _extract_metrics_list(ckpt)
        metrics_df = None
        if mlist:
            metrics_df = _metrics_df_from_list(mlist, run_id)
            metrics_df_list.append(metrics_df)

        # Summarize per-epoch metrics into per-run scalars and merge into row
        metrics_summary = _summarize_metrics(metrics_df, loss_series)
        if m:
            row = {
                "hidden_init": hidden_init,
                "input_type": input_type,
                "run_kind": "multirun",
                "run_id": run_id,
                "path": str(p),
                **m,
                **metrics_summary,
            }
            per_run_rows.append(row)

        # Get gradient norms time series (over recorded epochs)
        glist = _extract_grad_list(ckpt)
        if glist and isinstance(glist[0], dict):
            reduced = [_reduce_grad_snapshot_paramwise(snap) for snap in glist]
            gdf = _attach_epoch_to_list(
                reduced, epoch_list=ckpt.get("history", {}).get("epoch", None)
            )
            if gdf is not None:
                gdf["run_id"] = run_id
                grad_df_list.append(gdf)

        # Get history time series
        history = _extract_history(
            ckpt, keep=("epoch", "loss", "grad_norm"), summarize=False
        )
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["run_id"] = run_id
            history_df_list.append(hist_df)
    return per_run_rows, {
        "losses": losses_all,
        "metrics_df_list": metrics_df_list,
        "grad_df_list": grad_df_list,
        "history_df_list": history_df_list,
    }


# -------------------
# High-level collection across all settings
# -------------------
def collect_all(h_inits=None, in_types=None):
    h_inits = h_inits or hidden_weight_inits
    in_types = in_types or input_types

    all_rows = []
    ts_bucket = {}  # (hidden_init, input_type) -> per-run timeseries dict

    # Get per_run_row, losses, metrics, and gradients for each (hidden_init, input_type) setting
    for h in h_inits:
        for it in in_types:
            print(f"Collecting for (hidden_init={h}, input_type={it})")
            rows, ts = collect_for_setting(h, it)
            if rows:
                all_rows.extend(rows)
            ts_bucket[(h, it)] = ts
    per_run_df = (
        pd.DataFrame(all_rows)
        if all_rows
        else pd.DataFrame(
            columns=[
                "hidden_init",
                "input_type",
                "run_kind",
                "run_id",
                "path",
                "final_loss",
                "best_loss",
                "best_epoch",
                "loss_auc",
                "time_to_110pct_best",
            ]
        )
    )

    # Aggregate over multiruns (per setting)
    agg_rows = []
    if not per_run_df.empty:
        # identify all per-run scalar columns to aggregate by (exlude ids)
        exclude = {"hidden_init", "input_type", "run_kind", "run_id", "path"}
        scalar_cols = [
            c
            for c in per_run_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(per_run_df[c])
        ]

        for (h, it), group in per_run_df.groupby(["hidden_init", "input_type"]):
            g_multi = group[group["run_kind"] == "multirun"]
            row = {
                "hidden_init": h,
                "input_type": it,
                "num_runs": int(g_multi.shape[0]),
            }
            if g_multi.empty:
                for c in scalar_cols:
                    row[f"{c}_mean"] = np.nan
                    row[f"{c}_std"] = np.nan
            else:
                for c in scalar_cols:
                    row[f"{c}_mean"] = float(g_multi[c].mean())
                    row[f"{c}_std"] = (
                        float(g_multi[c].std(ddof=1)) if g_multi.shape[0] > 1 else 0.0
                    )
            agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    return per_run_df, agg_df, ts_bucket


# -------------------
# Plotting
# -------------------
'''
def plot_loss_single(
    torch_data: Dict, wh_type=None, i_type=None, save_dir: Optional[Path] = None
):
    loss = _extract_loss_series(torch_data)
    if not loss:
        print("[INFO] No 'loss' found.")
        return
    plt.figure(figsize=(6, 3))
    plt.plot(loss, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(
        f"Training Loss ({wh_type}, {i_type})"
        if wh_type and i_type
        else "Training Loss"
    )
    plt.tight_layout()
    if save_dir:
        _ensure_dir(save_dir)
        out = save_dir / f"loss_single_{wh_type}_{i_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
    plt.show()


def plot_loss_mean_std(
    loss_lists: List[List[float]],
    title: str,
    save_path: Optional[Path] = None,
    max_epochs: Optional[int] = None,
):
    if not loss_lists:
        print("[INFO] No losses to plot.")
        return
    arr = _stack_ragged_with_nan(loss_lists, max_len=max_epochs)  # [n_runs, T]
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    epochs = np.arange(mean.shape[0])

    plt.figure(figsize=(6, 3))
    plt.plot(epochs, mean, lw=2, label="Mean loss")
    plt.fill_between(
        epochs, mean - std, mean + std, alpha=0.25, linewidth=0, label="±1 SD"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()
    return epochs, mean, std


def plot_metrics_mean_std(
    metrics_df_list: List[pd.DataFrame],
    cols: List[str],
    title_prefix: str,
    save_dir: Optional[Path] = None,
):
    """
    Aggregates selected metric columns across runs over recorded epochs.
    Assumes each DF has an 'epoch' column (your saver does).
    """
    if not metrics_df_list:
        print("[INFO] No metrics DFs to aggregate.")
        return
    # Find common epoch grid
    all_epochs = sorted(
        set().union(*[set(df["epoch"]) for df in metrics_df_list if "epoch" in df])
    )
    if not all_epochs:
        print("[INFO] No 'epoch' column in metrics DFs.")
        return
    grid = pd.DataFrame({"epoch": all_epochs})

    # Align each DF to grid and stack
    aligned = []
    for df in metrics_df_list:
        if "epoch" not in df:
            continue
        aligned.append(grid.merge(df[["epoch"] + cols], on="epoch", how="left"))
    # Compute mean/std per epoch for each col
    agg = {"epoch": all_epochs}
    for c in cols:
        stack = np.vstack([d[c].to_numpy() for d in aligned])  # [n_runs, T] with NaNs
        agg[f"{c}_mean"] = np.nanmean(stack, axis=0)
        agg[f"{c}_std"] = (
            np.nanstd(stack, axis=0, ddof=1)
            if stack.shape[0] > 1
            else np.zeros(stack.shape[1])
        )
    agg_df = pd.DataFrame(agg)

    # Plot each col separately to keep visuals clean
    for c in cols:
        plt.figure(figsize=(6, 3))
        plt.plot(agg_df["epoch"], agg_df[f"{c}_mean"], lw=2, label=f"{c} mean")
        plt.fill_between(
            agg_df["epoch"],
            agg_df[f"{c}_mean"] - agg_df[f"{c}_std"],
            agg_df[f"{c}_mean"] + agg_df[f"{c}_std"],
            alpha=0.25,
            linewidth=0,
            label="±1 SD",
        )
        plt.xlabel("Epoch")
        plt.ylabel(c)
        plt.title(f"{title_prefix}: {c}")
        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = (
                save_dir / f"metrics_mean_std_{title_prefix.replace(' ', '_')}_{c}.png"
            )
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[saved] {out}")
        plt.show()
    return agg_df


# -------------------
# Comparative utilities
# -------------------
def best_seed_table(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best run (lowest best_loss) per (hidden_init, input_type)."""
    if per_run_df.empty:
        return pd.DataFrame()
    key = ["hidden_init", "input_type"]
    idx = per_run_df.groupby(key)["best_loss"].idxmin()
    return per_run_df.loc[
        idx,
        key
        + [
            "run_kind",
            "run_id",
            "best_loss",
            "best_epoch",
            "loss_auc",
            "t_to_110pct_best",
            "path",
        ],
    ].sort_values(key)


def rank_initializations(
    agg_df: pd.DataFrame, input_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Rank hidden inits by best_loss_mean (ascending) per input_type (or overall mean if None).
    """
    if agg_df.empty:
        return pd.DataFrame()
    df = agg_df.copy()
    if input_type is not None:
        df = df[df["input_type"] == input_type]
    # lower is better; if overall, average across inputs
    if input_type is None:
        df = df.groupby("hidden_init", as_index=False)["best_loss_mean"].mean()
    df["rank"] = df["best_loss_mean"].rank(method="min")
    return df.sort_values(["rank", "hidden_init"])


# -------------------
# Example driver
# -------------------
if __name__ == "__main__":
    # 1) Collect everything
    per_run_df, agg_df, ts_bucket = collect_all()

    print("\n=== Per-run metrics (first 20 rows) ===")
    print(per_run_df.head(20).to_string(index=False))

    print("\n=== Aggregates across multiruns ===")
    print(agg_df.to_string(index=False))

    # Save CSVs
    if not per_run_df.empty:
        _ensure_dir(CSV_ROOT)
        per_run_df.to_csv(CSV_ROOT / "per_run_metrics.csv", index=False)
        print(f"[saved] {CSV_ROOT / 'per_run_metrics.csv'}")
    if not agg_df.empty:
        _ensure_dir(CSV_ROOT)
        agg_df.to_csv(CSV_ROOT / "agg_metrics.csv", index=False)
        print(f"[saved] {CSV_ROOT / 'agg_metrics.csv'}")

    # 2) Plot mean ± SD loss curves per (hidden_init, input_type) where multiruns exist
    for (h, it), ts in ts_bucket.items():
        if not ts or not ts.get("losses"):
            continue
        title = f"Mean ± SD Loss ({h}, {it})"
        save_path = FIG_ROOT / h / it / f"loss_mean_std_{h}_{it}.png"
        plot_loss_mean_std(ts["losses"], title=title, save_path=save_path)

    # 3) Plot mean ± SD of selected *metrics* (from metrics list) across runs
    # Choose metrics you care about:
    metric_cols = [
        "spectral_radius",
        "frob",
        "orth_err",
        "act_mean",
        "act_std",
        "tanh_sat",
        "loss",
    ]
    for (h, it), ts in ts_bucket.items():
        mlist = ts.get("metrics_df_list", [])
        if not mlist:
            continue
        save_dir = FIG_ROOT / h / it / "metrics"
        plot_metrics_mean_std(
            mlist,
            cols=[c for c in metric_cols if any(c in df.columns for df in mlist)],
            title_prefix=f"{h}, {it}",
            save_dir=save_dir,
        )

    # 4) Best-seed table and ranking examples
    if not per_run_df.empty:
        print("\n=== Best seed per setting (by best_loss) ===")
        print(best_seed_table(per_run_df).to_string(index=False))

        print("\n=== Rank initializations by mean best_loss across inputs ===")
        print(rank_initializations(agg_df).to_string(index=False))

        # If you want per-input-type ranking:
        for it in input_types:
            r = rank_initializations(agg_df, input_type=it)
            if not r.empty:
                print(f"\n=== Rank initializations for input={it} ===")
                print(r.to_string(index=False))
'''
