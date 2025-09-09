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

hidden_weights_inits = [
    "he",
    "shift",
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
        return torch.load(p)
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


def _extract_history(ckpt) -> Optional[Dict[str, List]]:
    """Returns dict with keys present in history: 'epoch', 'grad_norm', 'loss', etc. Only keeps the list-like fields of equal length to 'epoch'."""
    if ckpt is None:
        return None
    history = ckpt.get("history", None)
    if not history or "epoch" not in history or not isinstance(history["epoch"], list):
        return None
    L = len(history["epoch"])
    out = {"epoch": list(map(int, history["epoch"]))}
    for k, v in history.items():
        if k == "epoch":
            continue
        if isinstance(v, list) and len(v) == L:
            out[k] = list(v)
    return out


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


def _attach_epoch_to_list(list: List[Dict], epoch_list=None) -> Optional[pd.DataFrame]:
    """Given a list of dicts (e.g. grad snapshots), attach epoch number if available."""
    if not list:
        return None
    df = pd.DataFrame(list)
    if epoch_list and len(epoch_list) == len(list):
        df["epoch"] = epoch_list
    else:
        df["epoch"] = range(len(list))
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


def _reduce_grad_snapshot_paramwise(d: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Reduce a single gradient snapshot (param -> stats dict) into global scalars. Keeps robust, comparable summaries"""
    if not d:
        return {}
    keys = ["mean", "std", "l2_norm", "mean_sq", "max_abs", "sparsity"]
    out = {f"grad_{k}_sum": 0.0 for k in keys}
    out.update({f"grad_{k}_mean": 0.0 for k in keys})
    out.update({f"grad_{k}_max": float("-inf") for k in keys})
    count = 0
    for stats in d.values():
        count += 1
        for k in keys:
            v = float(stats.get(k, 0.0))
            out[f"grad_{k}_sum"] += v
            out[f"grad_{k}_max"] = max(out[f"grad_{k}_max"], v)
        if count > 0:
            for k in keys:
                out[f"grad_{k}_mean"] = out[f"grad_{k}_sum"] / count
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
    """
    base = model_root / hidden_init / input_type
    per_run_rows = []
    losses_all = []
    metrics_df_list = []
    grad_df_list = []

    for run_id, p in _iter_multirun_files(base):
        if run_id == "00":
            print(f"Run {run_id}: {p}")
            # Load checkpoint
            ckpt = _load_torch(p)
            print(f"  Keys: {list(ckpt.keys())}")

            # Extract loss series
            loss_series = _extract_loss_series(ckpt)
            if loss_series:
                losses_all.append(loss_series)

            # Load loss metrics from loss_series
            m = _metrics_from_loss(loss_series)
            if m:
                per_run_rows.append(
                    {
                        "hidden_init": hidden_init,
                        "input_type": input_type,
                        "run_kind": "multirun",
                        "run_id": run_id,
                        "path": str(p),
                        **m,
                    }
                )

            # Get metrics time series (over recorded epochs)
            mlist = _extract_metrics_list(ckpt)
            if mlist:
                metrics_df_list.append(_metrics_df_from_list(mlist, run_id))

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

            # Get history (epoch, grad_norm, loss, etc.)
            history = _extract_history(ckpt)
            if history:
                hist_df = pd.DataFrame(history)
                hist_df["run_id"] = run_id
        return per_run_rows, {
            "losses": losses_all,
            "metrics_df_list": metrics_df_list,
            "grad_df_list": grad_df_list,
            "history_df": hist_df,
        }


# -------------------
# High-level collection across all settings
# -------------------
def collect_all(h_inits=None, in_types=None):
    h_inits = h_inits or hidden_weight_inits
    in_types = in_types or input_types

    all_rows = []
    ts_bucket = {}  # (hidden_init, input_type) -> per-run timeseries dict
    for h in h_inits:
        for it in in_types:
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
                "t_to_110pct_best",
            ]
        )
    )

    # Aggregates over multiruns (per setting)
    agg_rows = []
    if not per_run_df.empty:
        for (h, it), group in per_run_df.groupby(["hidden_init", "input_type"]):
            g_multi = group[group["run_kind"] == "multi"]
            if g_multi.empty:
                agg_rows.append(
                    {
                        "hidden_init": h,
                        "input_type": it,
                        "n_runs": 0,
                        "final_loss_mean": np.nan,
                        "final_loss_std": np.nan,
                        "best_loss_mean": np.nan,
                        "best_loss_std": np.nan,
                        "best_epoch_mean": np.nan,
                        "best_epoch_std": np.nan,
                        "loss_auc_mean": np.nan,
                        "loss_auc_std": np.nan,
                        "t_to_110pct_best_mean": np.nan,
                        "t_to_110pct_best_std": np.nan,
                    }
                )
            else:

                def s(col):
                    return (
                        float(g_multi[col].mean()),
                        float(g_multi[col].std(ddof=1))
                        if g_multi.shape[0] > 1
                        else 0.0,
                    )

                fl_m, fl_s = s("final_loss")
                bl_m, bl_s = s("best_loss")
                be_m, be_s = s("best_epoch")
                auc_m, auc_s = s("loss_auc")
                tt_m, tt_s = s("t_to_110pct_best")
                agg_rows.append(
                    {
                        "hidden_init": h,
                        "input_type": it,
                        "n_runs": int(g_multi.shape[0]),
                        "final_loss_mean": fl_m,
                        "final_loss_std": fl_s,
                        "best_loss_mean": bl_m,
                        "best_loss_std": bl_s,
                        "best_epoch_mean": be_m,
                        "best_epoch_std": be_s,
                        "loss_auc_mean": auc_m,
                        "loss_auc_std": auc_s,
                        "t_to_110pct_best_mean": tt_m,
                        "t_to_110pct_best_std": tt_s,
                    }
                )
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
