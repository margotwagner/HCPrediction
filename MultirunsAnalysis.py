# ============================================================
# Multirun analysis: metrics, plots, and comparisons
# ============================================================

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -------------------
# CONFIG
# -------------------
data_dir = Path("../data/Ns100_SeqN100/")  # input data root (not strictly needed here)
model_root = Path("../Elman_SGD/Remap_predloss/N100T100/")  # model outputs root

hidden_weight_inits = ["he", "shift", "cyclic-shift"]  # extend if you have more
input_types = ["gaussian", "onehot", "khot", "small-gaussian"]

# Directory / file naming
SINGLE_DIR = "single-run"
MULTIRUNS_DIR = "multiruns"
RUN_PREFIX = "run_"
MODEL_FILENAME = "Ns100_SeqN100_predloss_full.pth.tar"
HIDDEN_WEIGHTS_SUBDIR = "hidden-weights"

# Results dirs
FIG_ROOT = model_root / "figs"
CSV_ROOT = model_root / "csv"


# -------------------
# File handling utility functions
# -------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _safe_load(p: Path):
    try:
        return torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Could not load {p}: {e}")
        return None


def _extract_loss_series(ckpt):
    if ckpt is None:
        return None
    if "loss" in ckpt:
        return [float(x) for x in ckpt["loss"]]
    else:
        print("[WARN] No loss series found in checkpoint.")
    return None


def _extract_history(ckpt) -> Optional[Dict]:
    if ckpt is None:
        return None
    return ckpt.get("history", None)


def _extract_metrics_list(ckpt) -> Optional[List[Dict]]:
    if ckpt is None:
        return None
    # metrics saved as a list of dicts (per recorded epoch)
    m = ckpt.get("metrics", None)
    if isinstance(m, list) and (len(m) == 0 or isinstance(m[0], dict)):
        return m
    return None


def _extract_grad_list(ckpt) -> Optional[List[Dict]]:
    if ckpt is None:
        return None
    g = ckpt.get("grad_list", None)
    if isinstance(g, list):
        return g
    return None


def _get_single_run_file(base: Path) -> Optional[Path]:
    fname = base / SINGLE_DIR / MODEL_FILENAME
    if fname.exists():
        return fname
    return None


def _iter_multirun_files(base: Path):
    mult = base / MULTIRUNS_DIR
    if not mult.exists():
        return
    for run_dir in sorted(mult.glob(f"{RUN_PREFIX}*")):
        p = run_dir / MODEL_FILENAME
        if p.exists():
            yield run_dir.name.replace(RUN_PREFIX, "", 1), p  # ('00', path)


# -------------------
# Metrics utility functions
# -------------------
def _metrics_from_loss(loss: Optional[List[float]]) -> Optional[Dict[str, float]]:
    if not loss:
        return None
    final_loss = float(loss[-1])
    best_epoch = int(np.argmin(loss))
    best_loss = float(loss[best_epoch])
    # area under curve (lower is better); trapezoid rule
    auc = float(np.trapz(loss, dx=1.0))
    # time-to-110% of best (how fast it gets "close" to best)
    thresh = 1.10 * best_loss
    t_close = int(next((i for i, v in enumerate(loss) if v <= thresh), len(loss) - 1))
    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "loss_auc": auc,
        "t_to_110pct_best": t_close,
    }


def _stack_ragged_with_nan(
    series_list: List[List[float]], max_len: Optional[int] = None
) -> np.ndarray:
    """Pad ragged sequences with NaNs to align for nanmean/nanstd."""
    if not series_list:
        return np.empty((0, 0))
    L = max(len(s) for s in series_list)
    if max_len is not None:
        L = min(L, max_len)
    arr = np.full((len(series_list), L), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        use = min(len(s), L)
        arr[i, :use] = np.asarray(s[:use], dtype=float)
    return arr


def _metrics_dataframe_from_list(metrics_list: List[Dict]) -> pd.DataFrame:
    """metrics_list is your saved list of dicts (one per recorded epoch)."""
    if not metrics_list:
        return pd.DataFrame()
    return pd.DataFrame(metrics_list)


def _reduce_grad_snapshot_paramwise(d: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Reduce a single grad snapshot (param -> stats dict) into global scalars.
    Keeps robust, comparable summaries; extend if you need more.
    """
    if not d:
        return {}
    keys = ["l2_norm", "mean_sq", "max_abs"]  # core ones
    out = {f"grad_{k}_sum": 0.0 for k in keys}
    out.update({f"grad_{k}_mean": 0.0 for k in keys})
    out.update({f"grad_{k}_max": -np.inf for k in keys})
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
def collect_for_setting(hidden_init: str, in_type: str):
    """
    Returns:
      per_run_rows: list of per-run scalar metrics dicts
      per_run_timeseries: dict with keys 'losses', 'metrics_df_list', 'grad_df_list'
    """
    base = model_root / hidden_init / in_type
    per_run_rows = []
    losses_all = []
    metrics_df_list = []
    grad_df_list = []

    # single run (optional)
    single_path = _get_single_run_file(base)
    if single_path:
        ckpt = _safe_load(single_path)
        loss = _extract_loss_series(ckpt)
        m = _metrics_from_loss(loss)
        if m:
            per_run_rows.append(
                {
                    "hidden_init": hidden_init,
                    "input_type": in_type,
                    "run_kind": "single",
                    "run_id": None,
                    "path": str(single_path),
                    **m,
                }
            )
        if loss:
            losses_all.append(loss)

        # metrics timeseries
        mlist = _extract_metrics_list(ckpt)
        if mlist:
            metrics_df_list.append(_metrics_dataframe_from_list(mlist))
        # grad list (optional)
        glist = _extract_grad_list(ckpt)
        if glist and isinstance(glist[0], dict):
            # reduce each snapshot to scalars
            reduced = []
            for snap in glist:
                reduced.append(_reduce_grad_snapshot_paramwise(snap))
            grad_df_list.append(pd.DataFrame(reduced))

    # multiruns
    for run_id, p in _iter_multirun_files(base):
        ckpt = _safe_load(p)
        loss = _extract_loss_series(ckpt)
        m = _metrics_from_loss(loss)
        if m:
            per_run_rows.append(
                {
                    "hidden_init": hidden_init,
                    "input_type": in_type,
                    "run_kind": "multi",
                    "run_id": run_id,
                    "path": str(p),
                    **m,
                }
            )
        if loss:
            losses_all.append(loss)
        mlist = _extract_metrics_list(ckpt)
        if mlist:
            metrics_df_list.append(_metrics_dataframe_from_list(mlist))
        glist = _extract_grad_list(ckpt)
        if glist and isinstance(glist[0], dict):
            reduced = []
            for snap in glist:
                reduced.append(_reduce_grad_snapshot_paramwise(snap))
            grad_df_list.append(pd.DataFrame(reduced))

    return per_run_rows, {
        "losses": losses_all,
        "metrics_df_list": metrics_df_list,
        "grad_df_list": grad_df_list,
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
