# ============================================================
# Multirun analysis: combine and save metrics
# ============================================================

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import pandas as pd
import pickle
import json, re, os

# -------------------
# CONFIG
# -------------------
DATA_DIR = Path("./data/Ns100_SeqN100/")
MODEL_ROOT = Path("./SymAsymRNN/N100T100/")
CSV_ROOT = MODEL_ROOT / "csv"
CSV_ROOT.mkdir(parents=True, exist_ok=True)

# whh_type (hidden weight type)
HIDDEN_WEIGHT_INITS = ["baseline", "cycshift", "shiftcycmh"]  # add more as needed

# whh_norm (norms used in training)
INPUT_TYPES = ["none", "frobenius", "spectral", "variance"]

MULTIRUNS_DIR = "multiruns"
RUN_PREFIX = "run_"
MODEL_FNAME = "Ns100_SeqN100_predloss_full.pth.tar"
HIDDEN_WEIGHTS_SUBDIR = "hidden-weights"


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


import json, re, os


def _infer_input_type_from_meta(ckpt_path: Path) -> str:
    """Parses meta['args']['input'] to get the <INPUT> suffix."""
    meta_path = ckpt_path.with_suffix("").with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text())
        inp = meta.get("args", {}).get("input", "") or ""
        m = re.search(r"Ns100_SeqN100_(.+?)\.pth\.tar$", inp)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def _iter_run_dirs_for_type(whh_type: str):
    """
    Yields (whh_norm, run_dir, ckpt_path, hw_dir) across baseline and norm subfolders.
    Baseline has no norm directory; non-baseline have /<norm>.
    """
    base = MODEL_ROOT / whh_type
    candidates = [base] if whh_type == "baseline" else [base / n for n in WHH_NORMS]
    for root in candidates:
        if not root.exists():
            continue
        multiruns = root / MULTIRUNS_DIR
        if not multiruns.exists():
            continue
        for d in sorted(multiruns.glob(f"{RUN_PREFIX}*")):
            ckpt = d / MODEL_FNAME
            if not ckpt.exists():
                continue
            hw_dir = HIDDEN_WEIGHTS_SUBDIR
            whh_norm = "none" if whh_type == "baseline" else root.name
            yield whh_norm, d, ckpt, hw_dir


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


def _extract_grad_list(ckpt):
    """Return a list of gradient snapshots or an empty list."""
    gl = ckpt.get("grad_list", None)
    if gl is None:
        return []
    # Some runs may have been saved as tuple or np.obj arrays; normalize
    if isinstance(gl, (list, tuple)):
        return list(gl)
    try:
        return list(gl)
    except Exception:
        return []


def _extract_history(ckpt, keep=("epoch", "loss", "grad_norm"), summarize=True):
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


def _iter_multirun_files_limited(base_dir: Path, max_runs: int = 10):
    """Yield (run_id, path) pairs for each run file found under multiruns/run_XX, up to max_runs."""
    multiruns_dir = base_dir / MULTIRUNS_DIR
    if not multiruns_dir.exists():
        print(f"[WARN] Multirun dir does not exist: {multiruns_dir}")
        return
    count = 0
    for run_dir in sorted(multiruns_dir.glob(f"{RUN_PREFIX}*")):
        if count >= max_runs:
            break
        path = run_dir / MODEL_FNAME
        if path.exists():
            run_id = run_dir.name.replace(
                RUN_PREFIX, "", 1
            )  # Extract run ID ('00' from 'run_00')
            yield run_id, path
            count += 1


def _attach_epoch_to_list(rows, ckpt):
    """
    rows: list[dict] reduced gradient rows
    ckpt: checkpoint dict (for recorded_epochs/recordep/history fallback)
    """
    import pandas as pd

    if not rows:
        return None
    df = pd.DataFrame(rows)

    # Preferred: explicit recorded epochs
    rec_epochs = ckpt.get("recorded_epochs", None)
    if isinstance(rec_epochs, list) and len(rec_epochs) > 0:
        n = min(len(rows), len(rec_epochs))
        if n < len(rows) or n < len(rec_epochs):
            print(
                f"[WARN] grad rows ({len(rows)}) vs recorded_epochs ({len(rec_epochs)}) mismatch; truncating to {n}."
            )
        df = df.iloc[:n, :].copy()
        df["epoch"] = [int(e) for e in rec_epochs[:n]]
        return df

    # Fallback: use history['epoch'] if available
    hist_epochs = ckpt.get("history", {}).get("epoch", None)
    if isinstance(hist_epochs, list) and len(hist_epochs) > 0:
        n = min(len(rows), len(hist_epochs))
        if n < len(rows) or n < len(hist_epochs):
            print(
                f"[WARN] grad rows ({len(rows)}) vs history epochs ({len(hist_epochs)}) mismatch; truncating to {n}."
            )
        df = df.iloc[:n, :].copy()
        df["epoch"] = [int(e) for e in hist_epochs[:n]]
        return df

    # Last fallback: synthesize evenly spaced epochs using recordep (default 1)
    rec_ep = int(ckpt.get("recordep", 1))
    df["epoch"] = [i * rec_ep for i in range(len(df))]
    print(
        "[INFO] No explicit epoch indices for grads; synthesized using recordep =",
        rec_ep,
    )
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


def _reduce_grad_snapshot_paramwise(snap):
    """
    Expected 'snap' schema:
      {
        "per_param": {"<name>": {"l2_norm":..., "mean":..., "std":..., "cos_prev":...}, ...},
        "per_group_norm": {"rnn": float, "linear": float, ...}
      }
    """
    import numpy as np

    per_param = (snap or {}).get("per_param", {}) or {}
    per_group = (snap or {}).get("per_group_norm", {}) or {}

    l2s = [p.get("l2_norm", np.nan) for p in per_param.values()]
    means = [p.get("mean", np.nan) for p in per_param.values()]
    stds = [p.get("std", np.nan) for p in per_param.values()]
    cos = [p.get("cos_prev", np.nan) for p in per_param.values()]

    out = {
        "grad_l2_sum": float(np.nansum(l2s)) if len(l2s) else np.nan,
        "grad_l2_mean": float(np.nanmean(l2s)) if len(l2s) else np.nan,
        "grad_mean_mean": float(np.nanmean(means)) if len(means) else np.nan,
        "grad_std_mean": float(np.nanmean(stds)) if len(stds) else np.nan,
        "grad_cos_prev_mean": float(np.nanmean(cos)) if len(cos) else np.nan,
    }
    for g, v in per_group.items():
        out[f"group_{g}_l2"] = float(v)
    return out


# -------------------
# Saving helpers
# -------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_df_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# -------------------
# Collection for one (hidden_init, input_type)
# -------------------
def collect_for_setting(hidden_init: str, _unused_input_type: str = None):
    """
    Collect runs for a hidden_init (whh_type) across all its norm folders.
    Input type is inferred per-run from .meta.json.
    """
    rows = []
    ts = {
        "loss_series": [],
        "metrics_df_list": [],
        "grad_df_list": [],
        "run_ids": [],
        "whh_norms": [],
        "input_types": [],
        "hw_dirs": [],
    }

    for whh_norm, run_dir, ckpt_path, hw_dir in _iter_run_dirs_for_type(hidden_init):
        run_id = run_dir.name.replace(RUN_PREFIX, "", 1)
        ckpt = _load_torch(ckpt_path)
        if ckpt is None:
            continue

        input_type = _infer_input_type_from_meta(ckpt_path)

        # --- extract what you already had ---
        loss_series = _extract_loss_series(ckpt)
        m_basic = _metrics_from_loss(loss_series)

        mlist = _extract_metrics_list(ckpt)
        metrics_df = _metrics_df_from_list(mlist, run_id) if mlist else None
        metrics_summary = _summarize_metrics(metrics_df, loss_series)

        grad_rows = _extract_grad_list(ckpt)
        grad_df = _reduce_grad_snapshot_paramwise(grad_rows)
        grad_df = _attach_epoch_to_list(grad_rows, grad_df, ckpt)

        rows.append(
            {
                "hidden_init": hidden_init,
                "whh_norm": whh_norm,
                "input_type": input_type,
                "run_kind": "multirun",
                "run_id": run_id,
                "path": str(ckpt_path),
                **(m_basic or {}),
                **(metrics_summary or {}),
            }
        )

        if loss_series:
            ts["loss_series"].append(loss_series)
        if metrics_df is not None:
            ts["metrics_df_list"].append(metrics_df)
        if grad_df is not None:
            ts["grad_df_list"].append(grad_df)
        ts["run_ids"].append(run_id)
        ts["whh_norms"].append(whh_norm)
        ts["input_types"].append(input_type)
        ts["hw_dirs"].append(str(hw_dir) if hw_dir else "")

    return rows, ts


# -------------------
# High-level collection across all settings
# -------------------
def collect_all(h_inits=None):
    h_inits = h_inits or HIDDEN_WEIGHT_INITS
    all_rows, ts_bucket = [], {}

    for h in h_inits:
        rows, ts_all = collect_for_setting(h)
        all_rows.extend(rows)

        # split ts by input_type to preserve (hidden_init, input_type) API
        if ts_all.get("input_types"):
            for it in sorted(set(ts_all["input_types"])):
                mask = [t == it for t in ts_all["input_types"]]
                ts_bucket[(h, it)] = {
                    k: [v for v, m in zip(ts_all[k], mask) if m]
                    for k in [
                        "loss_series",
                        "metrics_df_list",
                        "grad_df_list",
                        "run_ids",
                        "whh_norms",
                        "input_types",
                        "hw_dirs",
                    ]
                }
        else:
            ts_bucket[(h, "unknown")] = ts_all

    per_run_df = pd.DataFrame(all_rows)
    agg_df = _aggregate_per_run(per_run_df) if not per_run_df.empty else pd.DataFrame()

    if not per_run_df.empty:
        _save_df_csv(per_run_df, CSV_ROOT / "per_run_metrics.csv")
    if not agg_df.empty:
        _save_df_csv(agg_df, CSV_ROOT / "agg_metrics.csv")
    with open(CSV_ROOT / "ts_bucket.pkl", "wb") as f:
        pickle.dump(ts_bucket, f)

    return per_run_df, agg_df, ts_bucket


# -------------------
# MAIN
# -------------------
def main():
    print("Starting multirun aggregation...")
    per_run_df, agg_df, ts_bucket = collect_all()

    # Save global aggregate CSVs
    print("Saving global aggregate CSVs...")
    _ensure_dir(CSV_ROOT)
    if not per_run_df.empty:
        _save_df_csv(per_run_df, CSV_ROOT / "per_run_metrics.csv")
        print(f"[saved] {CSV_ROOT / 'per_run_metrics.csv'}")
    if not agg_df.empty:
        _save_df_csv(agg_df, CSV_ROOT / "agg_metrics.csv")
        print(f"[saved] {CSV_ROOT / 'agg_metrics.csv'}")
    with open(CSV_ROOT / "ts_bucket.pkl", "wb") as f:
        pickle.dump(ts_bucket, f)
        print(f"[saved] {CSV_ROOT / 'ts_bucket.pkl'}")

    # Save per setting to the correct multirun path
    # ts_bucket keys are (hidden_init, input_type)
    print("Saving per-setting CSVs and timeseries...")
    settings = list(ts_bucket.keys())
    for h, it in settings:
        combo_dir = MODEL_ROOT / h / it / MULTIRUNS_DIR / "aggregate_exports"
        _ensure_dir(combo_dir)

        # Filter per_run_df & agg_df for this setting
        per_run_combo = per_run_df[
            (per_run_df["hidden_init"] == h) & (per_run_df["input_type"] == it)
        ]
        agg_combo = agg_df[(agg_df["hidden_init"] == h) & (agg_df["input_type"] == it)]

        # Save CSVs
        if not per_run_combo.empty:
            _save_df_csv(per_run_combo, combo_dir / "per_run_metrics.csv")
            print(f"[saved] {combo_dir / 'per_run_metrics.csv'}")
        if not agg_combo.empty:
            _save_df_csv(agg_combo, combo_dir / "agg_metrics.csv")
            print(f"[saved] {combo_dir / 'agg_metrics.csv'}")

        # Save timeseries bucket
        with open(combo_dir / "ts_bucket.pkl", "wb") as f:
            pickle.dump(ts_bucket[(h, it)], f)
            print(f"[saved] {combo_dir / 'ts_bucket.pkl'}")

        print(f"Finished saving data for setting (hidden_init={h}, input_type={it})")

    print("[main] Done.")


if __name__ == "__main__":
    main()


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
