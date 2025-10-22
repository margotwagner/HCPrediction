#!/usr/bin/env python3
"""
aggregate_metrics.py
Aggregate per-run metrics into a run-level CSV and per-condition summary CSV.

What it ingests (per run_XX/):
  - Offline heavy metrics (required for rich summaries):
      * <stub>_offline_metrics.csv    # created by offline_metrics.py
        (multi-row: snapshots; init is snapshot_idx=0, last is the max epoch)
  - Evaluation metrics (optional, as many as exist):
      * <stub>_replay.csv
      * <stub>_prediction.csv
      * <stub>_openloop.csv
      * <stub>_closedloop.csv
    (These are produced by evaluate.py; names are flexible—see glob flags.)

Outputs:
  - run_level.csv         # one row per run
  - condition_summary.csv # one row per "condition" (parent of run_XX), with mean/std

Design choices:
  - No dependence on torch; all CSV-based.
  - Condition = directory right above 'run_XX' (e.g., runs/ElmanRNN/random-init/random_n100).
  - "Last snapshot" = row with largest epoch in *_offline_metrics.csv.
  - If multiple eval CSVs found, we prefix columns by the mode (inferred from filename)
    and take the first (or only) row; you can expand easily later.

Usage:
1) Everything under a root directory:
  python aggregate_metrics.py \
      --root runs/ElmanRNN \
      --glob_offline "*_offline_metrics.csv" \
      --glob_eval "*_replay.csv,*_prediction.csv,*_openloop.csv,*_closedloop.csv" \
      --outdir runs/ElmanRNN
      
2) Single condition subtree
python aggregate_metrics.py \
  --root runs/ElmanRNN/shift-variants/shiftcyc/raw/sym0p90/shiftcyc_n100_raw      
"""

import argparse
import csv
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RUN_DIR_RE = re.compile(r"run_(\d+)$")


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and RUN_DIR_RE.search(os.path.basename(path)) is not None


def _find_run_dirs(root: str) -> List[str]:
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if RUN_DIR_RE.search(base):
            run_dirs.append(dirpath)
    run_dirs.sort()
    return run_dirs


def _parent_condition_dir(run_dir: str) -> str:
    return os.path.dirname(run_dir)


def _run_id(run_dir: str) -> str:
    m = RUN_DIR_RE.search(os.path.basename(run_dir))
    return m.group(1) if m else "NA"


def _pick_mode_from_filename(csv_path: str) -> str:
    # heuristics — you can edit these easily
    name = os.path.basename(csv_path).lower()
    for key in ("replay", "prediction", "openloop", "closedloop", "eval"):
        if key in name:
            return key
    return "eval"


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_offline_last_snapshot(offline_csv: str) -> Dict[str, float]:
    """
    Returns a flat dict summarizing init/last snapshot from *_offline_metrics.csv.
    We pull commonly useful columns and compute a few deltas.
    """
    df = _safe_read_csv(offline_csv)
    if df is None or df.empty:
        return {}

    # Identify init & last by snapshot_idx and epoch
    # Convention in offline_metrics.py: snapshot_idx=0 is init, subsequent are epochs
    # If epoch exists, we’ll use max epoch; otherwise max snapshot_idx.
    df = df.copy()
    if "epoch" in df.columns and df["epoch"].notna().any():
        # Some runs store init as epoch=0; guard against all NaN
        df["epoch_filled"] = df["epoch"].fillna(-1)
        last = df.loc[df["epoch_filled"].idxmax()]
    else:
        last = df.loc[df["snapshot_idx"].idxmax()]
    init = df.loc[df["snapshot_idx"].idxmin()]

    def g(row, col):  # get float if present
        return float(row[col]) if col in row and pd.notna(row[col]) else np.nan

    # Core fields from offline_metrics.py
    keys = [
        "fro_W",
        "fro_S",
        "fro_A",
        "mix_A_over_S",
        "sym_ratio",
        "asym_ratio",
        "non_normality_comm",
        "spectral_radius_W",
        "op_norm_2",
        "cond_number",
        "spectral_radius_S",
        "spectral_radius_A",
    ]
    out = {}
    for k in keys:
        out[f"init_{k}"] = g(init, k)
        out[f"last_{k}"] = g(last, k)
        out[f"delta_{k}"] = out[f"last_{k}"] - out[f"init_{k}"]

    # Keep record of epochs if present
    out["init_epoch"] = g(init, "epoch") if "epoch" in df.columns else np.nan
    out["last_epoch"] = g(last, "epoch") if "epoch" in df.columns else np.nan
    return out


def _collect_eval_csvs(
    run_dir: str, eval_globs: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Search for evaluation CSVs under run_dir using provided globs.
    Returns: {mode: {col: value, ...}}
    We take the FIRST row from each CSV (common case is single-row per run).
    Numeric cols only; we skip obviously non-metric columns.
    """
    metrics_by_mode: Dict[str, Dict[str, float]] = {}
    matched_files = set()

    for pat in eval_globs:
        for path in glob.glob(os.path.join(run_dir, "**", pat), recursive=True):
            if not path.lower().endswith(".csv"):
                continue
            if path in matched_files:
                continue
            # Skip offline_metrics.csv to avoid double counting
            if path.endswith("_offline_metrics.csv"):
                continue
            df = _safe_read_csv(path)
            if df is None or df.empty:
                continue

            mode = _pick_mode_from_filename(path)
            row = df.iloc[0]  # simple case: one-row CSV from evaluate.py

            # Keep numeric columns only (scalars); prefix them with the mode
            vals = {}
            for col in df.columns:
                val = row[col]
                if pd.api.types.is_number(val):
                    safe_col = f"{mode}_{col}"
                    vals[safe_col] = float(val)
            if vals:
                metrics_by_mode[mode] = {**metrics_by_mode.get(mode, {}), **vals}
                matched_files.add(path)

    return metrics_by_mode


def _find_offline_csv(run_dir: str, offline_glob: str) -> Optional[str]:
    # look for exactly one offline CSV; if multiple exist, pick the most recent by mtime
    cand = glob.glob(os.path.join(run_dir, "**", offline_glob), recursive=True)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _flatten(prefix: str, d: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _summarize_condition(
    df: pd.DataFrame, group_cols: List[str], agg_cols: List[str]
) -> pd.DataFrame:
    """
    Build condition-level summary with mean and std over runs.
    """
    grouped = df.groupby(group_cols, dropna=False)
    mean_df = grouped[agg_cols].mean().add_prefix("mean_")
    std_df = grouped[agg_cols].std(ddof=0).add_prefix("std_")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate run-level and condition-level metrics."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory to search (contains condition dirs and run_XX subdirs).",
    )
    ap.add_argument(
        "--glob_offline",
        default="*_offline_metrics.csv",
        help="Glob (relative) used to find offline metrics CSV within each run (default: *_offline_metrics.csv).",
    )
    ap.add_argument(
        "--glob_eval",
        default="*_replay.csv,*_prediction.csv,*_openloop.csv,*_closedloop.csv",
        help="Comma-separated globs (relative) for evaluation CSVs (default covers common cases).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to write the aggregated CSVs (default: <root>).",
    )
    args = ap.parse_args()

    outdir = args.outdir or args.root
    os.makedirs(outdir, exist_ok=True)

    eval_globs = [g.strip() for g in args.glob_eval.split(",") if g.strip()]

    run_dirs = _find_run_dirs(args.root)
    if not run_dirs:
        print(f"[warn] No run_XX directories found under {args.root}")
        return

    run_rows: List[Dict[str, float]] = []

    for run_dir in run_dirs:
        cond_dir = _parent_condition_dir(run_dir)
        rid = _run_id(run_dir)

        row: Dict[str, float] = {
            "condition_dir": cond_dir,
            "run_dir": run_dir,
            "run_id": rid,
        }

        # OFFLINE (required for rich summaries; if missing, we still proceed)
        offline_csv = _find_offline_csv(run_dir, args.glob_offline)
        if offline_csv is not None:
            off = _load_offline_last_snapshot(offline_csv)
            row.update(_flatten("", off))
            row["offline_csv"] = offline_csv
        else:
            row["offline_csv"] = ""

        # EVALUATION (optional)
        eval_dict = _collect_eval_csvs(run_dir, eval_globs=eval_globs)
        # merge all modes; columns are already prefixed
        for mode, kv in eval_dict.items():
            row.update(kv)

        run_rows.append(row)

    # -----------------------------
    # Write run-level CSV
    # -----------------------------
    run_df = pd.DataFrame(run_rows)
    run_csv = os.path.join(outdir, "run_level.csv")
    run_df.to_csv(run_csv, index=False)
    print(f"[ok] wrote {run_csv}  (rows={len(run_df)})")

    # -----------------------------
    # Write condition-level summary
    # -----------------------------
    # group by condition_dir; aggregate numeric columns
    num_cols = [
        c
        for c in run_df.columns
        if c not in ("condition_dir", "run_dir", "run_id", "offline_csv")
    ]
    # keep numeric only
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(run_df[c])]
    if num_cols:
        cond_df = _summarize_condition(
            run_df, group_cols=["condition_dir"], agg_cols=num_cols
        )
        cond_csv = os.path.join(outdir, "condition_summary.csv")
        cond_df.to_csv(cond_csv, index=False)
        print(f"[ok] wrote {cond_csv}  (rows={len(cond_df)})")
    else:
        print("[warn] No numeric columns to summarize at condition level.")


if __name__ == "__main__":
    main()
