# AnalyzeSymAsym.py
# Simple, personal-use aggregator & plots for SymAsymRNN (one condition; 10 runs)

from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict


# ----------------------------
# CONFIG: set these 3 and run
# ----------------------------
WHH_TYPE = "baseline"  # e.g. baseline|cycshift|shiftcycmh|identity|...
WHH_NORM = "none"  # frobenius|spectral|variance|none  (baseline uses 'none' folder per your script)
INPUT = "asym1"  # matches <INPUT> in Ns100_SeqN100_<INPUT>.pth.tar

ROOT = Path("SymAsymRNN") / "N100T100" / WHH_TYPE / WHH_NORM / INPUT / "multiruns"

# Filenames inside each run folder
CKPT_NAME = "Ns100_SeqN100_predloss_full.pth.tar"
METAF_NAME = "Ns100_SeqN100_predloss_full.meta.json"  # optional but nice to have
HW_SUBDIR = "hidden-weights"  # as saved by Main_s4.py

# ----------------------------
# Helpers
# ----------------------------


def _sorted_runs(root: Path):
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    return runs


_epoch_re = re.compile(r".*epoch(\d+)\.pt$")


def _snapshot_list(run_dir: Path):
    """Return sorted list of (epoch:int, Wh_path:Path, S_path:Path|None, A_path:Path|None)"""
    hw = run_dir / HW_SUBDIR
    if not hw.exists():
        return []
    W_files = sorted(hw.glob("Wh_epoch*.pt"))
    out = []
    for w in W_files:
        m = _epoch_re.match(str(w))
        if not m:
            continue
        ep = int(m.group(1))
        S = hw / f"S_epoch{ep:06d}.pt"
        A = hw / f"A_epoch{ep:06d}.pt"
        out.append((ep, w, S if S.exists() else None, A if A.exists() else None))
    return out


def _eigvals_np(W: np.ndarray):
    vals = np.linalg.eigvals(W)
    return vals


def _sv_smax_smin(W: np.ndarray):
    # robust SVD for spectral norm & condition number
    s = np.linalg.svd(W, compute_uv=False)
    smax = float(s.max()) if s.size else np.nan
    smin = float(s.min()) if s.size else np.nan
    return smax, smin


def _metrics_from_W(W: np.ndarray):
    H = W.shape[0]
    frob = float(np.linalg.norm(W, "fro"))
    # spectral radius
    vals = _eigvals_np(W)
    rho = float(np.max(np.abs(vals))) if vals.size else np.nan
    # spectral norm & cond number
    smax, smin = _sv_smax_smin(W)
    cond = (
        float(smax / (smin + 1e-12))
        if np.isfinite(smax) and smin is not None
        else np.nan
    )
    # orthogonality error ||W^T W - I||_F
    I = np.eye(H, dtype=W.dtype)
    orth_err = float(np.linalg.norm(W.T @ W - I, "fro"))
    # symmetry/antisymmetry Frobenius norms
    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)
    s_frob = float(np.linalg.norm(S, "fro"))
    a_frob = float(np.linalg.norm(A, "fro"))
    return dict(
        frob=frob,
        spectral_radius=rho,
        spectral_norm=smax,
        cond_num=cond,
        orth_err=orth_err,
        sym_frob=s_frob,
        asym_frob=a_frob,
    )


def _load_ckpt(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[warn] cannot load {path}: {e}")
        return None


def _safe(x, default=np.nan):
    try:
        return float(x)
    except:
        return default


# ----------------------------
# Aggregation (one condition)
# ----------------------------


def aggregate_condition(root: Path):
    """
    Walk run_XX folders, read:
      - checkpoint loss curve
      - recorded metrics (to grab lambda if present)
      - hidden-weight snapshots and recompute offline metrics
    Returns:
      per_run_df: one row per run with summary stats
      ts_df:      time-series rows aligned to recorded epochs (stacked over runs)
    """
    rows = []
    ts_rows = []

    runs = _sorted_runs(root)
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")

    print(f"[info] found {len(runs)} runs under {root}")

    for run_dir in runs:
        run_id = run_dir.name.replace("run_", "")
        ckpt = _load_ckpt(run_dir / CKPT_NAME)
        if ckpt is None:
            print(f"[warn] skipping {run_dir} (no checkpoint)")
            continue

        # --- loss curve summaries ---
        loss = ckpt.get("loss", [])
        final_loss = _safe(loss[-1]) if loss else np.nan
        best_loss = _safe(np.min(loss)) if len(loss) else np.nan
        auc_loss = _safe(np.trapz(loss)) if len(loss) else np.nan

        # --- lambda series (if present) ---
        metrics_list = ckpt.get("metrics", [])
        lam_series = [
            (m.get("epoch"), _safe(m.get("lambda")))
            for m in metrics_list
            if "lambda" in m
        ]
        lam_map = {int(ep): val for ep, val in lam_series if ep is not None}

        # --- recompute metrics from saved Wh snapshots ---
        snaps = _snapshot_list(run_dir)
        if not snaps:
            print(f"[warn] no snapshots under {run_dir / HW_SUBDIR}")
            continue

        for ep, Wp, Sp, Ap in snaps:
            W = torch.load(Wp, map_location="cpu").cpu().numpy()
            met = _metrics_from_W(W)
            lam = lam_map.get(ep, np.nan)
            ts_rows.append(
                {
                    "run_id": run_id,
                    "epoch": ep,
                    "loss": _safe(loss[ep])
                    if (isinstance(ep, int) and ep < len(loss))
                    else np.nan,
                    "lambda": lam,
                    **met,
                }
            )

        # final snapshot metrics for the per-run summary
        last_ep, last_Wp, _, _ = snaps[-1]
        W_last = torch.load(last_Wp, map_location="cpu").cpu().numpy()
        met_last = _metrics_from_W(W_last)

        rows.append(
            {
                "run_id": run_id,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "loss_auc": auc_loss,
                "last_epoch": int(last_ep),
                **{f"last_{k}": v for k, v in met_last.items()},
                "base_path": str(run_dir),
            }
        )

    per_run_df = pd.DataFrame(rows).sort_values("run_id")
    ts_df = pd.DataFrame(ts_rows).sort_values(["epoch", "run_id"])
    return per_run_df, ts_df


# ----------------------------
# Simple plots
# ----------------------------


def plot_loss_mean_band(ts_df: pd.DataFrame, title="Loss (mean ± std)"):
    g = ts_df.groupby("epoch")["loss"]
    ep = g.mean().index.values
    mu = g.mean().values
    sd = g.std().values
    plt.figure()
    plt.plot(ep, mu, label="mean")
    plt.fill_between(ep, mu - sd, mu + sd, alpha=0.2, label="±1 sd")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_metric(ts_df: pd.DataFrame, key: str, title=None):
    g = ts_df.groupby("epoch")[key]
    ep = g.mean().index.values
    mu = g.mean().values
    sd = g.std().values
    plt.figure()
    plt.plot(ep, mu, label="mean")
    plt.fill_between(ep, mu - sd, mu + sd, alpha=0.2, label="±1 sd")
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.title(title or key)
    plt.legend()
    plt.tight_layout()


def plot_eigs_last(root: Path, max_points_per_run=1000):
    """Scatter eigenvalues of last snapshot for each run (downsample if large)."""
    runs = _sorted_runs(root)
    plt.figure()
    for run_dir in runs:
        snaps = _snapshot_list(run_dir)
        if not snaps:
            continue
        _, Wp, _, _ = snaps[-1]
        W = torch.load(Wp, map_location="cpu").cpu().numpy()
        vals = _eigvals_np(W)
        # optional downsample to avoid overplotting
        if vals.size > max_points_per_run:
            idx = np.random.choice(vals.size, max_points_per_run, replace=False)
            vals = vals[idx]
        plt.scatter(np.real(vals), np.imag(vals), s=4, alpha=0.5)
    plt.axhline(0, lw=0.5, c="k")
    plt.axvline(0, lw=0.5, c="k")
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.3)
    plt.gca().add_artist(circle)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Eigenvalues (last snapshot, all runs)")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.tight_layout()


# ----------------------------
# Overlay helpers (multi-condition)
# ----------------------------
def overlay_loss(
    condition_series: Dict[str, pd.DataFrame],
    title: str = "Loss (mean ± sd)",
    fontsize: int = 12,
    figsize: Tuple[float, float] = (6.0, 4.0),
    fit: bool = True,
    fit_range: Optional[Tuple[int, int]] = None,  # e.g., (0, 10000); None = all
    plot_fit: bool = True,
    logy: bool = True,  # plot on log y-axis if True
    eps: float = 1e-12,  # positivity floor for log
):
    """
    Overlay loss curves for multiple conditions with adjustable font size and figure size.
    Optionally fit ln(loss) = slope*epoch + intercept (exponential decay) and plot the fit.

    Returns a DataFrame with columns:
      ["condition","init_epoch","init_loss","slope","intercept","r2","half_life_epochs","fit_emin","fit_emax"]
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    results = []
    any_series = False

    for label, ts in condition_series.items():
        g = ts.groupby("epoch")["loss"]
        ep = g.mean().index.values
        if ep.size == 0:
            continue

        mu = g.mean().values
        sd = g.std().values

        # initial mean loss at earliest epoch present
        first_ep = int(ep.min())
        init_mean = float(ts.loc[ts["epoch"] == first_ep, "loss"].mean())
        label_with_init = f"{label} (L0≈{init_mean:.3g})"

        # plotting arrays
        if logy:
            mu_plot = np.clip(mu, eps, None)
            lo_plot = np.clip(mu - sd, eps, None)
            hi_plot = np.clip(mu + sd, eps, None)
        else:
            mu_plot = mu
            lo_plot = mu - sd
            hi_plot = mu + sd

        # plot mean ± sd band
        plt.plot(ep, mu_plot, label=label_with_init)
        plt.fill_between(ep, lo_plot, hi_plot, alpha=0.15)

        # ---- optional fit on ln(loss) ----
        slope = intercept = r2 = half_life = np.nan
        f_emin = ep.min() if fit_range is None else fit_range[0]
        f_emax = (
            ep.max() if fit_range is None else fit_range[1] if len(ep) else ep.max()
        )

        if fit:
            mask = (ep >= f_emin) & (ep <= f_emax)
            ep_fit = ep[mask]
            y_fit = np.log(np.clip(mu[mask], eps, None))
            if ep_fit.size >= 2 and np.all(np.isfinite(y_fit)):
                # linear least squares on log-loss
                slope, intercept = np.polyfit(ep_fit, y_fit, 1)

                y_pred = slope * ep_fit + intercept
                ss_res = float(np.sum((y_fit - y_pred) ** 2))
                ss_tot = (
                    float(np.sum((y_fit - y_fit.mean()) ** 2))
                    if ep_fit.size > 1
                    else np.nan
                )
                r2 = (
                    1.0 - (ss_res / ss_tot) if (ss_tot not in (0.0, np.nan)) else np.nan
                )

                half_life = (np.log(2) / -slope) if slope < 0 else np.inf

                if plot_fit:
                    ep_line = np.linspace(ep_fit.min(), ep_fit.max(), 200)
                    mu_fit = np.exp(slope * ep_line + intercept)
                    mu_fit_plot = np.clip(mu_fit, eps, None) if logy else mu_fit
                    plt.plot(ep_line, mu_fit_plot, linestyle="--", alpha=0.8)

                # append slope to last line's legend entry
                plt.gca().lines[-1].set_label(f"{label_with_init} (k≈{slope:.2e})")

        results.append(
            {
                "condition": label,
                "init_epoch": first_ep,
                "init_loss": init_mean,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "half_life_epochs": half_life,
                "fit_emin": f_emin,
                "fit_emax": f_emax,
            }
        )
        any_series = True

    if not any_series:
        plt.close()
        return pd.DataFrame(
            columns=[
                "condition",
                "init_epoch",
                "init_loss",
                "slope",
                "intercept",
                "r2",
                "half_life_epochs",
                "fit_emin",
                "fit_emax",
            ]
        )

    # styling + axes
    if logy:
        plt.yscale("log")
        plt.ylabel("loss (log)", fontsize=fontsize)
        if "log scale" not in title.lower():
            title = f"{title} (log scale)"
    else:
        plt.ylabel("loss", fontsize=fontsize)

    plt.xlabel("epoch", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=max(6, fontsize - 1))
    plt.tick_params(axis="both", which="major", labelsize=max(6, fontsize - 2))
    plt.tight_layout()

    return pd.DataFrame(results)


def overlay_metric(
    condition_series: Dict[str, pd.DataFrame],
    key: str,
    title: Optional[str] = None,
    fontsize: int = 12,
    figsize: Tuple[float, float] = (6.0, 4.0),
    logy: bool = False,  # optional log plotting
    fit: bool = False,  # optional curve fitting
    fit_on_log: bool = True,  # if True, fit ln(mean(metric)) vs epoch
    fit_range: Optional[Tuple[int, int]] = None,  # (emin, emax)
    plot_fit: bool = True,
    eps: float = 1e-12,
):
    """
    Overlay any metric (mean ± sd across runs) with adjustable fonts/size,
    optional log plotting, and optional curve fitting.

    Fitting options:
      - if fit and fit_on_log:   ln(mean(metric)) ~ slope * epoch + intercept  (exp-like trends)
      - if fit and not fit_on_log: mean(metric) ~ slope * epoch + intercept   (linear trend)

    Returns a DataFrame with columns:
      ["condition","slope","intercept","r2","half_life_epochs","fit_emin","fit_emax"]
      (half_life_epochs only meaningful for exp fits with negative slope)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if title is None:
        title = key

    plt.figure(figsize=figsize)
    results = []
    any_series = False

    for label, ts in condition_series.items():
        if key not in ts.columns:
            continue

        g = ts.groupby("epoch")[key]
        ep = g.mean().index.values
        if ep.size == 0:
            continue

        mu = g.mean().values
        sd = g.std().values

        # plotting arrays
        if logy:
            mu_plot = np.clip(mu, eps, None)
            lo_plot = np.clip(mu - sd, eps, None)
            hi_plot = np.clip(mu + sd, eps, None)
        else:
            mu_plot = mu
            lo_plot = mu - sd
            hi_plot = mu + sd

        plt.plot(ep, mu_plot, label=label)
        plt.fill_between(ep, lo_plot, hi_plot, alpha=0.15)

        # ---- optional fit ----
        slope = intercept = r2 = half_life = np.nan
        f_emin = ep.min() if fit_range is None else fit_range[0]
        f_emax = (
            ep.max() if fit_range is None else fit_range[1] if len(ep) else ep.max()
        )

        if fit:
            mask = (ep >= f_emin) & (ep <= f_emax)
            ep_fit = ep[mask]
            y_raw = mu[mask]
            if fit_on_log:
                y_fit = np.log(np.clip(y_raw, eps, None))
            else:
                y_fit = y_raw

            if ep_fit.size >= 2 and np.all(np.isfinite(y_fit)):
                slope, intercept = np.polyfit(ep_fit, y_fit, 1)

                y_pred = slope * ep_fit + intercept
                ss_res = float(np.sum((y_fit - y_pred) ** 2))
                ss_tot = (
                    float(np.sum((y_fit - y_fit.mean()) ** 2))
                    if ep_fit.size > 1
                    else np.nan
                )
                r2 = (
                    1.0 - (ss_res / ss_tot) if (ss_tot not in (0.0, np.nan)) else np.nan
                )

                if fit_on_log and slope < 0:
                    half_life = np.log(2) / -slope
                else:
                    half_life = np.nan

                if plot_fit:
                    ep_line = np.linspace(ep_fit.min(), ep_fit.max(), 200)
                    if fit_on_log:
                        y_line = np.exp(slope * ep_line + intercept)
                    else:
                        y_line = slope * ep_line + intercept
                    y_line_plot = np.clip(y_line, eps, None) if logy else y_line
                    plt.plot(ep_line, y_line_plot, linestyle="--", alpha=0.8)

                # annotate slope on the most recent line's label
                kind = "log-fit" if fit_on_log else "lin-fit"
                plt.gca().lines[-1].set_label(f"{label} (k≈{slope:.2e}, {kind})")

        results.append(
            {
                "condition": label,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "half_life_epochs": half_life,
                "fit_emin": f_emin,
                "fit_emax": f_emax,
            }
        )
        any_series = True

    if not any_series:
        plt.close()
        return pd.DataFrame(
            columns=[
                "condition",
                "slope",
                "intercept",
                "r2",
                "half_life_epochs",
                "fit_emin",
                "fit_emax",
            ]
        )

    # styling + axes
    if logy:
        plt.yscale("log")
        plt.ylabel(f"{key} (log)", fontsize=fontsize)
        if "log scale" not in title.lower():
            title = f"{title} (log scale)"
    else:
        plt.ylabel(key, fontsize=fontsize)

    plt.xlabel("epoch", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=max(6, fontsize - 1))
    plt.tick_params(axis="both", which="major", labelsize=max(6, fontsize - 2))
    plt.tight_layout()

    return pd.DataFrame(results)


def overlay_quick(condition_series: dict):
    overlay_loss(condition_series)
    for key in [
        "spectral_radius",
        "spectral_norm",
        "cond_num",
        "orth_err",
        "sym_frob",
        "asym_frob",
        "lambda",
    ]:
        if any(key in ts.columns for ts in condition_series.values()):
            overlay_metric(condition_series, key, title=key)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    print(f"[info] loading runs from: {ROOT}")
    per_run_df, ts_df = aggregate_condition(ROOT)

    out_dir = ROOT.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = out_dir / "per_run_metrics.csv"
    ts_csv = out_dir / "timeseries_metrics.csv"

    per_run_df.to_csv(per_run_csv, index=False)
    ts_df.to_csv(ts_csv, index=False)
    print(f"[save] {per_run_csv}")
    print(f"[save] {ts_csv}")

    # Quick default plots
    """
    plot_loss_mean_band(ts_df, title=f"Loss — {WHH_TYPE}/{WHH_NORM}/{INPUT}")
    plot_metric(ts_df, "spectral_radius", title="Spectral radius")
    plot_metric(ts_df, "spectral_norm", title="Spectral norm")
    plot_metric(ts_df, "cond_num", title="Condition number")
    plot_metric(ts_df, "orth_err", title="Orthogonality error")
    plot_metric(ts_df, "sym_frob", title="||Sym||_F")
    plot_metric(ts_df, "asym_frob", title="||Asym||_F")
    if "lambda" in ts_df.columns and ts_df["lambda"].notna().any():
        plot_metric(ts_df, "lambda", title="λ (SymAsymRNN blend)")

    plot_eigs_last(ROOT)

    plt.show()
    """
