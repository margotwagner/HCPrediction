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
WHH_NORM = "none"  # frobenius|spectral|variance|none  (baseline uses 'none' folder)
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


def _spectral_summary(
    W: np.ndarray, topk: int = 5, bulk_q: float = 0.95, eps: float = 1e-12
):
    """
    Minimal spectral summary for 'bulk + outliers':
      - radius_over_bulk: max|λ| / quantile_q(|λ|)
      - var_topk:         fraction of |λ|^2 mass in top-k eigenvalues
      - eff_dim:          participation ratio of |λ|^2 (effective dimension)
    """
    vals = np.linalg.eigvals(W)
    mags = np.abs(vals)
    if mags.size == 0:
        return dict(
            radius_over_bulk=np.nan,
            var_topk=np.nan,
            eff_dim=np.nan,
        )

    # radius vs bulk (95th percentile by default)
    radius = float(mags.max())
    bulk_r = float(np.quantile(mags, bulk_q))
    radius_over_bulk = float(radius / (bulk_r + eps))

    # power distribution over eigenvalues (use |λ|^2)
    power = mags**2
    total = float(power.sum())
    if total <= 0:
        return dict(
            radius_over_bulk=radius_over_bulk,
            var_topk=0.0,
            eff_dim=0.0,
        )

    p = power / total  # probability-like weights over modes

    # variance explained by top-k
    order = np.argsort(p)[::-1]
    var_topk = float(np.cumsum(p[order])[min(topk, len(p)) - 1])

    # participation ratio (effective dimension)
    eff_dim = float((p.sum() ** 2) / (np.sum(p**2) + eps))

    return dict(
        radius_over_bulk=radius_over_bulk,
        var_topk=var_topk,
        eff_dim=eff_dim,
    )


# ----------------------------
# Aggregation (one condition)
# ----------------------------


def aggregate_condition(root: Path):
    """
    Walk run_XX folders, read:
      - checkpoint loss curve
      - recorded metrics (to grab lambda if present)
      - hidden-weight snapshots and recompute offline metrics
      - gradient norms (total + per-group) aligned to recorded epochs

    Returns:
      per_run_df: one row per run with summary stats
      ts_df:      time-series rows aligned to recorded epochs (stacked over runs)
                 Columns now include:
                   - loss, lambda, [recomputed W metrics...]
                   - grad_norm, grad_S, grad_A, grad_input_linear, grad_linear, grad_lambda_raw (if present)
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

        # --- gradients (history + per_group) aligned by recorded epochs ---
        history = ckpt.get("history", {}) or {}
        hist_epochs = history.get("epoch", []) or []
        hist_grad = history.get("grad_norm", []) or []
        ep_to_grad = {int(e): _safe(g) for e, g in zip(hist_epochs, hist_grad)}

        grad_list = ckpt.get("grad_list", []) or []
        # map epoch -> per_group_norm dict, using history epoch order
        ep_to_group = {}
        for i, e in enumerate(hist_epochs):
            if i < len(grad_list):
                gentry = grad_list[i] or {}
                ep_to_group[int(e)] = gentry.get("per_group_norm", {}) or {}

        # --- recompute metrics from saved Wh snapshots ---
        snaps = _snapshot_list(run_dir)
        if not snaps:
            print(f"[warn] no snapshots under {run_dir / HW_SUBDIR}")
            continue

        for ep, Wp, Sp, Ap in snaps:
            W = torch.load(Wp, map_location="cpu").cpu().numpy()
            met = _metrics_from_W(W)
            lam = lam_map.get(ep, np.nan)
            spec = _spectral_summary(W)

            # gradient fields (total + expected groups, NaN if missing)
            g_total = ep_to_grad.get(ep, np.nan)
            g_groups = ep_to_group.get(ep, {})

            # normalize a few common group names
            g_S = _safe(g_groups.get("S", np.nan))
            g_A = _safe(g_groups.get("A", np.nan))
            g_in = _safe(g_groups.get("input_linear", np.nan))
            g_out = _safe(g_groups.get("linear", np.nan))
            g_lam = _safe(g_groups.get("lambda_raw", np.nan))  # only if trainable

            ts_rows.append(
                {
                    "run_id": run_id,
                    "epoch": ep,
                    "loss": _safe(loss[ep])
                    if (isinstance(ep, int) and ep < len(loss))
                    else np.nan,
                    "lambda": lam,
                    **met,
                    "grad_norm": g_total,
                    "grad_S": g_S,
                    "grad_A": g_A,
                    "grad_input_linear": g_in,
                    "grad_linear": g_out,
                    "grad_lambda_raw": g_lam,
                    **spec,
                }
            )

        # final snapshot metrics for the per-run summary
        last_ep, last_Wp, _, _ = snaps[-1]
        W_last = torch.load(last_Wp, map_location="cpu").cpu().numpy()
        met_last = _metrics_from_W(W_last)
        spec_last = _spectral_summary(W_last)

        rows.append(
            {
                "run_id": run_id,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "loss_auc": auc_loss,
                "last_epoch": int(last_ep),
                **{f"last_{k}": v for k, v in met_last.items()},
                **{f"last_{k}": v for k, v in spec_last.items()},
                "base_path": str(run_dir),
            }
        )

    per_run_df = pd.DataFrame(rows).sort_values("run_id")
    ts_df = pd.DataFrame(ts_rows).sort_values(["epoch", "run_id"])
    return per_run_df, ts_df


# ----------------------------
# Simple plots
# ----------------------------


def plot_loss_mean_band_condition(
    label: str,
    ts_df: pd.DataFrame,
    title: str = "Loss (mean ± sd)",
    fontsize: int = 12,
    figsize: Tuple[float, float] = (6.0, 4.0),
    fit: bool = True,
    fit_range: Optional[Tuple[int, int]] = None,  # e.g., (0, 10000)
    plot_fit: bool = True,
    logy: bool = True,
    eps: float = 1e-12,
):
    """
    Single-condition convenience wrapper around overlay_loss.
    Accepts the same knobs so your notebook calls feel consistent.
    Returns the same fit-stats DataFrame that overlay_loss returns, but with one row.
    """
    condition_series = {label: ts_df}
    return overlay_loss(
        condition_series=condition_series,
        title=title,
        fontsize=fontsize,
        figsize=figsize,
        fit=fit,
        fit_range=fit_range,
        plot_fit=plot_fit,
        logy=logy,
        eps=eps,
    )


def plot_metric_condition(
    label: str,
    ts_df: pd.DataFrame,
    key: str,
    title: Optional[str] = None,
    fontsize: int = 12,
    figsize: Tuple[float, float] = (6.0, 4.0),
    logy: bool = False,
    fit: bool = False,
    fit_on_log: bool = True,
    fit_range: Optional[Tuple[int, int]] = None,
    plot_fit: bool = True,
    eps: float = 1e-12,
):
    """
    Single-condition convenience wrapper around overlay_metric.
    Same interface as overlay_metric but for one (label, ts_df) pair.
    Returns the fit-stats DataFrame (one row).
    """
    condition_series = {label: ts_df}
    return overlay_metric(
        condition_series=condition_series,
        key=key,
        title=title,
        fontsize=fontsize,
        figsize=figsize,
        logy=logy,
        fit=fit,
        fit_on_log=fit_on_log,
        fit_range=fit_range,
        plot_fit=plot_fit,
        eps=eps,
    )


def plot_eigs_condition(
    root: Path,
    label: Optional[str] = None,
    **kwargs,
):
    """Single-condition convenience wrapper (no S/A fallback)."""
    if label is None:
        label = Path(root).name
    return overlay_eigs_snapshots({label: Path(root)}, **kwargs)


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

    Fit line color matches the corresponding mean curve and is dashed.
    Legend text is attached to the original (solid) line, not the fit.
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

        # plot mean (capture handle) and ± sd band
        (line,) = plt.plot(ep, mu_plot, label=label_with_init)
        c = line.get_color()
        plt.fill_between(ep, lo_plot, hi_plot, alpha=0.15, color=c)

        # ---- optional fit on ln(loss) ----
        slope = intercept = r2 = half_life = np.nan
        f_emin = ep.min() if fit_range is None else fit_range[0]
        f_emax = ep.max() if fit_range is None else fit_range[1]

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
                    (1.0 - (ss_res / ss_tot))
                    if (ss_tot not in (0.0, np.nan))
                    else np.nan
                )

                half_life = (np.log(2) / -slope) if slope < 0 else np.inf

                if plot_fit:
                    ep_line = np.linspace(ep_fit.min(), ep_fit.max(), 200)
                    mu_fit = np.exp(slope * ep_line + intercept)
                    mu_fit_plot = np.clip(mu_fit, eps, None) if logy else mu_fit
                    # match color, dashed; no legend entry
                    plt.plot(ep_line, mu_fit_plot, linestyle="--", alpha=0.9, color=c)

                # Attach slope to the ORIGINAL line's legend entry
                line.set_label(f"{label_with_init} (k≈{slope:.2e})")

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

    Fit line color matches the corresponding mean curve and is dashed.
    Legend text is attached to the original (solid) line, not the fit.

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

        # plot mean and capture handle/color
        (line,) = plt.plot(ep, mu_plot, label=label)
        c = line.get_color()
        plt.fill_between(ep, lo_plot, hi_plot, alpha=0.15, color=c)

        # ---- optional fit ----
        slope = intercept = r2 = half_life = np.nan
        f_emin = ep.min() if fit_range is None else fit_range[0]
        f_emax = ep.max() if fit_range is None else fit_range[1]

        if fit:
            mask = (ep >= f_emin) & (ep <= f_emax)
            ep_fit = ep[mask]
            y_raw = mu[mask]
            y_fit = np.log(np.clip(y_raw, eps, None)) if fit_on_log else y_raw

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
                    (1.0 - (ss_res / ss_tot))
                    if (ss_tot not in (0.0, np.nan))
                    else np.nan
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
                    # dashed fit with the SAME color; no legend entry
                    plt.plot(ep_line, y_line_plot, linestyle="--", alpha=0.9, color=c)

                kind = "log-fit" if fit_on_log else "lin-fit"
                # Attach slope/kind info to the ORIGINAL line's legend label
                line.set_label(f"{label} (k≈{slope:.2e}, {kind})")

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


def overlay_eigs_snapshots(
    condition_roots: Dict[str, Path],
    snapshot: str = "last",  # "first" | "middle" | "last"
    matrix: str = "W",  # "W" | "S" | "A"
    title: str = "Eigenvalues",
    fontsize: int = 12,
    figsize: Tuple[float, float] = (6.0, 6.0),
    max_points_per_run: int = 1000,
    unit_circle: bool = True,
    s: float = 6.0,  # marker size
    alpha: float = 0.5,  # point alpha
    seed: Optional[int] = 0,  # for reproducible downsampling
):
    """
    Overlay eigenvalues for one or more conditions, choosing which snapshot and which matrix.
    NO fallback: if matrix='S' or 'A' and the file for that snapshot is missing, that run is skipped.
    """
    rng = np.random.default_rng(seed)
    snapshot = snapshot.lower().strip()
    matrix = matrix.upper().strip()  # W/S/A

    if snapshot not in {"first", "middle", "last"}:
        raise ValueError("snapshot must be one of {'first','middle','last'}")
    if matrix not in {"W", "S", "A"}:
        raise ValueError("matrix must be one of {'W','S','A'}")

    def _pick_index(n: int) -> Optional[int]:
        if n <= 0:
            return None
        if snapshot == "first":
            return 0
        if snapshot == "last":
            return n - 1
        return n // 2  # middle

    plt.figure(figsize=figsize)
    any_points = False

    for label, root in condition_roots.items():
        runs = _sorted_runs(Path(root))
        if not runs:
            print(f"[warn] no runs under {root}")
            continue

        all_re, all_im = [], []

        for run_dir in runs:
            snaps = _snapshot_list(
                run_dir
            )  # list of (epoch, Wp, Sp_or_None, Ap_or_None)
            if not snaps:
                continue

            idx = _pick_index(len(snaps))
            if idx is None:
                continue

            ep, Wp, Sp, Ap = snaps[idx]

            # strict selection: require the exact file for S/A
            if matrix == "W":
                sel_path = Wp
            elif matrix == "S":
                if Sp is None or (not Sp.exists()):
                    print(
                        f"[warn] missing S file for run {run_dir.name} @ epoch {ep}; skipping this run."
                    )
                    continue
                sel_path = Sp
            else:  # "A"
                if Ap is None or (not Ap.exists()):
                    print(
                        f"[warn] missing A file for run {run_dir.name} @ epoch {ep}; skipping this run."
                    )
                    continue
                sel_path = Ap

            W = torch.load(sel_path, map_location="cpu").cpu().numpy()
            vals = _eigvals_np(W)

            # optional downsample per run
            if vals.size > max_points_per_run:
                idxs = rng.choice(vals.size, max_points_per_run, replace=False)
                vals = vals[idxs]

            all_re.append(np.real(vals))
            all_im.append(np.imag(vals))

        if all_re:
            X = np.concatenate(all_re)
            Y = np.concatenate(all_im)
            plt.scatter(X, Y, s=s, alpha=alpha, label=label)
            any_points = True

    if not any_points:
        plt.close()
        print(
            "[warn] nothing to plot (no snapshots found or required S/A files missing)"
        )
        return

    # axes & styling
    ax = plt.gca()
    ax.axhline(0, lw=0.5, c="k")
    ax.axvline(0, lw=0.5, c="k")
    if unit_circle:
        circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.3)
        ax.add_artist(circle)
    ax.set_aspect("equal", adjustable="box")

    # Legend outside
    plt.title(f"{title} — {matrix}@{snapshot}", fontsize=fontsize)
    plt.xlabel("Re(λ)", fontsize=fontsize)
    plt.ylabel("Im(λ)", fontsize=fontsize)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=max(6, fontsize - 1),
    )
    plt.tick_params(axis="both", which="major", labelsize=max(6, fontsize - 2))
    plt.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))  # leave room for outside legend


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
