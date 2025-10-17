# evaluate.py
# ------------------------------------------------------------
# Minimal evaluation for ElmanRNN checkpoints (CSV only).
# Modes:
#   - open:       teacher-forced evaluation on (saved or held-out) data
#   - replay:     drive the trained model with noise input (no teacher forcing)
#   - prediction: teacher-forced prefix, then continue with constructed inputs
#   - closed:     teacher-forced prefix, then fully autonomous (output -> next input)
#
# Output:
#   Appends one row of metrics per run to a CSV you specify (or a default).
#
# Assumptions:
#   - Checkpoint contains: state_dict, args, X_mini, Target_mini
#   - Tensors are [batch, time, features]
#   - Input and output dims both N (ring task)
#   - RNN_Class.ElmanRNN_pytorch_module_v2 returns (output_seq, hidden_seq)
#
# Author: you + a helpful robot friend :)
# ------------------------------------------------------------

import argparse
from pathlib import Path
import math
import csv
import numpy as np
import torch
import torch.nn as nn

from RNN_Class import ElmanRNN_pytorch_module_v2  # your model class


# ------------------------- Rebuild helpers -------------------------


def _set_output_activation(net: nn.Module, ac_output: str):
    """Match training-time output activation if it was overridden."""
    if ac_output == "tanh":
        net.act = nn.Tanh()
    elif ac_output == "relu":
        net.act = nn.ReLU()
    elif ac_output == "sigmoid":
        net.act = nn.Sigmoid()


def _maybe_use_relu(net: nn.Module, use_relu: bool, N: int, H: int):
    """Match training-time RNN nonlinearity toggle (--rnn_act relu)."""
    if use_relu:
        net.rnn = nn.RNN(N, H, 1, batch_first=True, nonlinearity="relu")


def _load_ckpt(ckpt_path: Path, map_location="cpu"):
    """Load a training checkpoint saved by Main_clean.py."""
    return torch.load(ckpt_path, map_location=map_location)


def _rebuild_model_from_args(saved_args: dict, device: str):
    """Reconstruct the ElmanRNN with the same dims/activations used during training."""
    N = int(saved_args.get("n"))
    H = int(saved_args.get("hidden_n"))
    net = ElmanRNN_pytorch_module_v2(N, H, N).to(device)
    rnn_act = saved_args.get("rnn_act", "")
    _maybe_use_relu(net, rnn_act == "relu", N, H)
    ac_output = saved_args.get("ac_output", "")
    if ac_output:
        _set_output_activation(net, ac_output)
    return net, N, H


def _forward_sequence(state_dict, X: torch.Tensor, saved_args: dict, device: str):
    """Teacher-forced forward pass on a whole sequence X (returns CPU tensor)."""
    net, N, H = _rebuild_model_from_args(saved_args, device)
    net.load_state_dict(state_dict)
    net.eval()
    with torch.no_grad():
        h0 = torch.zeros(1, X.shape[0], H, device=device)
        Y_out, _ = net(X.to(device), h0)
    return Y_out.cpu()


# ------------------------- Angle / residual utils -------------------------


def _normalize_distribution(x: torch.Tensor, dim=-1, eps=1e-12):
    """Clamp>=0 and L1-normalize along dim to get a valid probability vector."""
    x = x.clamp_min(0)
    s = x.sum(dim=dim, keepdim=True)
    return x / (s + eps)


def _targets_to_angles(Target: torch.Tensor):
    """Convert ring distributions to angles θ ∈ (-pi, pi]."""
    B, T, N = Target.shape
    idx = torch.arange(N, device=Target.device)
    theta = 2 * math.pi * idx / N
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)
    C = (Target * cos_th).sum(dim=2)
    S = (Target * sin_th).sum(dim=2)
    return torch.atan2(S, C)


def _angles_from_distribution(P: torch.Tensor):
    """Same as above, but first normalize arbitrary outputs to a distribution."""
    Pn = _normalize_distribution(P, dim=2)
    return _targets_to_angles(Pn)


def _wrap_circular(delta: torch.Tensor):
    """Wrap angular differences to (-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _angle_error_concentration(output: torch.Tensor, Target: torch.Tensor):
    """Circular concentration R of Δθ = θ_pred - θ_true (higher=better)."""
    theta_true = _targets_to_angles(_normalize_distribution(Target, dim=2))
    theta_pred = _angles_from_distribution(output)
    dtheta = _wrap_circular(theta_pred - theta_true)  # [B,T]
    c = torch.cos(dtheta).mean()
    s = torch.sin(dtheta).mean()
    R = torch.sqrt(c * c + s * s).item()
    return {"angle_error_R": float(R), "angle_error_circ_var": float(1.0 - R)}


def _residual_stats(output: torch.Tensor, Target: torch.Tensor):
    """Residual magnitude mean (L2) and lag-1 autocorrelation over time."""
    res = Target - output
    rnorm = _vector_norm_compat(res, dim=2)  # [B,T]
    mean_L2 = float(rnorm.mean().item())
    if rnorm.shape[1] < 3:
        return {"residual_lag1_autocorr": None, "residual_L2_mean": mean_L2}
    x = rnorm - rnorm.mean(dim=1, keepdim=True)
    num = (x[:, :-1] * x[:, 1:]).sum(dim=1)
    den = torch.sqrt((x[:, :-1] ** 2).sum(dim=1) * (x[:, 1:] ** 2).sum(dim=1)) + 1e-12
    lag1 = float((num / den).mean().item())
    return {"residual_lag1_autocorr": lag1, "residual_L2_mean": mean_L2}


def _ring_decode_r2_from_outputs(
    output: torch.Tensor, Target: torch.Tensor, ridge=1e-4
):
    """Ridge decode of [cosθ, sinθ] from outputs; report mean R^2."""
    B, T, N = output.shape
    out = output.reshape(B * T, N)
    X = _normalize_distribution(out, dim=1)
    theta = _targets_to_angles(_normalize_distribution(Target, dim=2)).reshape(B * T)
    Y = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [BT,2]
    XtX = X.T @ X
    I = torch.eye(N, device=X.device, dtype=X.dtype)
    W = W = torch.inverse(XtX + ridge * I) @ (X.T @ Y)
    Yhat = X @ W
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    ss_res = ((Y - Yhat) ** 2).sum(dim=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return {"ring_decode_R2": float(r2.mean().item())}


def _circular_error_series(output: torch.Tensor, Target: torch.Tensor):
    """|Δθ| time-series in radians (batch handled by later mean)."""
    theta_true = _targets_to_angles(_normalize_distribution(Target, dim=2))
    theta_pred = _angles_from_distribution(output)
    return _wrap_circular(theta_pred - theta_true).abs()  # [B,T]


def _time_to_divergence(output, Target, thresh_rad=math.pi / 6, consec=10):
    """First t where mean(|Δθ|) ≥ threshold for 'consec' consecutive steps."""
    dtheta = _circular_error_series(output, Target).mean(dim=0).numpy()
    if dtheta.shape[0] < consec:
        return None
    mask = (dtheta >= thresh_rad).astype(np.float64)
    win = np.convolve(mask, np.ones(consec, dtype=float), mode="valid")
    idx = np.where(win >= consec)[0]
    return int(idx[0]) if len(idx) > 0 else None


def _phase_drift_per_step(output, Target):
    """Slope (rad/step) of unwrapped signed Δθ over time (mean across batch)."""
    dtheta = _wrap_circular(
        _angles_from_distribution(output)
        - _targets_to_angles(_normalize_distribution(Target, dim=2))
    )
    dmean = dtheta.mean(dim=0).numpy()
    unwrapped = np.unwrap(dmean)
    t = np.arange(len(unwrapped))
    if len(t) < 2:
        return None
    b = np.polyfit(t, unwrapped, 1)[0]
    return float(b)


# ------------------------- Basic summaries -------------------------


def _metric_mse(Y_true, Y_pred):
    return float(torch.mean((Y_true - Y_pred) ** 2).item())


def _metric_corr_per_feat(Y_true, Y_pred):
    """Mean Pearson r across features, over (batch * time)."""
    yt = Y_true.reshape(-1, Y_true.shape[-1]).numpy()
    yp = Y_pred.reshape(-1, Y_pred.shape[-1]).numpy()
    yt = yt - yt.mean(axis=0, keepdims=True)
    yp = yp - yp.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.linalg.norm(yt, axis=0) * np.linalg.norm(yp, axis=0) + 1e-12
    r = num / den
    return float(np.nanmean(r))


# ------------------------- Closed-loop core -------------------------


def _forward_closed_loop(
    state_dict,
    saved_args: dict,
    X_true: torch.Tensor,
    prefix_T: int,
    free_steps: int,
    device: str,
    feedback_norm: str = "prob",
):
    """
    Closed-loop rollout with teacher-forced warm-up.
    Returns:
      Y_all: outputs for warm-up+free  [B, prefix_T+free_steps, N]
      X_all: inputs actually fed       [B, prefix_T+free_steps, N]
      prefix_T_used: int
    """
    net, N, H = _rebuild_model_from_args(saved_args, device)
    net.load_state_dict(state_dict)
    net.eval()

    B, T_avail, N_in = X_true.shape
    assert N_in == N, "Input/Output dimension mismatch."

    total_T = max(0, prefix_T) + max(0, free_steps)
    X_all = torch.zeros(B, total_T, N, device=device)
    Y_all = torch.zeros(B, total_T, N, device=device)

    with torch.no_grad():
        # warm-up
        prefix_T_used = max(0, min(prefix_T, T_avail))
        if prefix_T_used > 0:
            X_warm = X_true[:, :prefix_T_used, :].to(device)
            h0 = torch.zeros(1, B, H, device=device)
            Y_warm, H_warm_seq = net(X_warm, h0)
            X_all[:, :prefix_T_used, :] = X_warm
            Y_all[:, :prefix_T_used, :] = Y_warm
            h_prev = H_warm_seq[:, -1:, :]  # [B,1,H]
            y_prev = Y_warm[:, -1:, :]  # [B,1,N]
        else:
            h_prev = torch.zeros(B, 1, H, device=device)
            y_prev = torch.zeros(B, 1, N, device=device)

        # free run (step-by-step)
        for k in range(max(0, free_steps)):
            if feedback_norm == "prob":
                x_next = _normalize_distribution(y_prev, dim=2)
            elif feedback_norm == "none":
                x_next = y_prev
            else:
                raise ValueError(f"Unknown feedback_norm {feedback_norm}")

            y_next, h_seq = net(x_next, h_prev.transpose(0, 1))  # expects [1,B,H]
            h_prev = h_seq
            y_prev = y_next

            t = prefix_T_used + k
            X_all[:, t : t + 1, :] = x_next
            Y_all[:, t : t + 1, :] = y_next

    return Y_all.cpu(), X_all.cpu(), int(prefix_T_used)


# ------------------------- Evaluators -------------------------


def evaluate_open(ckpt_path, device="cpu", data_path=None):
    """Teacher-forced evaluation: true inputs X drive the model."""
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]

    if data_path is None:
        X = ckpt["X_mini"].clone()
        Y_true = ckpt["Target_mini"].clone()
    else:
        external = torch.load(data_path, map_location=device)
        X, Y_true = external["X_mini"], external["Target_mini"]

    Y_out = _forward_sequence(ckpt["state_dict"], X, args, device)

    out = {
        "mode": "open",
        "ckpt": str(ckpt_path),
        "data": "" if data_path is None else str(data_path),
        "noise_scale": "",
        "prefix_T": "",
        "div_thresh_deg": "",
        "div_consec": "",
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(_ring_decode_r2_from_outputs(Y_out, Y_true))
    out["time_to_divergence"] = ""
    out["phase_drift_per_step"] = ""
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    return out


def evaluate_replay(ckpt_path, device="cpu", noise_scale=0.01):
    """Replay: drive the trained model with Gaussian noise (same shape as X)."""
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()

    X_noise = torch.normal(mean=0.0, std=noise_scale, size=X.shape)
    Y_out = _forward_sequence(ckpt["state_dict"], X_noise, args, device)

    out = {
        "mode": "replay",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": float(noise_scale),
        "prefix_T": "",
        "div_thresh_deg": "",
        "div_consec": "",
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(_ring_decode_r2_from_outputs(Y_out, Y_true))
    out["time_to_divergence"] = ""
    out["phase_drift_per_step"] = ""
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    return out


def evaluate_prediction(
    ckpt_path,
    device="cpu",
    noise_scale=0.01,
    prefix_T=10,
    div_thresh_deg=30.0,
    div_consec=10,
):
    """
    Prediction (hybrid): teacher-forced prefix spliced into noisy input (still open-loop).
    Good for short-horizon robustness (not feedback yet).
    """
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()
    T = X.shape[1]
    prefix_T = max(0, min(prefix_T, T))

    X_pred = torch.normal(mean=0.0, std=noise_scale, size=X.shape)
    if prefix_T > 0:
        X_pred[:, :prefix_T, :] = X[:, :prefix_T, :]

    Y_out = _forward_sequence(ckpt["state_dict"], X_pred, args, device)

    out = {
        "mode": "prediction",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": float(noise_scale),
        "prefix_T": int(prefix_T),
        "div_thresh_deg": float(div_thresh_deg),
        "div_consec": int(div_consec),
        "feedback_norm": "",
        "free_steps": "",
        "mse": _metric_mse(Y_true, Y_out),
        "mean_corr": _metric_corr_per_feat(Y_true, Y_out),
    }
    out.update(_angle_error_concentration(Y_out, Y_true))
    out.update(_residual_stats(Y_out, Y_true))
    out.update(_ring_decode_r2_from_outputs(Y_out, Y_true))

    thresh_rad = math.radians(div_thresh_deg)
    out["time_to_divergence"] = _time_to_divergence(
        Y_out, Y_true, thresh_rad, div_consec
    )
    out["phase_drift_per_step"] = _phase_drift_per_step(Y_out, Y_true)
    out["mse_free"] = out["mean_corr_free"] = ""
    out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
    out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
    out["ring_decode_R2_free"] = ""
    return out


def evaluate_closed(
    ckpt_path,
    device="cpu",
    prefix_T=10,
    free_steps=100,
    feedback_norm="prob",
    div_thresh_deg=30.0,
    div_consec=10,
):
    """Closed-loop: teacher-forced warm-up (prefix_T), then fully autonomous."""
    ckpt = _load_ckpt(ckpt_path, map_location=device)
    args = ckpt["args"]
    X_true = ckpt["X_mini"].clone()
    Y_true = ckpt["Target_mini"].clone()

    T_avail = X_true.shape[1]
    prefix_T = max(0, min(prefix_T, T_avail))
    free_steps = max(0, min(free_steps, T_avail - prefix_T))

    Y_all, X_all, prefix_T_used = _forward_closed_loop(
        ckpt["state_dict"],
        args,
        X_true,
        prefix_T,
        free_steps,
        device,
        feedback_norm=feedback_norm,
    )
    Y_true_eval = Y_true[:, : prefix_T_used + free_steps, :]

    out = {
        "mode": "closed",
        "ckpt": str(ckpt_path),
        "data": "",
        "noise_scale": "",
        "prefix_T": int(prefix_T_used),
        "div_thresh_deg": float(div_thresh_deg),
        "div_consec": int(div_consec),
        "feedback_norm": feedback_norm,
        "free_steps": int(free_steps),
        "mse": _metric_mse(Y_true_eval, Y_all),
        "mean_corr": _metric_corr_per_feat(Y_true_eval, Y_all),
    }
    out.update(_angle_error_concentration(Y_all, Y_true_eval))
    out.update(_residual_stats(Y_all, Y_true_eval))
    out.update(_ring_decode_r2_from_outputs(Y_all, Y_true_eval))

    if free_steps > 1:
        sl = slice(prefix_T_used, prefix_T_used + free_steps)
        Y_free = Y_all[:, sl, :]
        T_free = Y_true[:, sl, :]
        thresh_rad = math.radians(div_thresh_deg)
        out["time_to_divergence"] = _time_to_divergence(
            Y_free, T_free, thresh_rad, div_consec
        )
        out["phase_drift_per_step"] = _phase_drift_per_step(Y_free, T_free)

        out["mse_free"] = _metric_mse(T_free, Y_free)
        out["mean_corr_free"] = _metric_corr_per_feat(T_free, Y_free)
        ang_free = _angle_error_concentration(Y_free, T_free)
        res_free = _residual_stats(Y_free, T_free)
        r2_free = _ring_decode_r2_from_outputs(Y_free, T_free)
        out["angle_error_R_free"] = ang_free["angle_error_R"]
        out["angle_error_circ_var_free"] = ang_free["angle_error_circ_var"]
        out["residual_lag1_autocorr_free"] = res_free["residual_lag1_autocorr"]
        out["residual_L2_mean_free"] = res_free["residual_L2_mean"]
        out["ring_decode_R2_free"] = r2_free["ring_decode_R2"]
    else:
        out["time_to_divergence"] = out["phase_drift_per_step"] = ""
        out["mse_free"] = out["mean_corr_free"] = ""
        out["angle_error_R_free"] = out["angle_error_circ_var_free"] = ""
        out["residual_lag1_autocorr_free"] = out["residual_L2_mean_free"] = ""
        out["ring_decode_R2_free"] = ""

    return out


# ------------------------- CSV I/O -------------------------


def _write_csv(row_dict: dict, csv_path: Path):
    """Append one row to CSV, creating header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        # run description
        "mode",
        "ckpt",
        "data",
        "noise_scale",
        "prefix_T",
        "div_thresh_deg",
        "div_consec",
        "feedback_norm",
        "free_steps",
        # core metrics
        "mse",
        "mean_corr",
        "angle_error_R",
        "angle_error_circ_var",
        "residual_lag1_autocorr",
        "residual_L2_mean",
        "ring_decode_R2",
        # dynamics
        "time_to_divergence",
        "phase_drift_per_step",
        # free-only (closed)
        "mse_free",
        "mean_corr_free",
        "angle_error_R_free",
        "angle_error_circ_var_free",
        "residual_lag1_autocorr_free",
        "residual_L2_mean_free",
        "ring_decode_R2_free",
    ]
    row = {k: ("" if row_dict.get(k) is None else row_dict.get(k)) for k in header}
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ------------------------- Sweep helpers -------------------------


def _iter_ckpts(glob_pattern: str):
    """
    Yield checkpoint Paths matching a glob (supports **).
    Example: './runs/**/*.pth.tar'
    """
    base = Path(".")
    # Path.glob supports '**' recursion directly
    for p in base.glob(glob_pattern):
        if p.is_file() and (
            p.suffixes[-2:] == [".pth", ".tar"]
            or p.suffix == ".pt"
            or p.suffix == ".pth"
        ):
            yield p


def _evaluate_one_ckpt(ckpt: Path, args, csv_path: Path):
    """Run the selected mode(s) for a single checkpoint and append to CSV."""
    if args.mode in ("open", "all"):
        r = evaluate_open(ckpt, device=args.device, data_path=args.data)
        _write_csv(r, csv_path)
        print(f"[Open]       {ckpt} -> {csv_path}")

    if args.mode in ("replay", "all"):
        r = evaluate_replay(ckpt, device=args.device, noise_scale=args.noise_scale)
        _write_csv(r, csv_path)
        print(f"[Replay]     {ckpt} -> {csv_path}")

    if args.mode in ("prediction", "all"):
        r = evaluate_prediction(
            ckpt,
            device=args.device,
            noise_scale=args.noise_scale,
            prefix_T=args.prefix_T,
            div_thresh_deg=args.div_thresh_deg,
            div_consec=args.div_consec,
        )
        _write_csv(r, csv_path)
        print(f"[Prediction] {ckpt} -> {csv_path}")

    if args.mode in ("closed", "all"):
        r = evaluate_closed(
            ckpt,
            device=args.device,
            prefix_T=args.prefix_T,
            free_steps=args.free_steps,
            feedback_norm=args.feedback_norm,
            div_thresh_deg=args.div_thresh_deg,
            div_consec=args.div_consec,
        )
        _write_csv(r, csv_path)
        print(f"[Closed]     {ckpt} -> {csv_path}")


# ------------------------- CLI -------------------------


def main():
    p = argparse.ArgumentParser(
        description="Evaluate ElmanRNN: open / replay / prediction / closed (CSV only)"
    )

    # Choose one checkpoint OR a glob
    p.add_argument(
        "--ckpt", type=Path, default=None, help="Path to a single .pth.tar checkpoint"
    )
    p.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob for multiple checkpoints, e.g. './runs/**/*.pth.tar'",
    )

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--mode",
        type=str,
        choices=["open", "replay", "prediction", "closed", "all"],
        default="all",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV. Default: single ckpt -> ./runs_eval/<stem>_<mode>.csv; "
        "glob -> ./runs_eval/aggregate_<mode>.csv",
    )

    # open-loop options
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional external .pt/.pth.tar with {'X_mini','Target_mini'} for open-loop",
    )

    # replay / prediction options
    p.add_argument("--noise-scale", type=float, default=0.01)
    p.add_argument("--prefix-T", type=int, default=10)  # used by prediction and closed

    # divergence/drift thresholds (prediction & closed)
    p.add_argument("--div-thresh-deg", type=float, default=30.0)
    p.add_argument("--div-consec", type=int, default=10)

    # closed-loop options
    p.add_argument(
        "--free-steps",
        type=int,
        default=100,
        help="Closed-loop autonomous rollout length after prefix",
    )
    p.add_argument(
        "--feedback-norm",
        type=str,
        default="prob",
        choices=["prob", "none"],
        help="Map output->input during closed loop: prob=normalize, none=raw",
    )

    args = p.parse_args()

    # Decide CSV path
    if args.glob:
        mode_tag = args.mode
        csv_path = (
            Path("./runs_eval") / f"aggregate_{mode_tag}.csv"
            if args.csv is None
            else args.csv
        )
        matched = list(_iter_ckpts(args.glob))
        if not matched:
            print(f"[WARN] No checkpoints match glob: {args.glob}")
            return
        print(f"[INFO] Found {len(matched)} checkpoints via glob.")
        for ckpt in matched:
            _evaluate_one_ckpt(ckpt, args, csv_path)
    else:
        if args.ckpt is None:
            raise SystemExit("Provide --ckpt or --glob")
        ckpt_stem = args.ckpt.stem
        default_name = f"{ckpt_stem}_{args.mode}.csv"
        csv_path = Path("./runs_eval") / default_name if args.csv is None else args.csv
        _evaluate_one_ckpt(args.ckpt, args, csv_path)


def _vector_norm_compat(x: torch.Tensor, dim=None, keepdim=False, eps=0.0):
    # L2 norm without torch.linalg
    if dim is None:
        return torch.sqrt(torch.clamp((x * x).sum(), min=eps))
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


if __name__ == "__main__":
    main()
