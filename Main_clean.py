"""
A clean, commented version of Main.py for revision models (Elman RNN trainer)
Original author: Y.C. (8/8/2023)


This script trains an Elman-style RNN on a time-series task using BPTT.
It supports options for:
• One-step prediction loss (pred mode)
• Autoencoder target (= input)
• Constraining/fixing specific parameter groups (input/output/recurrent)
• Partially training only a subset of parameters via masks
• Optional gradient-norm clipping and simple early stopping
• Periodic logging and checkpointing

Usage examples:

"""
import argparse
import sys
import os
import re
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from RNN_Class import *
from tqdm import tqdm
import random
from typing import Optional
import math

parser = argparse.ArgumentParser(description="PyTorch Elman BPTT Training")
parser.add_argument(
    "--epochs",
    default=50000,
    type=int,
    metavar="N",
    help="Number of total epochs to run (default: 50k)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="Initial learning rate (SGD only, default: 0.01)",
)
parser.add_argument(
    "--lr_step",
    default="",
    type=str,
    help="Comma-separated epochs at which LR is halved (e.g. 1000,2000)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1000,
    type=int,
    metavar="N",
    help="How often to log and store hidden/output snapshots",
)
parser.add_argument(
    "-g", "--gpu", default=1, type=int, help="Enable GPU computing if nonzero"
)
parser.add_argument(
    "-n",
    "--n",
    default=200,
    type=int,
    help="Input/output dimensionality (feature size)",
)
parser.add_argument("--hidden-n", default=200, type=int, help="Hidden dimension size")
parser.add_argument(
    "--savename", default="", type=str, help="Base path (no extension) for outputs"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Path to a checkpoint to resume from (loads state_dict only)",
)
parser.add_argument(
    "--ae",
    default=0,
    type=int,
    help="If 1, use Autoencoder objective (Target == Input). Used for testing, not training.",
)
parser.add_argument(
    "--partial",
    default=0,
    type=float,
    help="Sparsity level (0-1). Proportion of RNN params that are frozen by mask.",
)
parser.add_argument(
    "--input",
    default="",
    type=str,
    help="Path to a .pt/.pth.tar containing {'X_mini','Target_mini'}",
)
parser.add_argument(
    "--fixi",
    default=0,
    type=int,
    help="Fix input matrix (rnn.weight_ih_l0) with various modes (1,2,3-> see code)",
)
parser.add_argument(
    "--fixw",
    default=0,
    type=int,
    help="Fix recurrent weight (rnn.weight_hh_l0) and its bias to constants.",
)
parser.add_argument(
    "--constraini",
    default=0,
    type=int,
    help="If nonzero, clamp input matrix (weight_ih) to be nonnegative after each step.",
)
parser.add_argument(
    "--constraino",
    default=0,
    type=int,
    help="If nonzero, clamp output matrix (linear.weight) to be nonnegative after each step.",
)
parser.add_argument(
    "--fixo", default=0, type=int, help="Fix the output matrix (linear.weight)"
)
parser.add_argument(
    "--clamp_norm",
    default=0,
    type=float,
    help="If >0, clip total gradient norm to this value each step.",
)
parser.add_argument(
    "--nobias",
    default=0,
    type=int,
    help="whether to remove all bias term in RNN module",
)
parser.add_argument(
    "--rnn_act",
    default="",
    type=str,
    help="Hidden nonlinearity: 'relu' to override (default tanh)",
)
parser.add_argument(
    "--ac_output",
    default="",
    type=str,
    help="Output activation override: tanh|relu|sigmoid (default softmax along dim=2 in class)",
)
parser.add_argument(
    "--pred",
    default=0,
    type=int,
    help="If nonzero, use one-step prediction loss (shift X/Target by 1)",
)
parser.add_argument(
    "--noisy_train",
    default=0,
    type=float,
    help="If > 0, add multiplicative Gaussian noise ~N(0, (X*noisy_train)^2) to X and Target each step",
)
parser.add_argument("--seed", type=int, default=1337, help="Global RNG seed")
parser.add_argument(
    "--whh_type",
    type=str,
    default="none",
    choices=[
        "none",
        "cent",
        "cent-cyc",
        "shifted",
        "shifted-cyc",
        "identity",
        "shift",
        "shift-cyc",
    ],
    help="Hidden-weight init type; use 'none' for standard random init (no override).",
)
parser.add_argument(
    "--whh_norm",
    type=str,
    default="raw",
    choices=["frobenius", "raw"],
    help="Normalization strategy for hidden weight",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=None,
    help="Symmetry percent in [0,1]. Needed for shifted/shift variants; ignored for cent/cent-cyc/identity",
)
parser.add_argument(
    "--out_root",
    type=str,
    default="./runs",
    help="Root directory for outputs; subfolders mirror hidden-weight init structure",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default="",
    help="Optional extra tag appended to filename (e.g., seed or note)",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=1,
    help="How many sequential runs to perform; each run saves under run_XX/",
)
parser.add_argument(
    "--run_offset",
    type=int,
    default=-1,
    help="If >=0, start run numbering at this index; otherwise auto-pick the next free index.",
)
parser.add_argument(
    "--noisy",
    action="store_true",
    help="If set, use 'noisy' hidden-weight init base and save under ./runs/ElmanRNN/noisy/... (otherwise 'clean').",
)
parser.add_argument(
    "--whh_path",
    type=str,
    default="",
    help=(
        "Path to a hidden-weight .npy (H×H). If set, overrides --whh_type/--whh_norm/--alpha "
        "and outputs are saved under ./runs/ElmanRNN/<subdirs from whh_path>/"
        "<filename_stem>..."
    ),
)


def main():
    """Entry point: parse args, build data/model/optimizer, train, and save artifacts"""
    global args, f

    args = parser.parse_args()
    if args.whh_path and args.whh_type != "none":
        print(
            "[whh] NOTE: --whh_path provided; overriding --whh_type/--whh_norm/--alpha"
        )
    lr = args.lr
    n_epochs = args.epochs
    N = args.n
    hidden_N = args.hidden_n
    set_seed(args.seed)
    f = None  # global log handle (opened per run later)

    # Auto-build savename if not provided, mirroring init structure
    if not args.savename:
        if args.whh_path:
            # Mirror the provided hidden-weight path under ./runs/ElmanRNN/...
            # Example:
            # whh_path = ./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/learned/random_n100/raw/random_n100_learned.npy
            # out_dir  = ./runs/ElmanRNN/learned/random_n100/raw
            # savename = ./runs/ElmanRNN/learned/random_n100/raw/random_n100_learned
            from pathlib import Path

            p = Path(args.whh_path).resolve()
            parts = list(p.parts)
            stem = p.stem  # filename without extension

            # Find "ElmanRNN" in the path and take everything after it (except the filename)
            try:
                idx = parts.index("ElmanRNN")
                subdirs = parts[idx + 1 : -1]  # e.g., ["learned", "random_n100", "raw"]
            except ValueError:
                # Fallback: if "ElmanRNN" not present, just use no subdirs
                subdirs = []

            out_dir = os.path.join(args.out_root, "ElmanRNN", *subdirs)
            os.makedirs(out_dir, exist_ok=True)
            args.savename = os.path.join(out_dir, stem)

        else:
            # ORIGINAL LOGIC (unchanged) for whh_type/whh_norm/alpha
            if args.whh_type == "none":
                out_dir = os.path.join(
                    args.out_root, "ElmanRNN", _mode_dir(args.noisy), "random-init"
                )
                prefix = _prefix_from_type("none")  # "random"
                fname_bits = [f"{prefix}_n{hidden_N}"]
            else:
                variant = _resolve_variant(args.whh_type)
                norm_dir = args.whh_norm
                out_dir = os.path.join(
                    args.out_root,
                    "ElmanRNN",
                    _mode_dir(args.noisy),
                    variant,
                    args.whh_type,
                    norm_dir,
                )
                if (
                    args.whh_type in {"shifted", "shifted-cyc", "shift", "shift-cyc"}
                    and args.alpha is not None
                ):
                    out_dir = os.path.join(out_dir, _alpha_tag(args.alpha))
                prefix = _prefix_from_type(args.whh_type)
                norm_short = _norm_shortname(args.whh_norm)
                fname_bits = [f"{prefix}_n{hidden_N}_{norm_short}"]
                if (
                    args.whh_type in {"shifted", "shifted-cyc", "shift", "shift-cyc"}
                    and args.alpha is not None
                ):
                    fname_bits.append(_alpha_tag(args.alpha))
            suffix = _noisy_suffix(args.noisy)
            if not args.whh_path and suffix:
                fname_bits[-1] += suffix
            if not args.whh_path and args.run_tag:
                fname_bits.append(str(args.run_tag))
            if not args.whh_path and not args.savename:
                os.makedirs(out_dir, exist_ok=True)
                args.savename = os.path.join(out_dir, "_".join(fname_bits))

        if args.run_tag:
            args.savename += f"_{args.run_tag}"

    # -----------------
    # Load training data
    # -----------------
    # Expected keys in saved file: 'X_mini' (inputs), 'Target_mini' (targets)
    loaded = torch.load(args.input)
    X_mini = loaded["X_mini"]
    Target_mini = loaded["Target_mini"]

    # Autoencoder mode: predict input itself (testing)
    if args.ae:
        log("Autoencoder scenario: Target = Input")
        Target_mini = loaded["X_mini"]

    # One-step prediction: align X and Target for next-step forecasting
    if args.pred:
        X_mini = X_mini[:, :-1, :]
        Target_mini = Target_mini[:, 1:, :]
        log("Predicting one-step ahead")

    # ---------------------
    # Define the network
    # ---------------------
    # ElmanRNN_pytorch_module_v2: (input_dim=N, hidden_dim=hidden_N, output_dim=N)
    net = ElmanRNN_pytorch_module_v2(N, hidden_N, N)

    # --- override hidden weight from disk ---
    if args.whh_path:
        try:
            log(f"[whh] loading hidden weight from (explicit path): {args.whh_path}")
            # sanity check on shape
            W = np.load(args.whh_path)
            if W.shape != (hidden_N, hidden_N):
                raise ValueError(
                    f"Hidden weight shape mismatch: expected ({hidden_N},{hidden_N}), got {W.shape}"
                )
            _load_hidden_into_elman(
                net, args.whh_path, device=net.rnn.weight_hh_l0.device
            )
        except Exception as e:
            log(f"[whh] WARNING: failed to load init from whh_path: {e}")
    elif args.whh_type != "none":
        try:
            path = _resolve_hidden_path(
                hidden_N, args.whh_type, args.whh_norm, args.alpha, noisy=args.noisy
            )
            log(f"[whh] loading hidden weight from: {path}")
            _load_hidden_into_elman(net, path, device=net.rnn.weight_hh_l0.device)
        except Exception as e:
            log(f"[whh] WARNING: failed to load init: {e}")
    else:
        log("[whh] using default random init")

    # Cache initial recurrent matrix (after any override)
    W0 = net.rnn.weight_hh_l0.detach().cpu().numpy()

    # Optionally change the RNN hidden nonlinearity
    if args.rnn_act == "relu":
        net.rnn = nn.RNN(N, hidden_N, 1, batch_first=True, nonlinearity="relu")
        log("RNN nonlinearity: elementwise relu")

    # Optionally override output activation
    if args.ac_output == "tanh":
        net.act = nn.Tanh()
        log("Change output activation function to tanh")
    elif args.ac_output == "relu":
        net.act = nn.ReLU()
        log("Change output activation function to relu")
    elif args.ac_output == "sigmoid":
        net.act = nn.Sigmoid()
        log("Change output activation function to sigmoid")

    # ------------------------------
    # Optional parameter constraints
    # ------------------------------
    # Remove hidden bias entirely (zero + freeze)
    if args.nobias:
        for name, p in net.named_parameters():
            if name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                log("Fixing RNN bias to 0")

    # Fix input matrix with different modes and freeze it
    if args.fixi:
        for name, p in net.named_parameters():
            if name == "rnn.weight_ih_l0":
                if args.fixi == 1:
                    # Positive constant matrix (uniform average)
                    p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                    log("Fixing {} to positive constant".format(name))
                elif args.fixi == 2:
                    # Preserve initialization but freeze it
                    log("Fixing {} to initialization".format(name))
                elif args.fixi == 3:
                    # Make current init nonegative by folding absolute values
                    p.data = p.data + torch.abs(p.data)
                    log("Fixing {} to positive initiation".format(name))

                p.requires_grad = False

    # Fix output matrix with different modes and freeze it
    if args.fixo:
        for name, p in net.named_parameters():
            if name == "linear.weight":
                if args.fixo == 1:
                    p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                    log("Fixing {} to positive constant".format(name))
                elif args.fixo == 2:
                    log("Fixing {} to initialization".format(name))
                elif args.fixo == 3:
                    p.data = p.data + torch.abs(p.data)
                    log("Fixing {} to positive initiation".format(name))
                p.requires_grad = False

    # Fix recurrent weight and its bias (freeze both)
    if args.fixw:
        for name, p in net.named_parameters():
            if name == "rnn.weight_hh_l0":
                p.requires_grad = False
                # Random in [-1/sqrt(N), 1/sqrt(N)] (Elman-ish scale)
                p.data = torch.rand(p.data.shape) * 2 * 1 / np.sqrt(N) - 1 / np.sqrt(N)
                log("Fixing recurrent matrix to a random matrix")
            elif name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                log("Fixing recurrent bias to 0")

    # ------------------
    # Loss & checkpoint
    # ------------------
    criterion = nn.MSELoss(reduction="mean")  # mean MSE across batch/time/features

    # Optionally resume model weights from a checkpoint(.pth.tar)
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading previous network '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint["state_dict"])
            log("=> loaded previous network '{}' ".format(args.resume))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))

    # -----------------------
    # Initial hidden state h0
    # -----------------------
    # Shape: (num_layers=1, batch_size, hidden_N)
    h0 = torch.zeros(1, X_mini.shape[0], hidden_N)  # n_layers * BatchN * NHidden

    # -------------------------------------------------
    # Multi-run wrapper: create run_XX dirs and loop
    # -------------------------------------------------
    # Decide where to start numbering runs
    if args.run_offset >= 0:
        start_idx = args.run_offset
    else:
        start_idx = _next_free_run_idx(args.savename)

    for k in range(args.num_runs):
        # 1) Pick a unique run directory (never overwrite)
        run_dir, run_idx = _ensure_unique_run_dir(args.savename, start_idx + k)
        base_stub = os.path.basename(args.savename)
        savename_run = os.path.join(run_dir, base_stub)  # use this for ALL outputs

        # 2) Per-run seeding so each run is reproducible & distinct
        set_seed(args.seed + run_idx)

        # 3) Open a per-run log file (replaces the old one-off log)
        f = open(savename_run + ".log", "w")
        log("Settings:")
        log(str(sys.argv))
        log(f"[run] index={run_idx}, seed={args.seed + run_idx}")

        # ---------------------
        # Define the network
        # ---------------------
        net = ElmanRNN_pytorch_module_v2(N, hidden_N, N)

        # --- override hidden weight from disk ---
        if args.whh_path:
            try:
                log(
                    f"[whh] loading hidden weight from (explicit path): {args.whh_path}"
                )
                # sanity check on shape
                W = np.load(args.whh_path)
                if W.shape != (hidden_N, hidden_N):
                    raise ValueError(
                        f"Hidden weight shape mismatch: expected ({hidden_N},{hidden_N}), got {W.shape}"
                    )
                _load_hidden_into_elman(
                    net, args.whh_path, device=net.rnn.weight_hh_l0.device
                )
            except Exception as e:
                log(f"[whh] WARNING: failed to load init from whh_path: {e}")
        elif args.whh_type != "none":
            try:
                path = _resolve_hidden_path(
                    hidden_N, args.whh_type, args.whh_norm, args.alpha, noisy=args.noisy
                )
                log(f"[whh] loading hidden weight from: {path}")
                _load_hidden_into_elman(net, path, device=net.rnn.weight_hh_l0.device)
            except Exception as e:
                log(f"[whh] WARNING: failed to load init: {e}")
        else:
            log("[whh] using default random init")

        # Cache initial recurrent matrix (after any override)
        W0 = net.rnn.weight_hh_l0.detach().cpu().numpy()

        # Optionally change the RNN hidden nonlinearity
        if args.rnn_act == "relu":
            net.rnn = nn.RNN(N, hidden_N, 1, batch_first=True, nonlinearity="relu")
            log("RNN nonlinearity: elementwise relu")

        # Optionally override output activation
        if args.ac_output == "tanh":
            net.act = nn.Tanh()
            log("Change output activation function to tanh")
        elif args.ac_output == "relu":
            net.act = nn.ReLU()
            log("Change output activation function to relu")
        elif args.ac_output == "sigmoid":
            net.act = nn.Sigmoid()
            log("Change output activation function to sigmoid")

        # ------------------------------
        # Optional parameter constraints
        # ------------------------------
        if args.nobias:
            for name, p in net.named_parameters():
                if name == "rnn.bias_hh_l0":
                    p.requires_grad = False
                    p.data.fill_(0)
                    log("Fixing RNN bias to 0")

        if args.fixi:
            for name, p in net.named_parameters():
                if name == "rnn.weight_ih_l0":
                    if args.fixi == 1:
                        p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                        log("Fixing {} to positive constant".format(name))
                    elif args.fixi == 2:
                        log("Fixing {} to initialization".format(name))
                    elif args.fixi == 3:
                        p.data = p.data + torch.abs(p.data)
                        log("Fixing {} to positive initiation".format(name))
                    p.requires_grad = False

        if args.fixo:
            for name, p in net.named_parameters():
                if name == "linear.weight":
                    if args.fixo == 1:
                        p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                        log("Fixing {} to positive constant".format(name))
                    elif args.fixo == 2:
                        log("Fixing {} to initialization".format(name))
                    elif args.fixo == 3:
                        p.data = p.data + torch.abs(p.data)
                        log("Fixing {} to positive initiation".format(name))
                    p.requires_grad = False

        if args.fixw:
            for name, p in net.named_parameters():
                if name == "rnn.weight_hh_l0":
                    p.requires_grad = False
                    p.data = torch.rand(p.data.shape) * 2 * 1 / np.sqrt(
                        N
                    ) - 1 / np.sqrt(N)
                    log("Fixing recurrent matrix to a random matrix")
                elif name == "rnn.bias_hh_l0":
                    p.requires_grad = False
                    p.data.fill_(0)
                    log("Fixing recurrent bias to 0")

        # ------------------
        # Loss & resume
        # ------------------
        criterion = nn.MSELoss(reduction="mean")

        if args.resume:
            if os.path.isfile(args.resume):
                log("=> loading previous network '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                net.load_state_dict(checkpoint["state_dict"])
                log("=> loaded previous network '{}' ".format(args.resume))
            else:
                log("=> no checkpoint found at '{}'".format(args.resume))

        # -----------------------
        # Move to device
        # -----------------------
        if args.gpu:
            print(
                "Cuda device availability: {}".format(torch.cuda.is_available()), file=f
            )
            criterion = criterion.cuda()
            net = net.cuda()
            X_mini = X_mini.cuda()
            Target_mini = Target_mini.cuda()
            h0 = h0.cuda()

        # ---------------
        # Optimizer (SGD)
        # ---------------
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        # ------------------------------------------------
        # Build parameter masks for partial training (if any)
        # ------------------------------------------------
        if args.partial:
            log("Training sparsity:{}".format(args.partial))
            Mask_W = np.random.uniform(0, 1, (hidden_N, hidden_N))
            Mask_B = np.random.uniform(0, 1, (hidden_N))
            Mask_W = Mask_W > args.partial
            Mask_B = Mask_B > args.partial

            if hasattr(args, "nonoverlap") and args.nonoverlap:
                Mask_W = ~(~(Mask_W) & checkpoint["Mask_W"])
                Mask_B = ~(~(Mask_B) & checkpoint["Mask_B"])

            Mask = []
            for name, p in net.named_parameters():
                if name == "rnn.weight_hh_l0" or name == "hidden_linear.weight":
                    Mask.append(Mask_W)
                    log("Partially train RNN weight")
                elif name == "rnn.bias_hh_l0" or name == "hidden_linear.bias":
                    Mask.append(Mask_B)
                    log("Partially train RNN bias")
                else:
                    Mask.append(np.zeros(p.shape))
        else:
            Mask = []
            for name, p in net.named_parameters():
                Mask.append(np.zeros(p.shape))

        # ----------------------
        # Train and time it
        # ----------------------
        start = time.time()
        (
            net,
            loss_list,
            grad_list,
            hidden_rep,
            output_rep,
            snapshot_epochs,
            grad_metrics_history,
            hidden_metrics_history,
            weight_structure_history,
            W_hh_history,
            error_metrics_history,
        ) = train_partial(
            X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask, W0
        )
        end = time.time()
        deltat = end - start
        log("Total training time: {0:.1f} minutes".format(deltat / 60))

        # -----------------
        # Plot loss curve
        # -----------------
        plt.figure()
        plt.plot(loss_list)
        plt.title("Loss iteration")
        plt.savefig(savename_run + ".png")  # <--- use savename_run here

        # -----------------------------------
        # Save checkpoint + artifacts (PER RUN)
        # -----------------------------------
        save_dict = {
            "state_dict": net.state_dict(),
            "y_hat": np.array(output_rep),
            "hidden": np.array(hidden_rep),
            "X_mini": X_mini.detach().cpu(),
            "Target_mini": Target_mini.detach().cpu(),
            "loss": loss_list,
            "mean_squared_grads": grad_list,
            "grad_metrics": {
                "history": grad_metrics_history,
                "global_L2_pre": [
                    h["pre"]["global"]["L2"] for h in grad_metrics_history
                ],
                "global_L2_post": [
                    h["post"]["global"]["L2"] for h in grad_metrics_history
                ],
                "global_RMS_pre": [
                    h["pre"]["global"]["RMS"] for h in grad_metrics_history
                ],
                "global_RMS_post": [
                    h["post"]["global"]["RMS"] for h in grad_metrics_history
                ],
                "param_names": list(grad_metrics_history[0]["post"]["groups"].keys())
                if len(grad_metrics_history) > 0
                else [],
                "param_shapes": (
                    {
                        k: v["shape"]
                        for k, v in grad_metrics_history[0]["post"]["groups"].items()
                    }
                    if len(grad_metrics_history) > 0
                    else {}
                ),
            },
            "hidden_metrics": {"history": hidden_metrics_history},
            "weight_structure": {"history": weight_structure_history},
            "weights": {
                "W_hh_init": W0.astype(np.float16),
                "W_hh_history": np.array(W_hh_history, dtype=np.float16),
            },
            "error_metrics": {
                "history": error_metrics_history,
                "angle_error_R": [e["angle_error_R"] for e in error_metrics_history],
                "residual_lag1_autocorr": [
                    e["residual_lag1_autocorr"] for e in error_metrics_history
                ],
                "residual_L2_mean": [
                    e["residual_L2_mean"] for e in error_metrics_history
                ],
            },
            "n_epochs": int(n_epochs),
            "args": {
                **vars(args),
                "run_idx": run_idx,
            },  # store run_idx for traceability
            "env": env_report(),
            "rng": rng_snapshot(),
            "snapshot_epochs": snapshot_epochs,
        }
        if args.partial:
            save_dict["Mask_W"] = Mask_W
            save_dict["Mask_B"] = Mask_B

        torch.save(save_dict, savename_run + ".pth.tar")
        f.close()
        f = None  # close log file


# -------------------------------------------------
# Training loop with optional partial-parameter masks
# -------------------------------------------------
def train_partial(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask, W0=None
):
    """
    Train with optional untrainable masks on selected parameters.


    Args:
    X_mini (torch.Tensor): [batch, seq, feat]
    Target_mini (torch.Tensor): [batch, seq, feat]
    h0 (torch.Tensor): initial hidden state [1, batch, hidden]
    n_epochs (int): number of epochs
    net (nn.Module): RNN model
    criterion: loss function (MSE)
    optimizer: torch optimizer
    Mask (list[np.ndarray]): aligned with net.parameters(); True == freeze (zero grad)


    Returns:
    net: trained model
    loss_list (np.ndarray): per-epoch losses
    grad_list (list[list[float]]): mean(grad^2) per parameter group each epoch
    hidden_rep (list[np.ndarray]): periodic hidden snapshots
    output_rep (list[np.ndarray]): periodic output snapshots
    """
    # Count trainable parameters and echo names
    count = 0
    for name, p in net.named_parameters():
        if p.requires_grad:
            print(name)
            count += 1
    print("{} parameters to optimize".format(count))

    loss_list = []
    batch_size, SeqN, N = X_mini.shape
    _, _, hidden_N = h0.shape
    start = time.time()
    stop = False  # early stopping flag

    hidden_rep = []  # periodic hidden state snapshots
    output_rep = []  # periodic output snapshots
    grad_list = []  # mean(grad^2) per param group each snapshot
    grad_metrics_history = []  # detailed grad metrics per snapshot
    snapshot_epochs = []  # epochs at which snapshots were taken
    hidden_metrics_history = []  # per-snapshot hidden stats/dynamics/geometry/function
    weight_structure_history = []  # per-snapshot W_hh symmetry/asymmetry metrics
    W_hh_history = []  # raw W_hh per snapshot (float16)
    error_metrics_history = []  # per-snapshot error metrics (angle + residuals)

    with tqdm(total=n_epochs, desc="Progress", unit="epoch") as pbar:
        for epoch in range(n_epochs):
            if stop:
                break
            # Optional LR schedule: halve LR at specified epochs
            if args.lr_step:
                lr_step = list(map(int, args.lr_step.split(",")))
                if epoch in lr_step:
                    log("Decrease lr to 50per at epoch {}".format(epoch))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5

            # Optional noise injection (proportional to input magnitude)
            if args.noisy_train:
                random_part = torch.normal(
                    mean=torch.zeros(X_mini.shape), std=X_mini.cpu() * args.noisy_train
                ).to(X_mini.device)
                X = X_mini + random_part
                Target = Target_mini + random_part
            else:
                X = X_mini
                Target = Target_mini

            # Forward pass
            output, h_seq = net(X, h0)

            # Loss and backward pass
            optimizer.zero_grad()
            loss = criterion(output, Target)
            loss.backward()

            # --- Gradient metrics (PRE-CLIP / PRE-MASK) ---
            grad_metrics_pre = _compute_grad_metrics(net)

            # Record simple grad statistics (mean of squared gradients per param)
            tmp = []
            for name, p in net.named_parameters():
                if p.requires_grad:
                    if p.grad is None:
                        print(f"WARNING: {name} has no grad")
                        tmp.append(0.0)
                    else:
                        grad_np = p.grad.detach().cpu().numpy()
                        tmp.append(np.mean(grad_np**2))
            grad_list.append(tmp)

            # Apply masks: zero gradients where Mask == True (freeze for those entries)
            for l, p in enumerate(net.parameters()):
                if p.requires_grad:
                    mask = torch.from_numpy(Mask[l].astype(bool)).to(p.grad.device)
                    p.grad.data[mask] = 0.0

            # Optional gradient norm clipping (global norm)
            if args.clamp_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clamp_norm)

            # --- Gradient metrics (POST-CLIP / POST-MASK) ---
            grad_metrics_post = _compute_grad_metrics(net)

            grad_metrics_history.append(
                {
                    "epoch": int(epoch),
                    "pre": grad_metrics_pre,  # global + per-parameter
                    "post": grad_metrics_post,  # global + per-parameter
                    "clipped": bool(
                        args.clamp_norm
                        and (
                            grad_metrics_post["global"]["L2"]
                            < grad_metrics_pre["global"]["L2"]
                        )
                    ),
                }
            )

            optimizer.step()  # Update parameters

            # Optional nonegativity constraints after the step
            if args.constraini:
                for name, p in net.named_parameters():
                    if name == "rnn.weight_ih_l0":
                        p.data.clamp_(0)
            if args.constraino:
                for name, p in net.named_parameters():
                    if name == "linear.weight":
                        p.data.clamp_(0)

            # Simple early stoppping: requires >1000; checks small loss change and low absolute loss
            if epoch > 1000:
                diff = [
                    loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)
                ]
                mean_diff = np.mean(abs(np.array(diff[-5:-1])))
                init_loss = loss_list[0]
                if (
                    mean_diff < loss.item() * 0.00001
                    and loss.item() < init_loss * 0.010
                ):
                    stop = True

            # Bookkeeping
            loss_list = np.append(loss_list, loss.item())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

            # Periodic logging and snapshots of hidden/output states
            if epoch % args.print_freq == 0:
                end = time.time()
                deltat = end - start
                start = time.time()
                hidden_rep.append(h_seq.detach().cpu().numpy())
                output_rep.append(output.detach().cpu().numpy())
                snapshot_epochs.append(epoch)
                # --- Hidden-state metrics (Activation, Stability, Dynamics, Geometry, Function) ---
                act_stats = _hidden_activation_stats(
                    h_seq, act=("tanh" if args.rnn_act != "relu" else "relu")
                )
                dyn_metrics = _temporal_metrics(h_seq)
                geom_metrics = _geometry_metrics(h_seq, max_components=10)
                # Function: decode ring variable from Target (use same time slice as h_seq)
                tgt_slice = (
                    Target[:, : h_seq.shape[1], :]
                    if "Target" in locals()
                    else Target_mini[:, : h_seq.shape[1], :]
                )
                # --- Error-centric metrics (angular concentration + residual autocorr) ---
                # Use same time window as hidden/output snapshot
                out_slice = output[
                    :, : h_seq.shape[1], :
                ]  # output already matches, this is explicit
                err_ang = _angle_error_metrics(out_slice, tgt_slice)
                err_res = _residual_autocorr(out_slice, tgt_slice)
                error_metrics_history.append(
                    {"epoch": int(epoch), **err_ang, **err_res}
                )
                func_metrics = _decode_ring_linear(h_seq, tgt_slice)

                # Merge into one record
                hidden_metrics_history.append(
                    {
                        "epoch": int(epoch),
                        "activation": act_stats,
                        "dynamics": dyn_metrics,
                        "geometry": geom_metrics,
                        "function": func_metrics,
                    }
                )

                # --- Weight-structure metrics (S/A mix, non-normality, drift) ---
                wstruct = _weight_structure_metrics(net, W0=W0)
                weight_structure_history.append({"epoch": int(epoch), **wstruct})
                # --- Save raw W_hh for offline analysis (compact float16) ---
                W_snap = net.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float16)
                W_hh_history.append(W_snap)
                print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
                print("Loss: {:.4f}".format(loss.item()))
                print("Time Elapsed since last display: {0:.1f} seconds".format(deltat))
                print(
                    "Estimated remaining time: {0:.1f} minutes".format(
                        deltat * (n_epochs - epoch) / args.print_freq / 60
                    )
                )

    return (
        net,
        loss_list,
        grad_list,
        hidden_rep,
        output_rep,
        snapshot_epochs,
        grad_metrics_history,
        hidden_metrics_history,
        weight_structure_history,
        W_hh_history,
        error_metrics_history,
    )


# ------------------
# Utility functions
# ------------------
def set_seed(seed: int = 1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # keep cuDNN fast/non-deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def env_report():
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def rng_snapshot():
    return {
        "py_random": None,  # fill only if you seeded Python's random
        "np_random": np.random.get_state()[1].tolist(),  # the uint32 keys
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _alpha_tag(a: float) -> str:
    """0.90 -> 'sym0p90', 1.0 -> 'sym1p00'"""
    pct = int(round(a * 100))
    major = pct // 100
    minor = pct % 100
    return f"sym{major}p{minor:02d}"


def _resolve_variant(whh_type: str) -> str:
    if whh_type in {"cent", "cent-cyc", "shifted", "shifted-cyc"}:
        return "mh-variants"
    elif whh_type in {"identity", "shift", "shift-cyc"}:
        return "shift-variants"
    elif whh_type == "none":
        return "random-init"
    else:
        raise ValueError(f"Unknown whh_type {whh_type}")


def _prefix_from_type(whh_type: str) -> str:
    mapping = {
        "cent": "centmh",
        "cent-cyc": "centcycmh",
        "shifted": "shiftmh",
        "shifted-cyc": "shiftcycmh",
        "identity": "identity",
        "shift": "shift",
        "shift-cyc": "shiftcyc",
        "none": "random",
    }
    return mapping[whh_type]


def _norm_shortname(norm: str) -> str:
    return {"frobenius": "fro", "raw": "raw"}[norm]


def _mode_dir(noisy: bool) -> str:
    return "noisy" if noisy else "clean"


def _noisy_suffix(noisy: bool) -> str:
    return "_noisy" if noisy else ""


def _resolve_hidden_path(
    hidden_N: int,
    whh_type: str,
    whh_norm: str,
    alpha: Optional[float] = None,
    noisy: bool = False,
) -> str:
    variant = _resolve_variant(whh_type)
    norm_dir = whh_norm  # directory name is full strategy
    mode = _mode_dir(noisy)
    base = f"./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/{mode}/{variant}/{whh_type}/{norm_dir}"
    # alpha tag needed for {shifted, shifted-cyc, shift, shift-cyc}
    if whh_type in {"shifted", "shifted-cyc", "shift", "shift-cyc"}:
        if alpha is None:
            raise ValueError(
                "alpha is required for this whh_type (e.g., --alpha 0.90)."
            )
        base = f"{base}/{_alpha_tag(alpha)}"
    prefix = _prefix_from_type(whh_type)
    norm_short = _norm_shortname(whh_norm)
    # If your files are strictly n100, set hidden_N=100 (or hardcode 100 here).
    fname = f"{prefix}_n{hidden_N}_{norm_short}" + _noisy_suffix(noisy) + ".npy"
    return os.path.join(base, fname)


def _load_hidden_into_elman(
    net: nn.Module, npy_path: str, device: Optional[torch.device] = None
):
    """Copy an (H,H) numpy array into ElmanRNN's recurrent weight."""
    W = np.load(npy_path)
    thW = torch.as_tensor(W, dtype=torch.float32)
    if device is None:
        device = next(net.parameters()).device
    thW = thW.to(device)
    with torch.no_grad():
        # ElmanRNN_pytorch_module_v2 uses nn.RNN => weight_hh_l0 exists
        net.rnn.weight_hh_l0.copy_(thW)
        # keep biases as-is (minimal & straightforward)


def _compute_grad_metrics(net: nn.Module):
    """
    Returns a dict with global and per-parameter gradient metrics.
    Assumes .backward() has been called. Reads current .grad tensors.
    """
    total_sq = 0.0
    total_n = 0
    groups = {}  # name -> {'L2':..., 'RMS':..., 'n':..., 'shape':...}

    for name, p in net.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        sq = float(torch.sum(g * g).item())
        n = g.numel()
        # L2 = sqrt(sum g^2); RMS = sqrt(mean g^2)
        L2 = float(torch.sqrt(torch.sum(g * g)).item())
        RMS = float((sq / max(n, 1)) ** 0.5)

        groups[name] = {
            "L2": L2,
            "RMS": RMS,
            "n": int(n),
            "shape": tuple(p.shape),
        }
        total_sq += sq
        total_n += n

    global_L2 = float((total_sq**0.5))
    global_RMS = float(((total_sq / max(total_n, 1)) ** 0.5))

    return {
        "global": {"L2": global_L2, "RMS": global_RMS, "n": int(total_n)},
        "groups": groups,
    }


def _hidden_activation_stats(h_seq, act="tanh", sat_eps=0.95):
    """
    h_seq: torch.Tensor [batch, T, H] hidden activity sequence
    Returns scalar stats over batch×time×units (mean, std across all batchxtimexunits, sat_ratio fraction of activations near saturation if tanh, energy_L2_mean mean per-step L2 norm averaged over time).
    """
    h = h_seq.detach()
    mean = float(h.mean().item())
    std = float(h.std(unbiased=False).item())
    if act == "tanh":
        # fraction near saturation (|h| >= sat_eps)
        sat = float(((h.abs() >= sat_eps).float().mean().item()))
    else:
        sat = None
    # stability proxy: mean L2 over time (per-step norm averaged)
    # reshape to [B*T, H]
    BT, H = h.shape[0] * h.shape[1], h.shape[2]
    h2 = h.reshape(-1, H)
    energy = float(_vector_norm_compat(h2, dim=-1).mean().item())
    return {"mean": mean, "std": std, "sat_ratio": sat, "energy_L2_mean": energy}


def _temporal_metrics(h_seq):
    """
    Quantify simple temporal metrics from hidden state sequence.
    Returns:
      - lag1_autocorr: average across units of corr(h_t, h_{t+1})
      - dominant_freq_idx: argmax non-DC FFT bin averaged over units (index)
    """
    h = h_seq.detach()  # [B,T,H]
    B, T, H = h.shape
    if T < 3:
        return {"lag1_autocorr": None, "dominant_freq_idx": None}

    # center over time per (B, unit)
    x = h - h.mean(dim=1, keepdim=True)

    # lag-1 autocorr
    num = (x[:, :-1, :] * x[:, 1:, :]).sum(dim=1)
    den = (
        x[:, :-1, :].pow(2).sum(dim=1) * x[:, 1:, :].pow(2).sum(dim=1)
    ).sqrt() + 1e-12
    r1 = _nanmean_compat(num / den, dim=0)  # [H]
    lag1 = float(_nanmean_compat(r1).item())  # scalar

    # dominant frequency via 1D RFFT power (drop DC=0)
    # power: [B, F, H] along time dimension
    power = _rfft_power_time_compat(x, time_dim=1)  # helper handles old/new torch
    mag = power.mean(dim=0).mean(dim=1)  # [F], avg over batch & units
    if mag.numel() > 1:
        dom_idx = int(torch.argmax(mag[1:]).item() + 1)
    else:
        dom_idx = None

    return {"lag1_autocorr": lag1, "dominant_freq_idx": dom_idx}


def _geometry_metrics(h_seq, max_components=50):
    """
    Measure low-dimensional structure (e.g. ring manifold) via PCA.
    Low-d geometry via PCA on [B*T, H].
    Returns: participation_ratio, evr_top (list of first few EV ratios)
    """
    h = h_seq.detach()
    BT, H = h.shape[0] * h.shape[1], h.shape[2]
    X = h.reshape(BT, H) - h.reshape(BT, H).mean(dim=0, keepdim=True)
    # cov eigenvalues (use float64 for stability)
    C = (X.T @ X) / max(BT - 1, 1)
    evals = _eigvalsh_sym_compat(C.to(dtype=torch.float64)).clamp(min=0)
    s = evals.sum()
    pr = float((s**2 / (evals.pow(2).sum() + 1e-12)).item())  # participation ratio
    # explained variance ratios (first K descending)
    ev_sorted = torch.sort(evals, descending=True).values
    evr = (ev_sorted / (s + 1e-12)).tolist()
    return {"participation_ratio": pr, "evr_top": evr[: min(max_components, len(evr))]}


def _targets_to_angles(Target_stepwise):
    """
    Convert per-step target distribution over N positions to angle θ ∈ [-π, π).
    Target_stepwise: torch.Tensor [B, T, N] (not one-hot required; any nonneg weights)
    """
    B, T, N = Target_stepwise.shape
    idx = torch.arange(N, device=Target_stepwise.device)
    theta = 2 * math.pi * idx / N  # [N]
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    # vector mean on circle (per B,T)
    w = Target_stepwise / (Target_stepwise.sum(dim=2, keepdim=True) + 1e-12)
    C = (w * cos_th).sum(dim=2)
    S = (w * sin_th).sum(dim=2)
    ang = torch.atan2(S, C)  # [-pi, pi]
    return ang  # [B, T]


def _decode_ring_linear(h_seq, Target_stepwise):
    """
    Test how well hidden states linearly encode ring position
    Simple linear decode: h -> [cos θ, sin θ], returns R^2 mean of both heads.
    h_seq: [B, T, H], Target_stepwise: [B, T, N]
    """
    ang = _targets_to_angles(Target_stepwise)  # [B, T]
    y = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)  # [B, T, 2]
    X = h_seq.reshape(-1, h_seq.shape[2])
    Y = y.reshape(-1, 2)
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    # closed-form linear reg: W = (X^T X)^-1 X^T Y  (use ridge λ small)
    lam = 1e-6
    XtX = Xc.T @ Xc + lam * torch.eye(Xc.shape[1], device=X.device, dtype=X.dtype)
    W = _solve_spd_compat(XtX, Xc.T @ Yc, ridge=1e-6)  # [H,2]
    Yhat = Xc @ W + Y.mean(dim=0, keepdim=True)
    # R^2 per head
    ss_res = ((Y - Yhat) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0) + 1e-12
    r2 = (1.0 - ss_res / ss_tot).mean().item()
    return {"ring_decode_R2": float(r2)}


def _weight_structure_metrics(net, W0=None):
    """
    Summarize symmetry/asymmetry and non-normality of W_hh.
    If W0 is provided (numpy array), also report Frobenius drift and relative drift.
    """
    W = net.rnn.weight_hh_l0.detach().float().cpu().numpy()
    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)

    def fro(x):
        return float(np.linalg.norm(x, "fro"))

    nW, nS, nA = fro(W), fro(S), fro(A)
    sym_ratio = nS / (nW + 1e-12)
    asym_ratio = nA / (nW + 1e-12)
    mix = nA / (nS + 1e-12)
    # non-normality: || W W^T - W^T W ||_F
    comm = W @ W.T - W.T @ W
    nnorm = float(np.linalg.norm(comm, "fro"))

    out = {
        "fro_W": nW,
        "fro_S": nS,
        "fro_A": nA,
        "sym_ratio": sym_ratio,
        "asym_ratio": asym_ratio,
        "mix_A_over_S": mix,
        "non_normality_commutator": nnorm,
    }
    if W0 is not None:
        d = W - W0
        drift = float(np.linalg.norm(d, "fro"))
        out["fro_drift_W_minus_W0"] = drift
        out["rel_drift_W_minus_W0"] = drift / (float(np.linalg.norm(W0, "fro")) + 1e-12)
    return out


def _angle_error_metrics(output, Target_stepwise):
    """
    Angular error concentration for circular targets.
    Uses vector-mean angle of distributions to get θ̂ and θ, then Δθ = wrap(θ̂-θ).
    Returns: resultant length R (higher = more concentrated), circular variance = 1-R.
    """
    # Ensure proper distributions (nonneg, sum=1 over features)
    pred = output.detach()
    pred = pred.clamp_min(0)
    pred = pred / (pred.sum(dim=2, keepdim=True) + 1e-12)

    # True angles from targets (already a distribution)
    theta_true = _targets_to_angles(Target_stepwise)  # [B, T]

    # Predicted angles from model outputs
    B, T, N = pred.shape
    idx = torch.arange(N, device=pred.device)
    theta = 2 * math.pi * idx / N
    cos_th, sin_th = torch.cos(theta), torch.sin(theta)

    C_pred = (pred * cos_th).sum(dim=2)
    S_pred = (pred * sin_th).sum(dim=2)
    theta_pred = torch.atan2(S_pred, C_pred)  # [B, T]

    # Δθ wrapped to [-π, π)
    dtheta = torch.atan2(
        torch.sin(theta_pred - theta_true), torch.cos(theta_pred - theta_true)
    )  # [B, T]

    # Resultant length R of the phase errors
    R = torch.sqrt((torch.cos(dtheta).mean()) ** 2 + (torch.sin(dtheta).mean()) ** 2)
    R = float(R.item())
    return {"angle_error_R": R, "angle_error_circ_var": float(1.0 - R)}


def _residual_autocorr(output, Target_stepwise):
    """
    Lag-1 autocorrelation of residual magnitudes.
    Residual per step is L2 norm of (Target - Output) over features; then compute lag-1 autocorr.
    Also returns mean residual L2 for scale context.
    """
    pred = output.detach()
    res = Target_stepwise - pred  # [B, T, N]
    rnorm = _vector_norm_compat(res, dim=2)  # [B, T]  <-- compat (no torch.linalg)

    if rnorm.shape[1] < 3:
        return {
            "residual_lag1_autocorr": None,
            "residual_L2_mean": float(rnorm.mean().item()),
        }

    # center over time per sequence
    x = rnorm - rnorm.mean(dim=1, keepdim=True)  # [B, T]

    # lag-1 autocorr per sequence, then average across batch
    num = (x[:, :-1] * x[:, 1:]).sum(dim=1)  # [B]
    den = ((x[:, :-1] ** 2).sum(dim=1) * (x[:, 1:] ** 2).sum(dim=1)).sqrt() + 1e-12
    r1 = num / den  # [B]

    # if you have _nanmean_compat already, this is slightly safer:
    # lag1 = float(_nanmean_compat(r1).item())
    lag1 = float(r1.mean().item())

    return {
        "residual_lag1_autocorr": lag1,
        "residual_L2_mean": float(rnorm.mean().item()),
    }


def _vector_norm_compat(x: torch.Tensor, dim=None, keepdim=False, eps=0.0):
    # L2 norm without torch.linalg
    if dim is None:
        return torch.sqrt(torch.clamp((x * x).sum(), min=eps))
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def _nanmean_compat(x: torch.Tensor, dim=None, keepdim=False):
    """Mean that ignores NaN/Inf; supports dim=None or an int/tuple."""
    mask = torch.isfinite(x)
    x_safe = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))
    if dim is None:
        summed = x_safe.sum()
        count = mask.sum()
    else:
        summed = x_safe.sum(dim=dim, keepdim=keepdim)
        count = mask.sum(dim=dim, keepdim=keepdim)
    return summed / count.clamp(min=1)


def _rfft_power_time_compat(x: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """
    Returns the power spectrum |RFFT|^2 along the time dimension.
    - Accepts x shaped [..., T, ...] (e.g., [B, T, H] with time_dim=1).
    - Returns a real tensor with frequencies on the same axis position 'time_dim',
      i.e., shape is the same as x but with T replaced by F (onesided rfft length).
    """
    # Move time_dim to a known position for older torch.rfft or numpy
    x_moved = x.transpose(time_dim, -1)  # now last dim is time
    dev = x.device
    dtype = x.dtype

    # Path 1: modern torch.fft (preferred)
    if hasattr(torch, "fft") and hasattr(torch.fft, "rfft"):
        X = torch.fft.rfft(x_moved, dim=-1)  # complex
        power = X.real * X.real + X.imag * X.imag
        power = power.transpose(-1, time_dim)  # put freq axis back at time_dim
        return power

    # Path 2: old torch.rfft (returns last-dim=2 real-imag)
    if hasattr(torch, "rfft"):
        # torch.rfft expects transform along the last dim already
        Xri = torch.rfft(x_moved, 1, normalized=False, onesided=True)  # [..., F, 2]
        power = Xri[..., 0] ** 2 + Xri[..., 1] ** 2  # [..., F]
        power = power.transpose(-1, time_dim)  # freq axis back
        return power

    # Path 3: NumPy fallback (CPU)
    x_np = x_moved.detach().cpu().numpy()
    X_np = np.fft.rfft(x_np, axis=-1)  # complex
    power_np = (X_np.real**2 + X_np.imag**2).astype(np.float32)
    power = torch.from_numpy(power_np).to(dev)
    power = power.transpose(-1, time_dim)
    if power.dtype != dtype and dtype in (torch.float32, torch.float64):
        power = power.to(dtype)
    return power


def _eigvalsh_sym_compat(C: torch.Tensor) -> torch.Tensor:
    """
    Return eigenvalues of a real symmetric matrix C (float64 recommended).
    Tries torch.linalg.eigh, then torch.symeig, then NumPy.
    """
    # enforce symmetry numerically
    Cs = 0.5 * (C + C.transpose(-1, -2))
    # Newer Torch path
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        # returns eigenvalues,eigenvectors; we only need eigenvalues
        evals = torch.linalg.eigh(Cs, UPLO="U").eigenvalues
        return evals
    # Older Torch path
    if hasattr(torch, "symeig"):  # deprecated in new Torch but exists in old
        evals, _ = torch.symeig(Cs, eigenvectors=False, upper=True)
        return evals
    # NumPy fallback
    vals = np.linalg.eigvalsh(Cs.detach().cpu().numpy()).astype(np.float64)
    evals = torch.from_numpy(vals).to(device=C.device, dtype=C.dtype)
    return evals


def _solve_spd_compat(A: torch.Tensor, B: torch.Tensor, ridge: float = 1e-6):
    """
    Solve (A + ridge*I) X = B for X.
    Works on old/new Torch; adds small Tikhonov for stability.
    Assumes A is (approx) symmetric positive semidefinite (like X^T X).
    """
    I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    A_reg = A + ridge * I
    # Newer torch path
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        return torch.linalg.solve(A_reg, B)
    # Older torch: try inverse, then pinverse
    try:
        return torch.inverse(A_reg) @ B
    except Exception:
        return torch.pinverse(A_reg) @ B


def _run_dir_for(base_savename: str, run_idx: int) -> str:
    """
    base_savename: the computed savename (no extension),
                   e.g. runs/.../shiftmh_n100_fro
    returns: runs/.../shiftmh_n100_fro/run_00
    """
    base_dir = os.path.dirname(base_savename)
    base_stub = os.path.basename(base_savename)
    return os.path.join(base_dir, base_stub, f"run_{run_idx:02d}")


def _next_free_run_idx(base_savename: str) -> int:
    """
    Scan base folder for existing run_XX directories and return the next free index.
    Robust to gaps (will return max+1).
    """
    parent = os.path.join(
        os.path.dirname(base_savename), os.path.basename(base_savename)
    )
    if not os.path.isdir(parent):
        return 0
    pat = re.compile(r"^run_(\d+)$")
    seen = []
    for name in os.listdir(parent):
        m = pat.match(name)
        if m:
            try:
                seen.append(int(m.group(1)))
            except ValueError:
                pass
    return (max(seen) + 1) if seen else 0


def _ensure_unique_run_dir(base_savename: str, desired_idx: int) -> (str, int):
    """
    Try desired_idx; if it exists, bump until a free run_XX is found.
    Returns (run_dir, final_idx).
    """
    idx = max(0, int(desired_idx))
    while True:
        run_dir = _run_dir_for(base_savename, idx)
        try:
            os.makedirs(run_dir, exist_ok=False)  # fail if exists
            return run_dir, idx
        except FileExistsError:
            idx += 1


def log(*args, sep=" ", end="\n"):
    """Print to run log if open; otherwise to stdout."""
    global f
    msg = sep.join(str(a) for a in args) + end
    if f is not None:
        f.write(msg)
        f.flush()
    else:
        # fallback to stdout if no run log open yet
        print(msg, end="")


if __name__ == "__main__":
    main()
