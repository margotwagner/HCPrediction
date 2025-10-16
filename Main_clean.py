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
"""
import argparse
import sys
import os
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
    default="identity",
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


def main():
    """Entry point: parse args, build data/model/optimizer, train, and save artifacts"""
    global args

    args = parser.parse_args()
    lr = args.lr
    n_epochs = args.epochs
    N = args.n
    hidden_N = args.hidden_n
    set_seed(args.seed)

    # ---------------------
    # Open a simple log file
    # ---------------------
    global f
    f = open(args.savename + ".txt", "w") if args.savename else open("train.log", "w")
    print("Settings:", file=f)
    print(str(sys.argv), file=f)

    # -----------------
    # Load training data
    # -----------------
    # Expected keys in saved file: 'X_mini' (inputs), 'Target_mini' (targets)
    loaded = torch.load(args.input)
    X_mini = loaded["X_mini"]
    Target_mini = loaded["Target_mini"]

    # Autoencoder mode: predict input itself (testing)
    if args.ae:
        print("Autoencoder scenario: Target = Input", file=f)
        Target_mini = loaded["X_mini"]

    # One-step prediction: align X and Target for next-step forecasting
    if args.pred:
        X_mini = X_mini[:, :-1, :]
        Target_mini = Target_mini[:, 1:, :]
        print("Predicting one-step ahead", file=f)

    # ---------------------
    # Define the network
    # ---------------------
    # ElmanRNN_pytorch_module_v2: (input_dim=N, hidden_dim=hidden_N, output_dim=N)
    net = ElmanRNN_pytorch_module_v2(N, hidden_N, N)

    # --- override hidden weight from disk ---
    if args.whh_type != "none":
        try:
            path = _resolve_hidden_path(
                hidden_N, args.whh_type, args.whh_norm, args.alpha
            )
            print(f"[whh] loading hidden weight from: {path}", file=f)
            _load_hidden_into_elman(net, path, device=net.rnn.weight_hh_l0.device)
        except Exception as e:
            print(f"[whh] WARNING: failed to load init: {e}", file=f)
    else:
        print("[whh] using default random init", file=f)

    # Cache initial recurrent matrix (after any override)
    W0 = net.rnn.weight_hh_l0.detach().cpu().numpy()

    # Auto-build savename if not provided, mirroring init structure
    if not args.savename:
        if args.whh_type == "none":
            # No override: keep it simple but structured
            out_dir = os.path.join(args.out_root, "ElmanRNN", "random-init")
            prefix = _prefix_from_type("none")  # "random"
            norm_short = "pytorch"
            fname_bits = [f"{prefix}_n{hidden_N}_{norm_short}"]
        else:
            variant = _resolve_variant(args.whh_type)
            norm_dir = args.whh_norm
            out_dir = os.path.join(
                args.out_root, "ElmanRNN", variant, args.whh_type, norm_dir
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

    if args.run_tag:
        fname_bits.append(str(args.run_tag))
    # make folder and set savename base (no extension)
    os.makedirs(out_dir, exist_ok=True)
    args.savename = os.path.join(out_dir, "_".join(fname_bits))

    # Optionally change the RNN hidden nonlinearity
    if args.rnn_act == "relu":
        net.rnn = nn.RNN(N, hidden_N, 1, batch_first=True, nonlinearity="relu")
        print("RNN nonlinearity: elementwise relu", file=f)

    # Optionally override output activation
    if args.ac_output == "tanh":
        net.act = nn.Tanh()
        print("Change output activation function to tanh", file=f)
    elif args.ac_output == "relu":
        net.act = nn.ReLU()
        print("Change output activation function to relu", file=f)
    elif args.ac_output == "sigmoid":
        net.act = nn.Sigmoid()
        print("Change output activation function to sigmoid", file=f)

    # ------------------------------
    # Optional parameter constraints
    # ------------------------------
    # Remove hidden bias entirely (zero + freeze)
    if args.nobias:
        for name, p in net.named_parameters():
            if name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing RNN bias to 0", file=f)

    # Fix input matrix with different modes and freeze it
    if args.fixi:
        for name, p in net.named_parameters():
            if name == "rnn.weight_ih_l0":
                if args.fixi == 1:
                    # Positive constant matrix (uniform average)
                    p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                    print("Fixing {} to positive constant".format(name), file=f)
                elif args.fixi == 2:
                    # Preserve initialization but freeze it
                    print("Fixing {} to initialization".format(name), file=f)
                elif args.fixi == 3:
                    # Make current init nonegative by folding absolute values
                    p.data = p.data + torch.abs(p.data)
                    print("Fixing {} to positive initiation".format(name), file=f)

                p.requires_grad = False

    # Fix output matrix with different modes and freeze it
    if args.fixo:
        for name, p in net.named_parameters():
            if name == "linear.weight":
                if args.fixo == 1:
                    p.data = torch.ones(p.shape) / (p.shape[0] * p.shape[1])
                    print("Fixing {} to positive constant".format(name), file=f)
                elif args.fixo == 2:
                    print("Fixing {} to initialization".format(name), file=f)
                elif args.fixo == 3:
                    p.data = p.data + torch.abs(p.data)
                    print("Fixing {} to positive initiation".format(name), file=f)
                p.requires_grad = False

    # Fix recurrent weight and its bias (freeze both)
    if args.fixw:
        for name, p in net.named_parameters():
            if name == "rnn.weight_hh_l0":
                p.requires_grad = False
                # Random in [-1/sqrt(N), 1/sqrt(N)] (Elman-ish scale)
                p.data = torch.rand(p.data.shape) * 2 * 1 / np.sqrt(N) - 1 / np.sqrt(N)
                print("Fixing recurrent matrix to a random matrix", file=f)
            elif name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing input bias to 0", file=f)

    # ------------------
    # Loss & checkpoint
    # ------------------
    criterion = nn.MSELoss(reduction="mean")  # mean MSE across batch/time/features

    # Optionally resume model weights from a checkpoint(.pth.tar)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading previous network '{}'".format(args.resume), file=f)
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint["state_dict"])
            print("=> loaded previous network '{}' ".format(args.resume), file=f)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=f)

    # -----------------------
    # Initial hidden state h0
    # -----------------------
    # Shape: (num_layers=1, batch_size, hidden_N)
    h0 = torch.zeros(1, X_mini.shape[0], hidden_N)  # n_layers * BatchN * NHidden

    # --------------
    # Move to device
    # --------------
    if args.gpu:
        print("Cuda device availability: {}".format(torch.cuda.is_available()), file=f)
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
        # 'partial' is the proportion of parameters to be frozen (0-1) (untrainable)
        print("Training sparsity:{}".format(args.partial), file=f)
        # Masks for recurrent weight and bias (same shapes as parameters)
        Mask_W = np.random.uniform(0, 1, (hidden_N, hidden_N))
        Mask_B = np.random.uniform(0, 1, (hidden_N))
        # True == UNTRAINED connections (frozen)
        Mask_W = Mask_W > args.partial
        Mask_B = Mask_B > args.partial

        # Optional non-overlap with a prior mask if resuming (kept from original)
        if hasattr(args, "nonoverlap") and args.nonoverlap:
            Mask_W = ~(~(Mask_W) & checkpoint["Mask_W"])
            Mask_B = ~(~(Mask_B) & checkpoint["Mask_B"])
        # Pack masks in order of net.parameters()
        Mask = []
        for name, p in net.named_parameters():
            if name == "rnn.weight_hh_l0" or name == "hidden_linear.weight":
                Mask.append(Mask_W)
                print("Partially train RNN weight", file=f)
            elif name == "rnn.bias_hh_l0" or name == "hidden_linear.bias":
                Mask.append(Mask_B)
                print("Partially train RNN bias", file=f)
            else:
                Mask.append(np.zeros(p.shape))  # no masking for other params
    else:
        # No partial training: masks are all zeros (no freezing beyond fix flags)
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
    ) = train_partial(
        X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask, W0
    )
    end = time.time()
    deltat = end - start
    print("Total training time: {0:.1f} minuetes".format(deltat / 60), file=f)

    # -----------------
    # Plot loss curve
    # -----------------
    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss iteration")
    plt.savefig(args.savename + ".png")

    # -----------------------------------
    # Save checkpoint + minimal artifacts
    # -----------------------------------
    save_dict = {
        "state_dict": net.state_dict(),
        "y_hat": np.array(y_hat),  # periodic output snapshots
        "hidden": np.array(hidden),  # periodic hidden state snapshots
        "X_mini": X_mini.cpu(),  # inputs (for analysis)
        "Target_mini": Target_mini.cpu(),  # targets (for analysis)
        "loss": loss_list,  # scalar training loss per epoch
        "mean_squared_grads": grad_list,  # simple grad stats list per epoch
        "grad_metrics": {
            "history": grad_metrics_history,
            # Quick series for easy plotting without unpacking:
            "global_L2_pre": [h["pre"]["global"]["L2"] for h in grad_metrics_history],
            "global_L2_post": [h["post"]["global"]["L2"] for h in grad_metrics_history],
            "global_RMS_pre": [h["pre"]["global"]["RMS"] for h in grad_metrics_history],
            "global_RMS_post": [
                h["post"]["global"]["RMS"] for h in grad_metrics_history
            ],
            # Parameter names & shapes (take from first epoch that had grads):
            "param_names": list(grad_metrics_history[0]["post"]["groups"].keys())
            if len(grad_metrics_history) > 0
            else [],
            "param_shapes": {
                k: v["shape"]
                for k, v in (
                    grad_metrics_history[0]["post"]["groups"].items()
                    if len(grad_metrics_history) > 0
                    else {}
                ).items()
            },
        },
        "hidden_metrics": {
            "history": hidden_metrics_history,
        },
        "weight_structure": {
            "history": weight_structure_history,
        },
        "weights": {
            "W_hh_init": W0.astype(np.float16),  # NEW: initial W (compact)
            "W_hh_history": np.array(
                W_hh_history, dtype=np.float16
            ),  # [num_snaps, H, H]
        },
        "n_epochs": int(n_epochs),
        "args": vars(args),  #  (CLI config)
        "env": env_report(),  #  (versions/flags)
        "rng": rng_snapshot(),  #  (resume deterministically)
        "snapshot_epochs": snapshot_epochs,  # epochs at which hidden/output were recorded
    }
    if args.partial:
        save_dict["Mask_W"] = Mask_W
        save_dict["Mask_B"] = Mask_B
    torch.save(save_dict, args.savename + ".pth.tar")


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

    with tqdm(total=n_epochs, desc="Progress", unit="epoch") as pbar:
        for epoch in range(n_epochs):
            if stop:
                break
            # Optional LR schedule: halve LR at specified epochs
            if args.lr_step:
                lr_step = list(map(int, args.lr_step.split(",")))
                if epoch in lr_step:
                    print("Decrease lr to 50per at epoch {}".format(epoch), file=f)
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
                    p.grad.data[torch.from_numpy(Mask[l].astype(bool))] = 0

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
                    if name == "linear_weight":
                        p.data.clamp_(0)

            # Simple early stoppping: requires >1000; checks small loss change and low absolute loss
            if epoch > 1000:
                diff = [
                    loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)
                ]
                mean_diff = np.mean(abs(np.array(diff[-5:-1])))
                init_loss = np.mean(np.array(loss_list[0]))
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


def _resolve_hidden_path(
    hidden_N: int,
    whh_type: str,
    whh_norm: str,
    alpha: Optional[float] = None,
) -> str:
    variant = _resolve_variant(whh_type)
    norm_dir = whh_norm  # directory name is full strategy
    base = f"./data/Ns100_SeqN100/hidden-weight-inits/ElmanRNN/{variant}/{whh_type}/{norm_dir}"
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
    fname = f"{prefix}_n{hidden_N}_{norm_short}.npy"
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
    energy = float(torch.linalg.vector_norm(h2, dim=1).mean().item())
    return {"mean": mean, "std": std, "sat_ratio": sat, "energy_L2_mean": energy}


def _temporal_metrics(h_seq):
    """
    Quantify simple temporal metrics from hidden state sequence.
    Returns simple temporal structure readouts:
    - lag1_autocorr: average across units of corr(h_t, h_{t+1})
    - dominant_freq_idx: argmax non-DC FFT bin averaged over units (index)
    """
    h = h_seq.detach()  # [B,T,H]
    B, T, H = h.shape
    if T < 3:
        return {"lag1_autocorr": None, "dominant_freq_idx": None}
    # center over time per (B,unit)
    x = h - h.mean(dim=1, keepdim=True)
    # lag-1 autocorr
    num = (x[:, :-1, :] * x[:, 1:, :]).sum(dim=1)
    den = (
        x[:, :-1, :].pow(2).sum(dim=1) * x[:, 1:, :].pow(2).sum(dim=1)
    ).sqrt() + 1e-12
    r1 = (num / den).nanmean(dim=0)  # [H]
    lag1 = float(r1.nanmean().item())
    # crude dominant frequency via FFT magnitude (drop DC=0)
    # average across batch and units
    X = torch.fft.rfft(x, dim=1)  # [B, F, H]
    mag = X.abs().mean(dim=0).mean(dim=1)  # [F]
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
    evals = torch.linalg.eigvalsh(C.to(dtype=torch.float64)).clamp(min=0)
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
    W = torch.linalg.solve(XtX, Xc.T @ Yc)  # [H,2]
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


if __name__ == "__main__":
    main()
