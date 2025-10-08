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
                    # Zero-symmetric init
                    print("Fixing {} to zero symmetric initiation".format(name), file=f)
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
    net, loss_list, grad_list, hidden, y_hat, snapshot_epochs = train_partial(
        X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask
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
        "grad_norm": grad_list,  # simple grad stats list per epoch
        "n_epochs": int(n_epochs),  # NEW
        "args": vars(args),  # NEW (CLI config)
        "env": env_report(),  # NEW (versions/flags)
        "rng": rng_snapshot(),  # NEW (resume deterministically)
        "snapshot_epochs": snapshot_epochs,  # epochs at which hidden/output were recorded
    }
    if args.partial:
        save_dict["Mask_W"] = Mask_W
        save_dict["Mask_B"] = Mask_B
    torch.save(save_dict, args.savename + ".pth.tar")


# -------------------------------------------------
# Training loop with optional partial-parameter masks
# -------------------------------------------------
def train_partial(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask):
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
    grad_list = []  # per-epoch gradient stats
    snapshot_epochs = []  # epochs at which snapshots were taken

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

            # Record simple grad statistics (mean of squared gradients per param)
            tmp = []
            for name, p in net.named_parameters():
                if p.requires_grad:
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
                print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
                print("Loss: {:.4f}".format(loss.item()))
                print("Time Elapsed since last display: {0:.1f} seconds".format(deltat))
                print(
                    "Estimated remaining time: {0:.1f} minutes".format(
                        deltat * (n_epochs - epoch) / args.print_freq / 60
                    )
                )

    return net, loss_list, grad_list, hidden_rep, output_rep, snapshot_epochs


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
    else:  # {"identity","shift","shift-cyc"}
        return "shift-variants"


def _prefix_from_type(whh_type: str) -> str:
    mapping = {
        "cent": "centmh",
        "cent-cyc": "centcycmh",
        "shifted": "shiftmh",
        "shifted-cyc": "shiftcycmh",
        "identity": "identity",
        "shift": "shift",
        "shift-cyc": "shiftcyc",
    }
    return mapping[whh_type]


def _norm_shortname(norm: str) -> str:
    return {"frobenius": "fro", "raw": "raw"}[norm]


def _resolve_hidden_path(
    hidden_N: int, whh_type: str, whh_norm: str, alpha: float | None
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
    net: nn.Module, npy_path: str, device: torch.device | None = None
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


if __name__ == "__main__":
    main()
