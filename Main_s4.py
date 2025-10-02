"""
Mini-batch based training:
    Default AE
    For Localization test
    For remap test
"""

import argparse, sys, os, time
import random, json, platform
from datetime import datetime
from tqdm import tqdm
import numpy as np
from numpy.linalg import eig
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN_Class import *


parser = argparse.ArgumentParser(description="PyTorch BPTT Training")
parser.add_argument(
    "--epochs",
    default=50000,
    type=int,
    help="number of total epochs to run (default:50k)",
)
parser.add_argument(
    "--batch-size", default=200, type=int, help="mini-batch size (default: 200)"
)
parser.add_argument(
    "-a", "--adam", default=0, type=int, help="whether to use ADAM or not (default: 0)"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate (default:0.01)",
)
parser.add_argument("--lr_step", default="", type=str, help="decreasing strategy")
parser.add_argument(
    "-p",
    "--print-freq",
    default=1000,
    type=int,
    metavar="N",
    help="print frequency (default: 1000)",
)
parser.add_argument(
    "-g",
    "--gpu",
    default=1,
    type=int,
    help="whether enable GPU computing (default:1), if more than one GPU, number indicates GPU device",
)
parser.add_argument(
    "--hidden-n", default=200, type=int, help="Hidden dimension size(default:200)"
)
parser.add_argument(
    "--savename", default="net", type=str, help="Default output saving name"
)
parser.add_argument(
    "--output_dir",
    default="Elman_SGD/Remap_predloss/N100T100",
    type=str,
    help="Directory to save training information to",
)
parser.add_argument(
    "--partial",
    default=0,
    type=float,
    help="sparsity level (0-1) amount of the partially trained parameter",
)
parser.add_argument(
    "--sparsity",
    default=0,
    type=float,
    help="Percentage of active cells in the input layer",
)
parser.add_argument(
    "--input", default="", type=str, help="Load in user defined input sequence"
)
parser.add_argument(
    "--ae", default=0, type=int, help="whether target = input (default=0)"
)
parser.add_argument(
    "--net",
    default="ConvRNN",
    type=str,
    help="Name of the network class (default: ConvRNN)",
)
parser.add_argument(
    "--act", default="sigmoid", type=str, help="Activation of output (default: sigmoid)"
)
parser.add_argument(
    "--rnn_act", default="", type=str, help="Activation of rnn units (default: tanh)"
)
parser.add_argument(
    "--pred", default=0, type=int, help="Prediction horizon (default:0)"
)
parser.add_argument(
    "--resume", default="", type=str, help="path to latest checkpoint (default: none)"
)
parser.add_argument(
    "--Hregularized_l1",
    default=0,
    type=float,
    help="penalty for hidden unit firing (l1 regularized)",
)
parser.add_argument(
    "--Hregularized",
    default=0,
    type=float,
    help="penalty for hidden unit firing (l2 regularized)",
)
parser.add_argument("--fixi", default=0, type=int, help="whether fix the input matrix")
parser.add_argument(
    "--clip", default=0, type=float, help="whether to clip gradient or not"
)
parser.add_argument(
    "--hidden_init",
    type=str,
    default=None,
    help="Path to a .npy file containing a hidden weight matrix. If None, use default PyTorch initialization (He)",
)
parser.add_argument(
    "--early_stop", default=0, type=int, help="whether to include early stopping"
)
parser.add_argument("--seed", type=int, default=1337, help="Global RNG seed")
parser.add_argument(
    "--deterministic",
    type=int,
    default=1,
    help="Deterministic CUDA/cuDNN (may be slower)",
)
parser.add_argument(
    "--run_tag", type=str, default="", help="Optional run tag for bookkeeping"
)
parser.add_argument(
    "--whh_type",
    default="baseline",
    choices=[
        "baseline",
        "centcycmh",
        "centcyctridiag",
        "centmh",
        "centtridiag",
        "cycshift",
        "identity",
        "shift",
        "shiftcycmh",
        "shiftcyctridiag",
        "shiftmh",
        "shifttridiag",
    ],
    help="Hidden-weight init type (default: baseline)",
)
parser.add_argument(
    "--whh_norm",
    default="none",
    choices=["frobenius", "spectral", "variance", "none"],
    help="Normalization strategy for hidden weight (ignored for baseline).",
)


def main():
    global args

    args = parser.parse_args()
    args.hidden_init = _resolve_hidden_init_path(args)
    print(f"[whh] type={args.whh_type} norm={args.whh_norm} -> {args.hidden_init}")
    lr = args.lr
    n_epochs = args.epochs
    RecordEp = args.print_freq
    set_seed(args.seed, bool(args.deterministic))

    global f
    savedir = args.savename.split("/")[:-1]
    savedir = "/".join(savedir)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)
    f = open(args.savename + ".txt", "w")
    print("Settings:", file=f)
    print(str(sys.argv), file=f)

    # add metadata
    meta = {
        "argv": sys.argv,
        "args": vars(args),
        "env": env_report(),
    }
    print("META:", file=f)
    print(json.dumps(meta, indent=2), file=f)

    if args.input:
        loaded = torch.load(args.input)
        X = loaded["X_mini"]
        Y = loaded["Target_mini"]
    if args.ae:
        print("Target=Input", file=f)
        Y = loaded["X_mini"]
    if args.pred:
        X = X[
            :,
            : -args.pred,
        ]
        Y = Y[
            :,
            args.pred :,
        ]

    ##  define network module
    D = X.shape[-1]
    net = eval(args.net)(D, args.hidden_n, D)
    if args.act == "sigmoid":
        net.act = nn.Sigmoid()
        print("output nonlinearity: elementwise sigmoid", file=f)
    elif args.act == "tanh":
        net.act = nn.Tanh()
        print("output nonlinearity: elementwise tanh", file=f)

    if args.rnn_act == "sigmoid":
        net.tanh = nn.Sigmoid()
        print("RNN nonlinearity: elementwise sigmoid", file=f)
    elif args.rnn_act == "relu":
        net.tanh = nn.ReLU()
        print("RNN nonlinearity: elementwise relu", file=f)
    else:
        print("RNN nonlinearity: elementwise tanh", file=f)

    if args.fixi:
        for name, p in net.named_parameters():
            if name == "input_linear.weight" or name == "rnn.weight_ih_l0":
                p.requires_grad = False
                p.data.fill_(0)
                p.data.fill_diagonal_(1)
                print("Fixing input matrix to identity matrix", file=f)
            elif name == "input_linear.bias" or name == "rnn.bias_ih_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing input bias to 0", file=f)

    h0 = torch.zeros((1, args.batch_size, args.hidden_n))

    ## MSE criteria
    criterion = nn.MSELoss(reduction="mean")

    ##  load checkpoint and resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading previous network '{}'".format(args.resume), file=f)
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=f)

    ## enable GPU computing
    if args.gpu == 1:
        print("Cuda device availability: {}".format(torch.cuda.is_available()), file=f)
        criterion = criterion.cuda()
        net = net.cuda()
        X = X.cuda()
        h0 = h0.cuda()
        Y = Y.cuda()
    elif args.gpu > 1:
        print("Cuda device availability: {}".format(torch.cuda.is_available()), file=f)
        criterion = criterion.cuda(args.gpu - 1)
        net = net.cuda(args.gpu - 1)
        X = X.cuda(args.gpu - 1)
        h0 = h0.cuda(args.gpu - 1)
        Y = Y.cuda(args.gpu - 1)

    # construct optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if args.adam:
        print("Using ADAM optimizer", file=f)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # start training or step-wise training
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hidden_init is not None:
        net = load_hidden_weight_into(
            net,
            npy_path=args.hidden_init,
            device=device,
            zero_bias=True,
            trainable=True,
        )
    else:
        print("Using standard PyTorch initialization for hidden weights")
    (
        net,
        loss_list,
        history,
        grad_list,
        metrics,
        rng_init,
        init_hidden,
    ) = train_minibatch(
        X, Y, h0, n_epochs, net, criterion, optimizer, RecordEp, args.Hregularized
    )
    end = time.time()
    deltat = end - start
    print("Total training time: {0:.1f} minuetes".format(deltat / 60), file=f)

    # basic model config
    model_cfg = {
        "arch": args.net,
        "input_dim": int(X.shape[-1]),
        "hidden_dim": int(args.hidden_n),
        "output_dim": int(Y.shape[-1]),
    }

    # seed info for closed-loop
    warmup_context = X[:, :30].detach().cpu()  # first 30 steps, adjust length if needed
    h0_eval = h0.detach().cpu()

    # save network input, state
    save_dict = {
        "state_dict": net.state_dict(),
        "loss": loss_list,
        "history": history,
        "grad_list": grad_list,
        "metrics": metrics,
        "rng_init": rng_init,
        "init_hidden": init_hidden,
        "rollout": {
            "h0_eval": h0_eval,
            "warmup_context": warmup_context,
        },
    }
    torch.save(save_dict, args.savename + ".pth.tar")

    # save metadata
    with open(args.savename + ".meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)

    # plot loss function iteration curve
    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss iteration")
    plt.savefig(args.savename + ".png")


def total_grad_norm(params, norm_type=2.0):
    """Compute total grad norm (like clip_grad_norm_ returns, but without clipping)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)
    return norm.item()


def train_minibatch(
    X,
    Y,
    h0,
    n_epochs,
    net,
    criterion,
    optimizer,
    RecordEp,
    lamda,
    sample_batch_idx=0,
    use_clip=None,
):
    """
    INPUT:
        X: N*TimeLength*Input Dimension
        Y: (target signal)
        lamda: H-regularized term
    OUTPUT:
        y_hat: RecordN*X.shape
    """
    """
    Train a model on minibatches of data, with logging of losses, gradients,
    hidden activations, and hidden weight matrix properties.

    Args:
        X (torch.Tensor): Input tensor, shape (N, SeqLen, InputDim).
        Y (torch.Tensor): Target tensor, same batch/time structure as X.
        h0 (torch.Tensor): Initial hidden state.
        n_epochs (int): Number of training epochs.
        net (nn.Module): RNN model.
        criterion (nn.Module): Loss function (e.g. MSE).
        optimizer (torch.optim.Optimizer): Optimizer.
        RecordEp (int): Frequency (in epochs) to log/record metrics.
        lamda (float): Weight for hidden state regularization term.
        sample_batch_idx (int, optional): Which minibatch to snapshot. Default=0.
        use_clip (float, optional): Gradient clipping threshold. Default=None.
        snapshots_dir (str, optional): Directory to save hidden weight matrices.

    Returns:
        net (nn.Module): The trained network.
        loss_list (list): Per-epoch mean loss values.
        history (dict): Input/output/hidden snapshots and loss/grad_norm logs.
        grad_list (list): Gradient statistics recorded every RecordEp epochs.
        metrics (list): Hidden weight and activation statistics every RecordEp.
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss_list, grad_list = [], []
    history = {
        "epoch": [],
        "y_hat": [],  # (batch, SeqN, output_dim)
        "hidden": [],  # (batch, hidden_dim)
        "X_mini": [],  # (batch, SeqN, input_dim)
        "Target_mini": [],  # (batch, SeqN, output_dim) or target shape
        "loss": [],  # scalar (mean of batch losses for that epoch)
        "grad_norm": [],  # scalar (L2)
    }
    metrics = []  # stores hidden-weight metrics per recording epoch

    N = X.shape[0]
    criterion_l1 = nn.L1Loss(reduction="mean")
    start = time.time()
    epoch, stop = 0, 0
    prev_grad_stats = None

    # respect existing clip flag if provided externally via args
    if use_clip is None:
        try:
            use_clip = args.clip
        except NameError:
            use_clip = None

    # save initial hidden weights for "drift from init" measurement
    if isinstance(net, SymAsymRNN):
        init_hidden = net.effective_W().detach().cpu().clone()
    elif "pytorch" in args.net:
        init_hidden = net.rnn.weight_hh_l0.detach().cpu().clone()
    else:
        init_hidden = net.hidden_linear.weight.detach().cpu().clone()

    rng_init = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }

    def _safe_sample_batch_idx(requested_idx: int, n_batches: int) -> int:
        if n_batches <= 0:
            return 0
        return min(max(0, requested_idx), n_batches - 1)

    # main training loop with tqdm progress bar
    with tqdm(total=n_epochs, desc="Training", unit="epoch") as pbar:
        while stop == 0 and epoch < n_epochs:
            # optional learning-rate step decay
            if args.lr_step:
                lr_step = list(map(int, args.lr_step.split(",")))
                if epoch in lr_step:
                    print("Decrease lr to 50per at epoch {}".format(epoch), file=f)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5

            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            batch_losses = []
            grad_norm_for_record = None
            y_hat_record = h_t_record = Xmini_record = Ymini_record = None

            num_batches = int(np.ceil(N / args.batch_size))
            bs = args.batch_size

            # compute a safe snapshot batch index for this epoch
            safe_b = _safe_sample_batch_idx(sample_batch_idx, num_batches)

            for b in range(num_batches):
                start_idx = b * bs
                end_idx = min((b + 1) * bs, N)
                X_mini = X[start_idx:end_idx]
                Y_mini = Y[start_idx:end_idx]

                # forward pass
                output, h_t = net(X_mini, h0)

                # loss: prediction + hidden regularizers
                loss1 = criterion(output, Y_mini)
                loss2 = lamda * criterion(
                    h_t, torch.zeros(h_t.shape).to(X_mini.device)
                ) + args.Hregularized_l1 * criterion_l1(
                    h_t, torch.zeros(h_t.shape).to(X_mini.device)
                )
                loss = loss1 + loss2
                loss.backward()  # Does backpropagation and calculates gradients

                # compute grad norm before clipping
                gn = _total_grad_norm(net.parameters(), p=2.0)

                # record on recording epochs and first epoch from safe batch
                if ((epoch % RecordEp) == 0 or epoch == 0) and (b == safe_b):
                    prev_grad_stats = record_grads(
                        net, grad_list, prev_stats=prev_grad_stats
                    )
                    # optional: quick health log
                    print(
                        f"[grad] epoch={epoch} batch={b}/{num_batches-1} norm={float(gn):.4f}"
                    )

                    # stash aligned snapshot bits
                    grad_norm_for_record = gn
                    y_hat_record = output.detach().cpu()
                    h_t_record = h_t.detach().cpu()
                    Xmini_record = X_mini.detach().cpu()
                    Ymini_record = Y_mini.detach().cpu()

                # optional gradient clipping
                if use_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

                optimizer.step()  # Updates the weights accordingly
                optimizer.zero_grad()

                batch_losses.append(loss.item())

            # epoch-end bookkeeping
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            loss_list.append(epoch_loss)

            # tqdm: update progress bar and set loss in postfix
            pbar.set_postfix({"loss": epoch_loss})
            pbar.update(1)

            # early stopping
            if args.early_stop == 0:
                tolerance = -1.0  # impossible tolerence to remove
            else:
                tolerance = 1e-7
            if epoch > 1000 and all(np.abs(np.diff(loss_list[-10:])) < tolerance):
                stop = 1
                print("Hit the stopping criterion < 1e-7", file=f)
            if epoch % RecordEp == 0:
                # save IO/hidden snapshots for analysis
                if y_hat_record is not None:
                    history["epoch"].append(epoch)
                    history["y_hat"].append(y_hat_record)
                    history["hidden"].append(h_t_record)
                    history["X_mini"].append(Xmini_record)
                    history["Target_mini"].append(Ymini_record)
                    history["loss"].append(epoch_loss)
                    history["grad_norm"].append(grad_norm_for_record)

                # hidden weight analysis
                if isinstance(net, SymAsymRNN):
                    Wh = net.effective_W().detach().cpu()
                elif "pytorch" in args.net:
                    Wh = net.rnn.weight_hh_l0.detach().cpu()
                else:
                    Wh = net.hidden_linear.weight.detach().cpu()

                # COMMENTING OUT EXPENSIVE CALCULATIONS -- CAN DO OFFLINE
                # drift = _frob(Wh - init_hidden)
                # frob_norm = _frob(Wh)
                # rho = _spectral_radius(Wh)
                # orth_err = None

                # if Wh.shape[0] == Wh.shape[1]:
                #    I = torch.eye(Wh.shape[0])
                #    orth_err = _frob(Wh.T @ Wh - I)  # orthogonality error

                # hidden activation stats on current minibatch
                with torch.no_grad():
                    out_dbg, h_dbg = net(X_mini, h0)
                    act_mean = float(h_dbg.mean().item())
                    act_std = float(h_dbg.std().item())
                    tanh_sat = float((h_dbg.abs() > 0.99).float().mean().item())
                    # s, _, _ = torch.svd(Wh)
                    # s_max = float(s.max().item())
                    # s_min = float(s.min().item()) if s.numel() > 0 else float("nan")
                    # cond = float(s_max / (s_min + 1e-12))

                    # Hidden weight aggregates
                    # max_abs_w = float(Wh.abs().max().item())
                    # sparsity_w = float((Wh.abs() < 1e-8).float().mean().item())

                # Batch-loss dispersion within this epoch
                loss_mean = float(np.mean(batch_losses))
                loss_std = float(np.std(batch_losses))

                # save raw hidden matrix for offline analysis
                os.makedirs(args.output_dir, exist_ok=True)
                fname = os.path.join(args.output_dir, f"Wh_epoch{epoch:06d}.pt")
                torch.save(Wh, fname)

                metrics.append(
                    {
                        "epoch": epoch,
                        "loss": epoch_loss,
                        "loss_batch_mean": loss_mean,
                        "loss_batch_std": loss_std,
                        # "frob": frob_norm,
                        # "drift_from_init": drift,
                        # "spectral_radius": rho,
                        # "spectral_norm": s_max,
                        # "min_singular": s_min,
                        # "cond_num": cond,
                        # "orth_err": orth_err,
                        # "w_max_abs": max_abs_w,
                        # "w_sparsity": sparsity_w,
                        "act_mean": act_mean,
                        "act_std": act_std,
                        "tanh_sat": tanh_sat,
                    }
                )

                if isinstance(net, SymAsymRNN):
                    metrics[-1]["lambda"] = float(net.effective_lambda().detach().cpu())
                    torch.save(
                        net.S.detach().cpu(),
                        os.path.join(args.output_dir, f"S_epoch{epoch:06d}.pt"),
                    )
                    torch.save(
                        net.A.detach().cpu(),
                        os.path.join(args.output_dir, f"A_epoch{epoch:06d}.pt"),
                    )

            epoch += 1

    # sanity ping if nothing was captured
    if len(grad_list) == 0:
        print("[WARN] grad_list is EMPTY. Check RecordEp and sample_batch_idx.")

    return net, loss_list, history, grad_list, metrics, rng_init, init_hidden


def _total_grad_norm(params, p=2.0):
    """
    Compute the total gradient norm (default L2) across all parameters.

    Args:
        params (iterable): Model parameters (with .grad).
        p (float): Which norm to compute (2.0 = L2).

    Returns:
        float: Total gradient norm.
    """
    grads = [p_.grad for p_ in params if (p_.requires_grad and p_.grad is not None)]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([torch.norm(g, p) for g in grads]), p).item())


def _spectral_radius(W: torch.Tensor) -> float:
    """
    Compute spectral radius (largest absolute eigenvalue).

    Args:
        W (torch.Tensor): Square weight matrix.

    Returns:
        float: Spectral radius of W.
    """
    if W.is_complex():
        # torch.eig only supports real matrices; use NumPy for complex
        vals = np.linalg.eigvals(W.numpy())
        return float(np.abs(vals).max())
    else:
        # torch.eig returns (n,2): columns are [real, imag]
        e = torch.eig(W, eigenvectors=False)[0]
        r = torch.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2).max()
        return float(r.item())


def _frob(W: torch.Tensor) -> float:
    """
    Compute Frobenius norm of a matrix.

    Args:
        W (torch.Tensor): Input matrix.

    Returns:
        float: Frobenius norm.
    """
    return float(torch.norm(W, p="fro").item())


def record_grads(net, grad_list, prev_stats=None):
    """
    Record per-parameter gradient stats; optionally compute cosine similarity vs previous.
    Returns current stats so caller can keep for next comparison.
    """
    stats = {}
    layer_group_norms = {}  # aggregate norms per 'layer' prefix

    for name, p in net.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().flatten().cpu()
            g_np = g.numpy()
            l2 = float(torch.norm(g, p=2).item())
            mean = float(g_np.mean())
            std = float(g_np.std())
            mean_sq = float((g_np**2).mean())
            max_abs = float(np.max(np.abs(g_np)))
            sparsity = float(np.mean(np.isclose(g_np, 0.0)))

            cos = None
            if (
                prev_stats is not None
                and name in prev_stats
                and "raw" in prev_stats[name]
            ):
                g_prev = prev_stats[name]["raw"]
                # safe cosine
                denom = (np.linalg.norm(g_np) * np.linalg.norm(g_prev)) + 1e-12
                cos = float(np.dot(g_np, g_prev) / denom)

            stats[name] = {
                "mean": mean,
                "std": std,
                "l2_norm": l2,
                "mean_sq": mean_sq,
                "max_abs": max_abs,
                "sparsity": sparsity,
                "cos_prev": cos,  # can be None on first record
                "raw": g_np,  # keep raw for next cosine; will be stripped before saving
            }

            # simple grouping: take prefix before first dot
            group = name.split(".")[0]
            layer_group_norms.setdefault(group, 0.0)
            layer_group_norms[group] += l2

    # append a copy that omits raw vectors to keep checkpoints small
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in stats.items()
    }
    grad_list.append({"per_param": slim, "per_group_norm": layer_group_norms})
    return stats  # return full (with raw) to compute cosine next time


def step_YC(x):
    return (x >= 0).float()


def load_hidden_weight_into(
    model,
    npy_path: str,
    device=None,
    zero_bias: bool = True,
    trainable: bool = True,
    layer: int = 0,
):
    """
    Load an (H, H) numpy array into the model's *recurrent* hidden-to-hidden weight.

    Supports:
      - nn.RNN/nn.GRU/nn.LSTM via model.rnn.weight_hh_l{layer}
      - Fallback: model.hidden_linear.weight (for custom modules)

    Args:
        model: torch.nn.Module with .rnn (RNN/GRU/LSTM) or .hidden_linear
        npy_path: path to .npy containing an (H,H) matrix
        device: torch.device or None to infer from model
        zero_bias: if True, zero recurrent bias (and input bias if present)
        trainable: if False, freeze the loaded parameters
        layer: RNN layer index (default 0)

    Returns:
        model (mutated in-place)
    """
    if isinstance(model, SymAsymRNN):
        W = np.load(npy_path)
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"Expected square (H,H) matrix, got shape {W.shape}")
        thW = torch.as_tensor(W, dtype=torch.float32, device=device)
        with torch.no_grad():
            model.S.copy_(0.5 * (thW + thW.T))
            model.A.copy_(0.5 * (thW - thW.T))
        model.S.requires_grad = trainable
        model.A.requires_grad = trainable
        print(f"Loaded {npy_path} into SymAsymRNN (trainable={trainable})")
        return model

    W = np.load(npy_path)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Expected square (H,H) matrix, got shape {W.shape}")

    thW = torch.as_tensor(W, dtype=torch.float32)

    if device is None:
        device = next(model.parameters()).device
    thW = thW.to(device)

    # Try RNN/GRU/LSTM first
    target_weight = None
    rnn_mod = getattr(model, "rnn", None)
    if isinstance(rnn_mod, (nn.RNN, nn.GRU, nn.LSTM)):
        # names like weight_hh_l0, bias_hh_l0, weight_ih_l0, bias_ih_l0
        w_name = f"weight_hh_l{layer}"
        if not hasattr(rnn_mod, w_name):
            raise AttributeError(f"Model.rnn has no attribute {w_name}")
        target_weight = getattr(rnn_mod, w_name)

        # Shape check (GRU/LSTM have 3H or 4H rows; Elman RNN should be HxH)
        if target_weight.shape != thW.shape:
            raise ValueError(
                f"Shape mismatch for {w_name}: model {tuple(target_weight.shape)} vs file {tuple(thW.shape)}"
            )

        with torch.no_grad():
            target_weight.copy_(thW)

        # Optionally zero biases (both hh and ih if present)
        if zero_bias:
            for b_name in (f"bias_hh_l{layer}", f"bias_ih_l{layer}"):
                if hasattr(rnn_mod, b_name):
                    b = getattr(rnn_mod, b_name)
                    if b is not None:
                        b.detach().zero_()

        # Set requires_grad
        target_weight.requires_grad = bool(trainable)
        for b_name in (f"bias_hh_l{layer}", f"bias_ih_l{layer}"):
            if hasattr(rnn_mod, b_name):
                getattr(rnn_mod, b_name).requires_grad = bool(trainable)

        print(f"Loaded {npy_path} into rnn.{w_name} (trainable={trainable})")
        return model

    # Fallback: custom module with hidden_linear (use with ElmanRNN_tp1)
    hidden_linear = getattr(model, "hidden_linear", None)
    if hidden_linear is not None and hasattr(hidden_linear, "weight"):
        if hidden_linear.weight.shape != thW.shape:
            raise ValueError(
                f"Shape mismatch for hidden_linear.weight: model {tuple(hidden_linear.weight.shape)} vs file {tuple(thW.shape)}"
            )
        with torch.no_grad():
            hidden_linear.weight.copy_(thW)
            if zero_bias and hidden_linear.bias is not None:
                hidden_linear.bias.zero_()
        hidden_linear.weight.requires_grad = bool(trainable)
        if hidden_linear.bias is not None:
            hidden_linear.bias.requires_grad = bool(trainable)
        print(f"Loaded {npy_path} into hidden_linear.weight (trainable={trainable})")
        return model

    # Not target found
    raise AttributeError(
        "Could not locate a recurrent hidden weight to load into. "
        "Expected model.rnn.weight_hh_l{layer} (RNN/GRU/LSTM) or model.hidden_linear.weight."
    )


def set_seed(seed: int = 1337, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def env_report():
    info = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.cuda.is_available()
        else None,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    return info


def _resolve_hidden_init_path(args) -> str:
    # 1) explicit path wins (back-compat)
    if args.hidden_init:
        return args.hidden_init

    # 2) baseline special-case
    if args.whh_type == "baseline":
        candidates = [
            "/data/Ns100_SeqN100/hidden-weight-inits/baseline/random_baseline.npy",  # absolute
            "data/Ns100_SeqN100/hidden-weight-inits/baseline/random_baseline.npy",  # relative
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Baseline file not found at any expected location.")

    # 3) constructed path for non-baseline
    norm = (args.whh_norm or "none").lower()
    short = {
        "frobenius": "fro",
        "spectral": "spec_gain0p90",
        "variance": "var",
        "none": "raw",
    }[norm]
    folder = norm if norm != "none" else "raw"

    root = "data/Ns100_SeqN100/hidden-weight-inits"
    fname = f"{args.whh_type}_n{args.hidden_n}_{short}.npy"
    path = os.path.join(root, args.whh_type, folder, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Hidden-weight init not found: {path}")
    return path


if __name__ == "__main__":
    main()
