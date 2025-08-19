"""
Mini-batch based training:
    Default AE
    For Localization test
    For remap test
"""

import argparse
import sys
import os
import shutil
import time
import numpy as np
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
    "--output-dir",
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


def main():
    global args

    args = parser.parse_args()
    lr = args.lr
    n_epochs = args.epochs
    RecordEp = args.print_freq

    global f
    f = open(args.savename + ".txt", "w")
    print("Settings:", file=f)
    print(str(sys.argv), file=f)

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
        print("output nonlinearity: elementwise sigmoid", file=f)

    if args.rnn_act == "sigmoid":
        net.tanh = nn.Sigmoid()
        print("RNN nonlinearity: elementwise sigmoid", file=f)
    elif args.rnn_act == "relu":
        net.tanh = nn.ReLU()
        print("RNN nonlinearity: elementwise relu", file=f)

    if args.fixi:
        for name, p in net.named_parameters():
            if name == "input_linear.weight":
                p.requires_grad = False
                p.data.fill_(0)
                p.data.fill_diagonal_(1)
                print("Fixing input matrix to identity matrix", file=f)
            elif name == "input_linear.bias":
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
    net, loss_list, history, grad_list, metrics = train_minibatch(
        X, Y, h0, n_epochs, net, criterion, optimizer, RecordEp, args.Hregularized
    )
    end = time.time()
    deltat = end - start
    print("Total training time: {0:.1f} minuetes".format(deltat / 60), file=f)

    # save network input, state
    save_dict = {
        "state_dict": net.state_dict(),
        "loss": loss_list,
        "history": history,
        "grad_list": grad_list,
        "metrics": metrics,
    }
    torch.save(save_dict, args.savename + ".pth.tar")

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

    # respect existing clip flag if provided externally via args
    if use_clip is None:
        try:
            use_clip = args.clip
        except NameError:
            use_clip = None

    # save initial hidden weights for "drift from init" measurement
    init_hidden = net.hidden_linear.weight.detach().cpu().clone()

    # main training loop
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

            # record grad for recording epoch on chosen minibatch
            if (epoch % RecordEp == 0) and (b == sample_batch_idx):
                record_grads(net, grad_list)

            # compute grad norm before clipping
            gn = _total_grad_norm(net.parameters(), p=2.0)

            # optional gradient clipping
            if use_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            optimizer.step()  # Updates the weights accordingly
            optimizer.zero_grad(set_to_none=True)

            batch_losses.append(loss.item())

            # save minibatch snapshot (IO, hidden, grad_norm)
            if b == sample_batch_idx:
                grad_norm_for_record = gn
                y_hat_record = output.detach().cpu()
                h_t_record = h_t.detach().cpu()
                Xmini_record = X_mini.detach().cpu()
                Ymini_record = Ymini_record.detach().cpu()

        # epoch-end bookkeeping
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        loss_list.append(epoch_loss)

        # early stopping
        if epoch > 1000 and all(np.abs(np.diff(loss_list[-10:])) < 1e-7):
            stop = 1
            print("Hit the stopping criterion < 1e-7", file=f)
        if epoch % RecordEp == 0:
            end = time.time()
            deltat = end - start
            start = time.time()
            print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
            print("Loss: {:.4f}".format(loss.item()))
            print("Time Elapsed since last display: {0:.1f} seconds".format(deltat))
            print(
                "Estimated remaining time: {0:.1f} minutes".format(
                    deltat * (n_epochs - epoch) / RecordEp / 60
                )
            )

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
            Wh = net.hidden_linear.weight.detach().cpu()
            drift = _frob(Wh - init_hidden)
            frob_norm = _frob(Wh)
            rho = _spectral_radius(Wh)
            orth_err = None

            if Wh.shape[0] == Wh.shape[1]:
                I = torch.eye(Wh.shape[0])
                orth_err = _frob(Wh.T @ Wh - I)  # orthogonality error

            # hidden activation stats on current minibatch
            with torch.no_grad():
                out_dbg, h_dbg = net(X_mini, h0)
                act_mean = float(h_dbg.mean().item())
                act_std = float(h_dbg.std().item())
                tanh_sat = float((h_dbg.abs() > 0.99).float().mean().item())

            # save raw hidden matrix for offline analysis
            torch.save(Wh, os.path.join(args.output_dir), f"Wh_epoch{epoch:06d}.pt")

            metrics.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "frob": frob_norm,
                    "drift_from_init": drift,
                    "spectral_radius": rho,
                    "orth_err": orth_err,
                    "act_mean": act_mean,
                    "act_std": act_std,
                    "tanh_sat": tanh_sat,
                }
            )
        epoch += 1

    return net, loss_list, history, grad_list, metrics


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
    eig = torch.linalg.eigvals(W).cpu()
    return float(torch.max(torch.abs(eig)).item())


def _frob(W: torch.Tensor) -> float:
    """
    Compute Frobenius norm of a matrix.

    Args:
        W (torch.Tensor): Input matrix.

    Returns:
        float: Frobenius norm.
    """
    return float(torch.linalg.matrix_norm(W, ord="fro").item())


def record_grads(net, grad_list):
    """
    Collect gradient statistics (mean, std, norm, mean_sq) for each parameter
    and append them to grad_list.

    Args:
        net (nn.Module): Model whose gradients we want to record.
        grad_list (list): A list that will store gradient dictionaries.
    """
    stats = {}
    for name, p in net.named_parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().cpu().numpy()
            stats[name] = {
                "mean": float(g.mean()),
                "std": float(g.std()),
                "l2_norm": float(np.linalg.norm(g)),
                "mean_sq": float((g**2).mean()),
            }
        grad_list.append(stats)


def step_YC(x):
    return (x >= 0).float()


def load_hidden_weight_into(
    model, npy_path, device=None, zero_bias=True, trainable=True
):
    """
    Loads an (n,n) matrix from .npy and copies it into model.hidden_linear.weight.
    """
    W = np.load(npy_path).astype(np.float32)
    thW = torch.from_numpy(W)

    if device is None:
        device = next(model.parameters()).device
    thW = thW.to(device)

    if model.hidden_linear.weight.shape != thW.shape:
        raise ValueError(
            f"Shape mismatch: layer {tuple(model.hidden_linear.weight.shape)} vs file {tuple(thW.shape)}"
        )

    with torch.no_grad():
        model.hidden_linear.weight.copy_(thW)
        if zero_bias and model.hidden_linear.bias is not None:
            model.hidden_linear.bias.zero_()

    # make it trainable or frozen
    model.hidden_linear.weight.requires_grad = bool(trainable)
    if model.hidden_linear.bias is not None:
        model.hidden_linear.bias.requires_grad = bool(trainable)

    print(f"Loaded {npy_path} into hidden_linear.weight (trainable={trainable})")
    return model


if __name__ == "__main__":
    main()
