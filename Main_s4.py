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
    net, loss_list = train_minibatch(
        X, Y, h0, n_epochs, net, criterion, optimizer, RecordEp, args.Hregularized
    )
    end = time.time()
    deltat = end - start
    print("Total training time: {0:.1f} minuetes".format(deltat / 60), file=f)

    # save network input, state
    save_dict = {"state_dict": net.state_dict(), "loss": loss_list}
    torch.save(save_dict, args.savename + ".pth.tar")

    # plot loss function iteration curve
    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss iteration")
    plt.savefig(args.savename + ".png")


def train_minibatch(X, Y, h0, n_epochs, net, criterion, optimizer, RecordEp, lamda):
    """
    INPUT:
        X: N*TimeLength*Input Dimension
        Y: (target signal)
        lamda: H-regularized term
    OUTPUT:
        y_hat: RecordN*X.shape
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss_list = []
    N = X.shape[0]
    criterion_l1 = nn.L1Loss(reduction="mean")
    # y_hat = np.zeros((np.int64(n_epochs/RecordEp),N,T,Ch,H_in,W_in))
    start = time.time()
    epoch = 0
    stop = 0
    while stop == 0 and epoch < n_epochs:
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(",")))
            if epoch in lr_step:
                print("Decrease lr to 50per at epoch {}".format(epoch), file=f)
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        loss_batch = []
        for batch in range(int(N / args.batch_size)):
            X_mini = X[batch * args.batch_size : (batch + 1) * args.batch_size,]
            Y_mini = Y[batch * args.batch_size : (batch + 1) * args.batch_size,]
            output, h_t = net(X_mini, h0)
            loss1 = criterion(output, Y_mini)
            loss2 = lamda * criterion(
                h_t, torch.zeros(h_t.shape).to(X_mini.device)
            ) + args.Hregularized_l1 * criterion_l1(
                h_t, torch.zeros(h_t.shape).to(X_mini.device)
            )
            loss = loss1 + loss2
            loss.backward()  # Does backpropagation and calculates gradients
            if args.clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()  # Updates the weights accordingly
            loss_batch.append(loss.item())
        loss_list.append(np.mean(loss_batch))
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
        epoch += 1
    return net, loss_list


def step_YC(x):
    return (x >= 0).float()


if __name__ == "__main__":
    main()
