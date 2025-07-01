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


parser = argparse.ArgumentParser(description="PyTorch Elman BPTT Training")
parser.add_argument(
    "--epochs",
    default=50000,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=200,
    type=int,
    metavar="N",
    help="mini-batch size (default: 200)",
)
parser.add_argument(
    "-o",
    "--one-sample",
    default=1,
    type=int,
    help="whether one traversal is one sample",
)
parser.add_argument(
    "-a", "--adam", default=0, type=int, help="whether to use ADAM or not"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
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
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on remapping scenario')
parser.add_argument(
    "-g", "--gpu", default=1, type=int, help="whether enable GPU computing"
)
parser.add_argument("-n", "--n", default=200, type=int, help="Input/output size")
parser.add_argument("--hidden-n", default=200, type=int, help="Hidden dimension size")
parser.add_argument(
    "-t", "--total-steps", default=2000, type=int, help="Total steps per traversal"
)
parser.add_argument(
    "--savename", default="net", type=str, help="Default output saving name"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--ae", default=0, type=int, help="Autoencoder or not")
parser.add_argument("--input_osci", default=0, type=int, help="Use oscilatory signal")
parser.add_argument(
    "--noisy", default=0, type=int, help="Gaussian noise sd (Percentage of input)"
)
parser.add_argument("--noisy2", default=0, type=int, help="whether per element std")
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
    "--nonoverlap",
    default=0,
    type=int,
    help="If input and trainable weights are nonoverlapping subsets (resume=True)",
)
parser.add_argument(
    "--input", default="", type=str, help="Load in user defined input sequence"
)
# parser.add_argument('--recordhidden', default=0, type=int, help='whether to record hidden state every step')
parser.add_argument(
    "--continuous",
    default=0,
    type=int,
    help="whether to inherent hidden state from previous epoch",
)
parser.add_argument(
    "--relu", default=0, type=int, help="relu activation function in the hidden layer"
)
parser.add_argument(
    "--interleaved",
    default=0,
    type=int,
    help="train one sample, update weight, train another, update weight",
)
parser.add_argument(
    "--interval", default=0, type=int, help="the interval of interleaved training"
)
parser.add_argument("--fixi", default=0, type=int, help="whether fix the input matrix")
parser.add_argument(
    "--fixw", default=0, type=int, help="whether fix the recurrent weight (plus bias)"
)
parser.add_argument(
    "--nobias",
    default=0,
    type=int,
    help="whether to remove all bias term in RNN module",
)
parser.add_argument(
    "--custom",
    default=0,
    type=float,
    help="self-defined RNN: hidden dropout probability",
)
parser.add_argument(
    "--ac_output",
    default="",
    type=str,
    help="set the output activation function to tanh (default softmax)",
)
parser.add_argument(
    "--Hregularized",
    default=0,
    type=float,
    help="regularization weight (/hidden_N) of the loss function",
)
parser.add_argument(
    "--noisytrain",
    default=0,
    type=int,
    help="Stochastic input during training, sd=args.noisy",
)
parser.add_argument(
    "--pred", default=0, type=int, help="whether use one-step future pred loss"
)
parser.add_argument(
    "--pred2",
    default=0,
    type=int,
    help="whether use multi-step future pred loss, update at each time step",
)
parser.add_argument(
    "--predfd", default=0, type=int, help="one-step future pred loss with feedback"
)
parser.add_argument(
    "--pred_d", default=0, type=int, help="multiple-step ahead prediction"
)
# parser.add_argument('--ownnet',default=0, type = float, help='user defined RNN')
# parser.add_argument('--momentum', default=0.01, type=float, metavar='M',
# help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
# metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--lr_step', default='40,60', help='decreasing strategy')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
# help='use pre-trained model')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
# help='manual epoch number (useful on restarts)')


def main():
    global args

    args = parser.parse_args()
    lr = args.lr
    n_epochs = args.epochs
    RecordEp = args.print_freq
    SeqN = args.batch_size
    N = args.n
    hidden_N = args.hidden_n
    TotalSteps = args.total_steps

    global f
    f = open(args.savename + ".txt", "w")
    print("Settings:", file=f)
    print(str(sys.argv), file=f)

    ## Generate network input
    # Circular input
    X, Target = BellShape_input(N, TotalSteps)
    if args.input_osci:
        X, Target = Cos_input(N, TotalSteps, args.input_osci)
    # sparse input signal
    if args.sparsity:
        print("sparsity of input: {}".format(args.sparsity), file=f)
        N_pre = np.int64(N * args.sparsity)
        X_pre, tmp = BellShape_input(N_pre, TotalSteps)
        np.random.seed(2)
        idx_active1 = np.random.choice(N, N_pre)
        np.random.seed(3)
        idx_active2 = np.random.choice(N, N_pre)
        X = np.zeros((N, TotalSteps))
        Target = np.zeros((N, TotalSteps))
        X[idx_active1, :] = X_pre
        Target[idx_active2, :] = X_pre
    # noisy input
    if args.noisy:
        if args.noisy2:
            print("Noisy input ({:d}% of each element)".format(args.noisy), file=f)
            X = X + args.noisy / 100 * X * np.random.normal(0, 1, X.shape)
        else:
            print("Noisy input ({:d}% of maximum)".format(args.noisy), file=f)
            sd = args.noisy * np.max(X) / 100
            X = X + np.random.normal(0, 1, X.shape) * sd
            X = X / np.amax(abs(X))
    if args.noisytrain:
        print(
            "Noisy input ({:d}%), Stochastic input and output during training".format(
                args.noisy
            ),
            file=f,
        )

    # Prepare input and target for the model: decrease the time resolution
    if args.one_sample:
        print("One travesal as one sample: SeqN={:d}".format(SeqN), file=f)
        Select_T = np.arange(0, TotalSteps, np.int64(TotalSteps / SeqN), dtype=int)
        tmp = np.expand_dims((X[:, Select_T].T), axis=0)
        X_mini = torch.tensor(tmp.astype(np.single))
        tmp = np.expand_dims((Target[:, Select_T].T), axis=0)
        Target_mini = torch.tensor(tmp.astype(np.single))  # Output: (batch*seq*feature)
    else:
        print("Splitting into multiple samples", file=f)
        b_idx = np.arange(0, X.shape[1], SeqN)
        X_batch = np.zeros(
            (b_idx.shape[0], SeqN, np.int64(N))
        )  # NBatch * NSeq * NFeature
        Target_batch = np.zeros((b_idx.shape[0], SeqN, np.int64(N)))
        for i in range(b_idx.shape[0]):
            X_batch[i, :, :] = X[:, b_idx[i] : b_idx[i] + SeqN].T
            Target_batch[i, :, :] = Target[:, b_idx[i] : b_idx[i] + SeqN].T
        X_mini = torch.tensor(X_batch.astype(np.single))
        Target_mini = torch.tensor(Target_batch.astype(np.single))

    ##  define network module
    net = ElmanRNN_pytorch_module(N, hidden_N, N)
    if args.ac_output == "none":
        print("Remove the output activation function", file=f)
        net = ElmanRNN_v3(N, hidden_N, N)
    if args.Hregularized:
        net = ElmanRNN_pytorch_module_v2(N, hidden_N, N)
    if args.pred:
        print("Network output prediction one-step ahead", file=f)
        net = ElmanRNN_pred(N, hidden_N, N)
        print("Input dim: ", net.input_dim)
        print("Hidden dim: ", net.hidden_dim)
        print("Output dim: ", net.output_dim)

        print("Input → Hidden weights:\n", net.input_linear.weight)
        print("Shape: ", net.input_linear.weight.shape)
        plt.figure()
        plt.imshow(net.input_linear.weight.detach().numpy())
        plt.savefig("./figures/mh_input_linear_weights.png")
        print()
        # print("Input → Hidden bias:\n", net.input_linear.bias)

        print("Hidden → Hidden weights:\n", net.hidden_linear.weight)
        print("Shape: ", net.hidden_linear.weight.shape)
        print()
        plt.figure()
        plt.imshow(net.hidden_linear.weight.detach().numpy())
        plt.savefig("./figures/mh_hidden_linear_weights.png")
        # print("Hidden → Hidden bias:\n", net.hidden_linear.bias)

        print("Hidden → Output weights:\n", net.linear3.weight)
        print("Shape: ", net.linear3.weight.shape)
        print()
        plt.figure()
        plt.imshow(net.linear3.weight.detach().numpy())
        plt.savefig("./figures/mh_output_linear_weights.png")
        # print("Hidden → Output bias:\n", net.linear3.bias)
        quit()
    if args.pred and args.Hregularized:
        print("Network output predeiction one-step ahead and Hregularized", file=f)
        net = ElmanRNN_pred_v2(N, hidden_N, N)
    if args.predfd:
        print("Predict one step ahead using feedback", file=f)
        net = ElmanRNN_pred_feedback(N, hidden_N, N)
    if args.pred_d:
        net = ElmanRNN_pred_v3(N, hidden_N, N, args.pred_d)
        print(
            "Network output prediction {}-step ahead".format(str(args.pred_d)), file=f
        )
    if args.relu:
        net.rnn = nn.RNN(N, hidden_N, 1, batch_first=True, nonlinearity="relu")
    if args.fixi:
        for name, p in net.named_parameters():
            if name == "rnn.weight_ih_l0":
                p.requires_grad = False
                p.data.fill_(0)
                p.data.fill_diagonal_(1)
                print("Fixing input matrix to identity matrix", file=f)
            elif name == "rnn.bias_ih_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing input bias to 0", file=f)
    if args.nobias:
        for name, p in net.named_parameters():
            if name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing RNN bias to 0", file=f)
    if args.fixw:
        for name, p in net.named_parameters():
            if name == "rnn.weight_hh_l0":
                p.requires_grad = False
                p.data = torch.rand(p.data.shape) * 2 * 1 / np.sqrt(N) - 1 / np.sqrt(N)
                print("Fixing recurrent matrix to a random matrix", file=f)
            elif name == "rnn.bias_hh_l0":
                p.requires_grad = False
                p.data.fill_(0)
                print("Fixing input bias to 0", file=f)
    if args.custom:
        print("randomly zero out hidden unit activity", file=f)
        net = ElmanRNN_sparse(N, hidden_N, N, args.custom)
    if args.ac_output == "tanh":
        net.act = nn.Tanh()
        print("Change output activation function to tanh", file=f)
    if args.ac_output == "relu":
        net.act = nn.ReLU()
        print("Change output activation function to relu", file=f)

    # if args.ownnet:
    #     print('Written RNN instead of nn modules', file = f)
    #     net = ElmanRNN(N,hidden_N,N)

    ## MSE criteria
    criterion = nn.MSELoss(reduction="sum")
    # criterion = nn.MSELoss(reduction='mean')

    ##  load checkpoint and resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading previous network '{}'".format(args.resume), file=f)
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint["state_dict"])
            print("=> loaded previous network '{}' ".format(args.resume), file=f)
            X = checkpoint["X_mini"]
            Target = checkpoint["Target_mini"]
            X_new = np.copy(X)
            Target_new = np.copy(Target)
            idx = np.arange(np.int64(N))
            np.random.seed(20)
            np.random.shuffle(idx)
            X_mini = X[:, :, idx]
            idx = np.arange(np.int64(N))
            np.random.seed(30)
            np.random.shuffle(idx)
            Target_mini = Target[:, :, idx]
            print("Shuffle the original input and target", file=f)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=f)

    ## if autoencoder
    if args.ae:
        print("Autoencoder scenario: Target = Input", file=f)
        Target_mini = X_mini

    ## if use user-defined input
    if args.input:
        loaded = torch.load(args.input)
        X_mini = loaded["X_mini"]
        Target_mini = loaded["Target_mini"]

    print(X_mini.shape)
    # H0 value
    h0 = torch.zeros(1, X_mini.shape[0], hidden_N)  # n_layers * BatchN * NHidden

    ## enable GPU computing
    if args.gpu:
        print("Cuda device availability: {}".format(torch.cuda.is_available()), file=f)
        criterion = criterion.cuda()
        net = net.cuda()
        X_mini = X_mini.cuda()
        Target_mini = Target_mini.cuda()
        h0 = h0.cuda()

    # construct optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if args.adam:
        print("Using ADAM optimizer", file=f)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # create weight mask or null mask
    if args.partial:
        # Currently only enable partial training for RNN weights
        print("Training sparsity:{}".format(args.partial), file=f)
        Mask_W = np.random.uniform(
            0, 1, (hidden_N, hidden_N)
        )  # determine the set of connections to be trained
        Mask_B = np.random.uniform(0, 1, (hidden_N))
        Mask_W = Mask_W > args.partial
        Mask_B = Mask_B > args.partial  # True == untrained connections
        if args.nonoverlap:
            Mask_W = ~(~(Mask_W) & checkpoint["Mask_W"])
            Mask_B = ~(~(Mask_B) & checkpoint["Mask_B"])
        Mask = []
        for name, p in net.named_parameters():
            if name == "rnn.weight_hh_l0" or name == "hidden_linear.weight":
                Mask.append(Mask_W)
                print("Partially train RNN weight", file=f)
            elif name == "rnn.bias_hh_l0" or name == "hidden_linear.bias":
                Mask.append(Mask_B)
                print("Partially train RNN bias", file=f)
            else:
                Mask.append(np.zeros(p.shape))
    else:
        Mask = []
        for name, p in net.named_parameters():
            Mask.append(np.zeros(p.shape))

    # For debug
    # # save network input, state
    # save_dict = {'state_dict': net.state_dict(),
    #     'X_mini': X_mini.cpu(),
    #     'Target_mini': Target_mini.cpu(),
    #     }
    # if args.partial:
    #     save_dict['Mask'] = Mask
    #     save_dict['Mask_W'] = Mask_W; save_dict['Mask_B'] = Mask_B
    # torch.save(save_dict, args.savename+'.pth.tar')

    # start training or step-wise training
    start = time.time()
    if args.interleaved:
        net, loss_list, y_hat, hidden = train_interleaved(
            X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask
        )
    elif args.interval:
        net, loss_list, y_hat, hidden = train_interval(
            X_mini,
            Target_mini,
            h0,
            n_epochs,
            net,
            criterion,
            optimizer,
            RecordEp,
            Mask,
            args.interval,
        )
    elif args.Hregularized:
        net, loss1_list, loss2_list, y_hat, hidden = train_Hregularized(
            X_mini,
            Target_mini,
            h0,
            n_epochs,
            net,
            criterion,
            optimizer,
            RecordEp,
            Mask,
            args.Hregularized / hidden_N,
        )
        loss_list = [loss1_list, loss2_list]
        print(
            "Add hidden unit firing cost, weight: {}".format(
                args.Hregularized / hidden_N
            ),
            file=f,
        )
    elif args.pred2:
        net, loss_list, y_hat, hidden = train_everyT(
            X_mini,
            Target_mini,
            h0,
            n_epochs,
            net,
            criterion,
            optimizer,
            RecordEp,
            Mask,
            args.pred2,
        )
        print(
            "Train at each time step and predicting {} steps into the future".format(
                args.pred2
            ),
            file=f,
        )
    else:
        net, loss_list, y_hat, hidden = train_partial(
            X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask
        )
    end = time.time()
    deltat = end - start
    print("Total training time: {0:.1f} minuetes".format(deltat / 60), file=f)

    # save network input, state
    save_dict = {
        "state_dict": net.state_dict(),
        "y_hat": y_hat,
        "X_mini": X_mini.cpu(),
        "Target_mini": Target_mini.cpu(),
        "hidden": hidden,
        "loss": loss_list,
    }
    if args.partial:
        save_dict["Mask_W"] = Mask_W
        save_dict["Mask_B"] = Mask_B
    torch.save(save_dict, args.savename + ".pth.tar")

    # plot loss function iteration curve
    plt.figure()
    if args.Hregularized:
        line1 = plt.plot(loss1_list)
        line2 = plt.plot(loss2_list)
        line3 = plt.plot(np.array(loss1_list) + np.array(loss2_list))
        plt.legend(["Target loss", "Hidden unit loss", "Total loss"])
        plt.title("Loss iteration: lamda={:.4f}".format(args.Hregularized / hidden_N))
    else:
        plt.plot(loss_list)
        plt.title("Loss iteration")
    plt.savefig(args.savename + ".png")


# def train(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp):
#     '''
#     Use SGD to train neural network
#     (y_hat and hidden only recorded for batch=1)
#         INPUT:
#             X_mini: batchN*seqN*featureN
#             Target_mini: batchN*seqN*featureN
#             n_epochs: number of epoches to train
#             net: nn.module: pre-defined network structure
#     '''
#     params = list(net.parameters())
#     print('{} parameters to optimize'.format(len(params)))
#     loss_list = []
#     y_hat = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     hidden = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     start = time.time()
#     for epoch in range(n_epochs):
#         output, hidden = net(X_mini, h0)
#         optimizer.zero_grad() # Clears existing gradients from previous epoch
#         loss = criterion(output,Target_mini)
#         loss.backward() # Does backpropagation and calculates gradients
#         optimizer.step() # Updates the weights accordingly
#         loss_list = np.append(loss_list,loss.item())
#         if epoch%RecordEp == 0:
#             end = time.time(); deltat= end - start; start = time.time()
#             print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))
#             print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
#             print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
#             y_hat[np.int(epoch/RecordEp),:,:] = output.cpu().detach().numpy()[0,:,:]
#             hidden[np.int(epoch/RecordEp),:,:] = hidden.cpu().detach().numpy()[0,:,:]
#     return net, loss_list, y_hat, hidden


def train_partial(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask
):
    """
    With untrainable weight mask
    Fix the intermediate recording of predloss (10/26/2021 Y.C.)
    Note first time-step info is not used (07/08/2021,Y.C.)
    Add stop criteria (07/11/2021, Y.C.)
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer:
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
        OUTPUT:
            y_hat: BatchN*RecordN*SeqN*HN
            hidden: BatchN*RecordN*SeqN*HN
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss_list = []
    h_t = h0
    batch_size, SeqN, N = X_mini.shape
    _, _, hidden_N = h_t.shape
    y_hat = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, N))
    hidden = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, hidden_N))
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
        if args.continuous:
            output, h_t = net(X_mini, h_t.detach())
        else:
            output, h_t = net(X_mini, h0)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        if args.pred_d:
            loss = criterion(
                output[:, args.pred_d :, :], Target_mini[:, args.pred_d :, :]
            )
        else:
            loss = criterion(
                output[:, 1:, :], Target_mini[:, 1:, :]
            )  # ignore the first time step
        loss.backward()  # Does backpropagation and calculates gradients
        for l, p in enumerate(net.parameters()):
            if p.requires_grad:
                p.grad.data[torch.from_numpy(Mask[l].astype(bool))] = 0
        optimizer.step()  # Updates the weights accordingly
        if epoch > 1000:
            diff = [loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            # init_loss = np.mean(np.array(loss_list[0:10]))
            init_loss = np.mean(np.array(loss_list[0]))
            if mean_diff < loss.item() * 0.00001 and loss.item() < init_loss * 0.010:
                stop = 1
        loss_list = np.append(loss_list, loss.item())
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
            if args.pred:
                hidden_seq = np.zeros((batch_size, SeqN, hidden_N))
                output = output.cpu().detach().numpy()
            elif args.continuous:
                output, hidden_seq = evaluate_onestep(
                    X_mini, Target_mini, h_t, net, criterion
                )
            else:
                output, hidden_seq = evaluate_onestep(
                    X_mini, Target_mini, h0, net, criterion
                )
            y_hat[:, np.int64(epoch / RecordEp), :, :] = output
            hidden[:, np.int64(epoch / RecordEp), :, :] = hidden_seq
            if args.noisytrain:
                sd1 = args.noisy * X_mini.cpu().numpy().max() / 100.0
                sd2 = args.noisy * Target_mini.cpu().numpy().max() / 100.0
                X_mini = X_mini + torch.tensor(
                    np.random.normal(0, 1, X_mini.shape).astype(np.single) * sd1
                ).to(X_mini.device)
                Target_mini = Target_mini + torch.tensor(
                    np.random.normal(0, 1, Target_mini.shape).astype(np.single) * sd2
                ).to(Target_mini.device)
                X_mini.detach()
                Target_mini.detach()
        epoch = epoch + 1
    return net, loss_list, y_hat, hidden


def train_everyT(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask, k=1
):
    """
    Modified from train_partial.py (07/17/2021)
    Collect loss function from every time step
    Predict k steps into the future
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer:
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
            k: number of steps predicting into the future
        OUTPUT:
            y_hat: BatchN*RecordN*SeqN*HN
            hidden: BatchN*RecordN*SeqN*HN
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss_list = []
    h_t = h0
    batch_size, SeqN, N = X_mini.shape
    _, _, hidden_N = h_t.shape
    y_hat = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, N))
    hidden = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, hidden_N))
    start = time.time()
    epoch = 0
    stop = 0
    while stop == 0 and epoch < n_epochs:
        # adaptively adjust learning rate
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(",")))
            if epoch in lr_step:
                print("Decrease lr to 50per at epoch {}".format(epoch), file=f)
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
        # form an average loss function
        loss_list_pre = []
        for t in np.arange(SeqN - k):
            X_t = X_mini[:, : t + k, :]
            X_t[:, t + 1 : t + k, :] = 0
            Target_t = Target_mini[:, : t + k, :]
            output, h_t = net(X_t, h0)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            loss = criterion(output, Target_t)
            loss.backward()  # Does backpropagation and calculates gradients
            for l, p in enumerate(net.parameters()):
                if p.requires_grad:
                    p.grad.data[Mask[l]] = 0
            optimizer.step()  # Updates the weights accordingly
            loss_list_pre = np.append(loss_list_pre, loss.item())
        loss_list = np.append(loss_list, np.mean(loss_list_pre))
        if epoch > 1000:
            diff = [loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            init_loss = np.mean(np.array(loss_list[0:10]))
            if mean_diff < loss.item() * 0.01 and loss.item() < init_loss * 0.1:
                stop = 1
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
            if args.continuous:
                output, hidden_seq = evaluate_onestep(
                    X_mini, Target_mini, h_t, net, criterion
                )
            else:
                output, hidden_seq = evaluate_onestep(
                    X_mini, Target_mini, h0, net, criterion
                )
            y_hat[:, np.int64(epoch / RecordEp), :, :] = output
            hidden[:, np.int64(epoch / RecordEp), :, :] = hidden_seq
            if args.noisytrain:
                sd1 = args.noisy * X_mini.cpu().numpy().max() / 100.0
                sd2 = args.noisy * Target_mini.cpu().numpy().max() / 100.0
                X_mini = X_mini + torch.tensor(
                    np.random.normal(0, 1, X_mini.shape).astype(np.single) * sd1
                ).to(X_mini.device)
                Target_mini = Target_mini + torch.tensor(
                    np.random.normal(0, 1, Target_mini.shape).astype(np.single) * sd2
                ).to(Target_mini.device)
                X_mini.detach()
                Target_mini.detach()
        epoch = epoch + 1
    return net, loss_list, y_hat, hidden


def train_interleaved(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask
):
    """
    Perform interleaved training: train one sample, update weight, train another sample, update again
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer:
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
        OUTPUT:
            loss: length: batch_size*epoch
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    batch_size = X_mini.shape[0]
    loss_list = []
    h_t = h0
    y_hat = np.zeros(
        (batch_size, np.int64(n_epochs / RecordEp), X_mini.shape[1], X_mini.shape[2])
    )
    hidden = np.zeros(
        (batch_size, np.int64(n_epochs / RecordEp), X_mini.shape[1], X_mini.shape[2])
    )
    start = time.time()
    for epoch in range(n_epochs):
        for b in np.arange(batch_size):
            if args.continuous:
                output, h_t = net(
                    X_mini[b : b + 1, :, :], h_t[:, b : b + 1, :].detach()
                )
            else:
                output, h_t = net(X_mini[b : b + 1, :, :], h0[:, b : b + 1, :])
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            loss = criterion(output, Target_mini[b : b + 1, :, :])
            loss.backward()  # Does backpropagation and calculates gradients
            for l, p in enumerate(net.parameters()):
                p.grad.data[Mask[l]] = 0
            optimizer.step()  # Updates the weights accordingly
            loss_list = np.append(loss_list, loss.item())
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
                if args.continuous:
                    output, hidden_seq = evaluate_onestep(
                        X_mini[b : b + 1, :, :],
                        Target_mini[b : b + 1, :, :],
                        h_t[:, b : b + 1, :],
                        net,
                        criterion,
                    )
                else:
                    output, hidden_seq = evaluate_onestep(
                        X_mini[b : b + 1, :, :],
                        Target_mini[b : b + 1, :, :],
                        h0[:, b : b + 1, :],
                        net,
                        criterion,
                    )
                y_hat[b, np.int64(epoch / RecordEp), :, :] = output[0, :, :]
                hidden[b, np.int64(epoch / RecordEp), :, :] = hidden_seq[0, :, :]
    return net, loss_list, y_hat, hidden


def train_interval(
    X_mini,
    Target_mini,
    h0,
    n_epochs,
    net,
    criterion,
    optimizer,
    RecordEp,
    Mask,
    interval,
):
    """
    Perform interleaved training with user-defined interval: train one sample for multiple times (interval)\
    ,update weight, train another sample for multiple times, update again
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer: 
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
            interval: training interval between two training samples
        OUTPUT:
            y_hat, hidden: only record at the last interval
            loss: length: batch_size*epoch
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    batch_size = X_mini.shape[0]
    loss_list = []
    h_t = h0
    y_hat = np.zeros(
        (batch_size, np.int64(n_epochs / RecordEp), X_mini.shape[1], X_mini.shape[2])
    )
    hidden = np.zeros(
        (batch_size, np.int64(n_epochs / RecordEp), X_mini.shape[1], X_mini.shape[2])
    )
    start = time.time()
    for epoch in range(n_epochs):
        for b in np.arange(batch_size):
            for it in np.arange(interval):
                if args.continuous:
                    output, h_t = net(
                        X_mini[b : b + 1, :, :], h_t[:, b : b + 1, :].detach()
                    )
                else:
                    output, h_t = net(X_mini[b : b + 1, :, :], h0[:, b : b + 1, :])
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                loss = criterion(output, Target_mini[b : b + 1, :, :])
                loss.backward()  # Does backpropagation and calculates gradients
                for l, p in enumerate(net.parameters()):
                    if p.requires_grad:
                        p.grad.data[Mask[l]] = 0
                optimizer.step()  # Updates the weights accordingly
                loss_list = np.append(loss_list, loss.item())
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
                if args.continuous:
                    output, hidden_seq = evaluate_onestep(
                        X_mini[b : b + 1, :, :],
                        Target_mini[b : b + 1, :, :],
                        h_t[:, b : b + 1, :],
                        net,
                        criterion,
                    )
                else:
                    output, hidden_seq = evaluate_onestep(
                        X_mini[b : b + 1, :, :],
                        Target_mini[b : b + 1, :, :],
                        h0[:, b : b + 1, :],
                        net,
                        criterion,
                    )
                y_hat[b, np.int64(epoch / RecordEp), :, :] = output[0, :, :]
                hidden[b, np.int64(epoch / RecordEp), :, :] = hidden_seq[0, :, :]
    return net, loss_list, y_hat, hidden


def train_Hregularized(
    X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask, lamda
):
    """
    Training with l2 regularization on hidden unit activity
    Note: use Elman_pytorch_module_v2 as net (output hidden unit: BatchN*SeqN*HiddenN)
    Note: the second output of net need to contain time sequence of hidden unit activity
    """
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss1_list = []
    loss2_list = []
    loss_list = []
    h_t = h0
    batch_size, SeqN, N = X_mini.shape
    hidden_N = h_t.shape[2]
    y_hat = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, N))
    hidden = np.zeros((batch_size, np.int64(n_epochs / RecordEp), SeqN, hidden_N))
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
        if args.continuous:
            output, h_t = net(X_mini, h_t[:, -1:, :].detach())
        else:
            output, h_t = net(X_mini, h0)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        loss1 = criterion(output[:, 1:, :], Target_mini[:, 1:, :])
        loss2 = lamda * criterion(h_t, torch.zeros(h_t.shape).to(X_mini.device))
        loss1_list.append(loss1.item())
        loss2_list.append(loss2.item())
        loss = loss1 + loss2
        loss_list.append(loss.item())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly
        if epoch > 1000:
            diff = [loss_list[i + 1] - loss_list[i] for i in range(len(loss_list) - 1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            init_loss = np.mean(np.array(loss_list[0]))
            if mean_diff < loss.item() * 0.01 and loss.item() < init_loss * 0.1:
                stop = 1
        for l, p in enumerate(net.parameters()):
            if p.requires_grad:
                p.grad.data[Mask[l]] = 0
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
            y_hat[:, np.int64(epoch / RecordEp), :, :] = output.cpu().detach().numpy()
            hidden[:, np.int64(epoch / RecordEp), :, :] = h_t.cpu().detach().numpy()
        epoch = epoch + 1
    return net, loss1_list, loss2_list, y_hat, hidden


# def train_continuH(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask):
#     '''
#     Loop the entire sequence each epoch
#     (y_hat and hidden only recorded for batch=1)
#         INPUT:
#             X_mini: batchN*seqN*featureN
#             Target_mini: batchN*seqN*featureN
#             n_epochs: number of epoches to train
#             net: nn.module: pre-defined network structure
#     '''
#     params = list(net.parameters())
#     print('{} parameters to optimize'.format(len(params)))
#     loss_list = []
#     y_hat = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     hidden = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     h_t = h0
#     start = time.time()
#     for epoch in range(n_epochs):
#         h_seq = np.zeros(X_mini.shape); output_seq = np.zeros(X_mini.shape)
#         optimizer.zero_grad() # Clears existing gradients from previous epoch
#         loss = 0; h_t = h_t.detach() # trucated BPTT, only BP one epoch
#         for t in np.arange(X_mini.shape[1]):
#             o_t,h_t = net(X_mini[:,t:t+1,:],h_t)
#             output_seq[:,t,:] = o_t.cpu().detach()
#             h_seq[:,t,:] = h_t.cpu().detach()
#             loss += criterion(o_t,Target_mini[:,t:t+1,:]);
#         loss.backward()
#         for l,p in enumerate(net.parameters()):
#             p.grad.data[Mask[l]] = 0
#         optimizer.step()
#         loss_list = np.append(loss_list,loss.item())
#         if epoch%RecordEp == 0:
#             end = time.time(); deltat= end - start; start = time.time()
#             print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))
#             print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
#             print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
#             y_hat[np.int(epoch/RecordEp),:,:] = output_seq[0,:,:]
#             hidden[np.int(epoch/RecordEp),:,:] = h_seq[0,:,:]
#     return net, loss_list, y_hat, hidden


def BellShape_input(N, TotalSteps):
    # generate bellshape circular input for N*TotalSteps
    X = np.zeros((np.int64(N), np.int64(TotalSteps)))  # input N*T
    Target = np.copy(X)  # pre-defined target firing rate
    tmp = np.linspace(norm.ppf(0.05), norm.ppf(0.95), np.int64(TotalSteps / 2))
    BellShape = norm.pdf(tmp)  # Bellshape vector
    template = np.concatenate((BellShape, np.zeros(np.int64(TotalSteps / 2))))
    X = np.zeros((np.int64(N), np.int64(TotalSteps)))  # time-shifting matrix
    for i in np.arange(np.int64(N)):
        X[i, :] = np.roll(template, np.int64(i * (TotalSteps / N)))
    X = X / np.sum(X, 0)  # Normalize X and Target by column
    idx = np.arange(np.int64(N))
    np.random.seed(10)
    np.random.shuffle(idx)
    Target = X[idx, :]
    return X, Target


def Cos_input(N, TotalSteps, T=2):
    """
    Generate cos-shape input with phase offsets for N*TotalSteps
    INPUT:
        T: number of oscillatory periods
    """
    X = np.zeros((np.int64(N), np.int64(TotalSteps)))  # input N*T
    Target = np.copy(X)  # pre-defined target firing rate
    omega = 2 * np.pi / (TotalSteps / T)
    phi = TotalSteps / T / N
    for i in np.arange(np.int64(N)):
        X[i, :] = np.cos(omega * np.arange(TotalSteps) + i * phi)
    idx = np.arange(np.int64(N))
    np.random.seed(10)
    np.random.shuffle(idx)
    Target = X[idx, :]
    return X, Target


def evaluate_onestep(X_mini, Target_mini, h_t, net, criterion):
    """
    Loop over entire sequence to record hidden activity
    """
    batch_size, SeqN, N = X_mini.shape
    _, _, hidden_N = h_t.shape
    h_seq = np.zeros((batch_size, SeqN, hidden_N))
    output_seq = np.zeros(X_mini.shape)
    h_t = h_t.detach()
    for t in np.arange(X_mini.shape[1]):
        o_t, h_t = net(X_mini[:, t : t + 1, :], h_t)
        output_seq[:, t : t + 1, :] = o_t.cpu().detach().numpy()
        h_seq[:, t : t + 1, :] = h_t.cpu().detach().numpy().transpose((1, 0, 2))
    return output_seq, h_seq


if __name__ == "__main__":
    main()
