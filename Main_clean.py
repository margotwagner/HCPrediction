# A clean version of Main.py for revision models
# Y.C. 8/8/2023
import argparse
import sys
import os
import shutil
import time
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN_Class import *


parser = argparse.ArgumentParser(description='PyTorch Elman BPTT Training')
parser.add_argument('--epochs', default=50000, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--lr_step', default='', type=str, help='decreasing strategy')
parser.add_argument('-p', '--print-freq', default=1000, type=int,metavar='N', help='print frequency (default: 1000)')
parser.add_argument('-g', '--gpu', default=1, type=int, help='whether enable GPU computing')
parser.add_argument('-n', '--n', default=200, type=int, help='Input/output size')
parser.add_argument('--hidden-n', default=200, type=int, help='Hidden dimension size')
parser.add_argument('-t','--total-steps', default=2000, type=int, help='Total steps per traversal')
parser.add_argument('--savename', default='', type = str, help='output saving name')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--ae', default=0, type=int, help='Autoencoder or not')
parser.add_argument('--partial', default=0, type=float, help='sparsity level (0-1) amount of the partially trained parameter')
parser.add_argument('--input', default='', type=str, help='Load in user defined input sequence')
parser.add_argument('--fixi', default=0, type=int, help='whether fix the input matrix')
parser.add_argument('--fixw', default=0, type=int, help='whether fix the recurrent weight (plus bias)')
parser.add_argument('--constraini', default=0, type=int, help='whether to constrain the input matrix to be positve')
parser.add_argument('--constraino', default=0, type=int, help='whether to constrain the output matrix to be positive')
parser.add_argument('--fixo', default=0, type=int, help='fix the output matrix')
parser.add_argument('--clamp_norm', default=0, type=float, help='if clamp the gradient norm')
parser.add_argument('--nobias', default=0, type=int, help='whether to remove all bias term in RNN module')
parser.add_argument('--rnn_act', default='', type=str, help='set the nonlinearity of hidden unit (default tanh)')
parser.add_argument('--ac_output', default='', type=str, help='set the output activation function to given nonlinearity (default softmax along dimension 2)')
parser.add_argument('--pred',default=0, type=int, help='whether use one-step future pred loss')
parser.add_argument('--noisy_train',default=0, type=float, help='whether add noise at each training step (default: None)')


def main():
    global args

    args = parser.parse_args()
    lr = args.lr
    n_epochs = args.epochs
    N = args.n
    hidden_N = args.hidden_n
    TotalSteps = args.total_steps

    global f
    f = open(args.savename+'.txt','w')
    print('Settings:', file = f)
    print(str(sys.argv), file = f)

    ## load in input
    loaded = torch.load(args.input)
    X_mini = loaded['X_mini']
    Target_mini = loaded['Target_mini']
    if args.ae:
        print('Autoencoder scenario: Target = Input', file = f)
        Target_mini = loaded['X_mini']

    if args.pred:
        X_mini = X_mini[:,:-1,:]
        Target_mini = Target_mini[:,1:,:]
        print('Predicting one-step ahead', file=f)

    ##  define network module
    net = ElmanRNN_pytorch_module_v2(N,hidden_N,N)
    # change rnn nonlinearity
    if args.rnn_act == 'relu':
        net.rnn = nn.RNN(N, hidden_N,1, batch_first=True, nonlinearity = 'relu')
        print('RNN nonlinearity: elementwise relu', file = f)
    # change output nonlinearity
    if args.ac_output == 'tanh':
        net.act = nn.Tanh()
        print('Change output activation function to tanh', file = f)
    elif args.ac_output == 'relu':
        net.act = nn.ReLU()
        print('Change output activation function to relu', file = f)
    elif args.ac_output == 'sigmoid':
        net.act = nn.Sigmoid()
        print('Change output activation function to sigmoid', file = f)


    if args.nobias:
        for name,p in net.named_parameters():
            if name == 'rnn.bias_hh_l0':
                p.requires_grad = False;
                p.data.fill_(0)
                print('Fixing RNN bias to 0', file=f)

    if args.fixi:
        for name,p in net.named_parameters():
            if name == 'rnn.weight_ih_l0':
                if args.fixi==1:
                    p.data = torch.ones(p.shape)/(p.shape[0]*p.shape[1])
                    print('Fixing {} to positive constant'.format(name), file=f)
                elif args.fixi==3:
                    p.data = p.data + torch.abs(p.data)
                    print('Fixing {} to positive initiation'.format(name), file=f)
                elif args.fixi==2:
                    print('Fixing {} to initializatin'.format(name),file=f)
                p.requires_grad = False
     
    if args.fixo:
        for name,p in net.named_parameters():
            if name == 'linear.weight':
                if args.fixo==1:
                    p.data = torch.ones(p.shape)/(p.shape[0]*p.shape[1])
                    print('Fixing {} to positive constant'.format(name), file=f)
                elif args.fixo==3:
                    p.data = p.data + torch.abs(p.data)
                    print('Fixing {} to positive initiation'.format(name), file=f)
                elif args.fixo==2:
                    print('Fixing {} to zero symmetric initiation'.format(name), file = f)
                p.requires_grad = False
        
    if args.fixw:
        for name,p in net.named_parameters():
            if name == 'rnn.weight_hh_l0':
                p.requires_grad = False;
                p.data = torch.rand(p.data.shape)*2*1/np.sqrt(N) - 1/np.sqrt(N)
                print('Fixing recurrent matrix to a random matrix', file=f)
            elif name == 'rnn.bias_hh_l0':
                p.requires_grad = False; 
                p.data.fill_(0)
                print('Fixing input bias to 0', file = f )
    

   ## MSE criteria
    criterion = nn.MSELoss(reduction='mean')

    ##  load checkpoint and resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading previous network '{}'".format(args.resume), file=f)
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded previous network '{}' ".format(args.resume), file=f)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=f)

    # H0 value
    h0 = torch.zeros(1,X_mini.shape[0],hidden_N) # n_layers * BatchN * NHidden

    ## enable GPU computing
    if args.gpu:
        print('Cuda device availability: {}'.format(torch.cuda.is_available()), file=f)
        criterion = criterion.cuda(); net= net.cuda()
        X_mini = X_mini.cuda(); Target_mini = Target_mini.cuda(); h0 = h0.cuda()

    # construct optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # if args.adam:
    #     print('Using ADAM optimizer', file=f)
    #     optimizer = torch.optim.Adam(net.parameters(),lr=lr)

    # create weight mask or null mask
    if args.partial:
        # Currently only enable partial training for RNN weights
        print('Training sparsity:{}'.format(args.partial),file=f)
        Mask_W = np.random.uniform(0,1,(hidden_N,hidden_N))# determine the set of connections to be trained
        Mask_B = np.random.uniform(0,1,(hidden_N))
        Mask_W = Mask_W > args.partial; Mask_B = Mask_B > args.partial # True == untrained connections
        if args.nonoverlap:
            Mask_W = ~(~(Mask_W) & checkpoint['Mask_W'])
            Mask_B = ~(~(Mask_B) & checkpoint['Mask_B'])
        Mask = []
        for name,p in net.named_parameters():
            if name == 'rnn.weight_hh_l0' or name == 'hidden_linear.weight': 
                Mask.append(Mask_W); print('Partially train RNN weight',file=f)
            elif name == 'rnn.bias_hh_l0' or name == 'hidden_linear.bias': 
                Mask.append(Mask_B); print('Partially train RNN bias',file=f)
            else:
                Mask.append(np.zeros(p.shape))
    else:
        Mask = [];
        for name,p in net.named_parameters():
            Mask.append(np.zeros(p.shape))


    # start training or step-wise training
    start = time.time()
    net, loss_list, grad_list, hidden, y_hat = train_partial(X_mini,Target_mini, h0, n_epochs, net, criterion, \
            optimizer, Mask)
    end = time.time(); deltat= end - start;
    print('Total training time: {0:.1f} minuetes'.format(deltat/60), file=f)

    # plot training curves
    plt.figure()
    plt.plot(loss_list);plt.title('Loss iteration'); 
    plt.savefig(args.savename+'.png')

    # save network input, state 
    save_dict = {'state_dict': net.state_dict(),
        'y_hat': np.array(y_hat),
        'hidden': np.array(hidden),
        'X_mini': X_mini.cpu(),
        'Target_mini': Target_mini.cpu(),
        'loss': loss_list,
        'grad_norm': grad_list}
    if args.partial:
        save_dict['Mask_W'] = Mask_W; save_dict['Mask_B'] = Mask_B
    torch.save(save_dict, args.savename+'.pth.tar')


def train_partial(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, Mask):
    '''
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
            net:
            loss_list:
            y_hat: BatchN*RecordN*SeqN*HN
    ''' 
    count = 0      
    for name,p in net.named_parameters():
        if p.requires_grad:
            print(name)
            count += 1
    print('{} parameters to optimize'.format(count))
    loss_list = []
    batch_size,SeqN,N = X_mini.shape; _,_,hidden_N = h0.shape
    start = time.time(); epoch = 0; stop = 0;
    hidden_rep = []
    output_rep = []
    grad_list = []
    while stop == 0 and epoch < n_epochs:
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(',')))
            if epoch in lr_step:
                print('Decrease lr to 50per at epoch {}'.format(epoch), file = f)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        if args.noisy_train:
            random_part = torch.normal(mean=torch.zeros(X_mini.shape),std=X_mini.cpu()*args.noisy_train).to(X_mini.device)
            X = X_mini + random_part
            Target = Target_mini + random_part
        else:
            X = X_mini
            Target = Target_mini
        output, h_seq = net(X,h0)
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        loss = criterion(output,Target) # ignore the first time step
        loss.backward() # Does backpropagation and calculates gradients
        # record the gradients
        tmp = []
        for name,p in net.named_parameters():
            if p.requires_grad:
                grad_np = p.grad.detach().cpu().numpy()
                tmp.append(np.mean(grad_np**2))
        grad_list.append(tmp)
        # partial training
        for l,p in enumerate(net.parameters()):
            if p.requires_grad:
                p.grad.data[torch.from_numpy(Mask[l].astype(bool))] = 0
        if args.clamp_norm:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clamp_norm)
        optimizer.step() # Updates the weights accordingly
        if args.constraini:
            for name,p in net.named_parameters():
                if name == 'rnn.weight_ih_l0':
                    p.data.clamp_(0)
        if args.constraino:
            for name,p in net.named_parameters():
                if name == 'linear_weight':
                    p.data.clamp_(0)

        # early stopping if plateu reached
        if epoch > 1000:
            diff = [loss_list[i+1]-loss_list[i] for i in range(len(loss_list)-1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            init_loss = np.mean(np.array(loss_list[0]))
            if mean_diff < loss.item()*0.00001 and loss.item() < init_loss*0.010: 
                stop = 1
        loss_list = np.append(loss_list,loss.item())
        epoch += 1
        if epoch%args.print_freq == 0:
            end = time.time(); deltat = end - start; start = time.time()
            hidden_rep.append(h_seq.detach().cpu().numpy())
            output_rep.append(output.detach().cpu().numpy())
            print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
            print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/args.print_freq/60))

    return net, loss_list, grad_list, hidden_rep, output_rep



if __name__ == '__main__':
    main()
