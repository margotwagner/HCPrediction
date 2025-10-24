import time
import os
import numpy as np
from numpy import loadtxt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import ndimage
from scipy.special import softmax
from scipy.stats import norm
import pickle as pk

# from mnist import MNIST
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN_Class import *
from deprecated.IO_plot import *

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

Path = "Elman_SGD/predloss/"

# trained MNIST models
Ns = 5
SeqN = 100
N = 68
hidden_N = 200
name = "MNIST_68PC_SeqN{}_Ns{}_partial".format(SeqN, Ns)
net = torch.load(Path + name + ".pth.tar")

X_mini = net["X_mini"].cpu()
Target_mini = net["Target_mini"].cpu()
model = ElmanRNN_pred(N, hidden_N, N)
model.act = nn.Tanh()
model.load_state_dict(net["state_dict"])

h_t = torch.zeros(1, Ns, hidden_N)
output, _ = model(X_mini, h_t)

# Keep on predicting using feedback signal
Stop_t = 17
X_null = torch.zeros((Ns, SeqN, N))
X_null[:, :Stop_t, :] = X_mini[:, :Stop_t, :]
output, _ = model(X_null, h_t)
o_future = torch.zeros(output.shape)
output_t, htp1 = model(X_mini[:, :Stop_t, :], torch.zeros(1, Ns, hidden_N))
o_future[:, :Stop_t, :] = output_t.detach()
for t in np.arange(SeqN - Stop_t):
    xtp1 = o_future[:, Stop_t + t - 1 : Stop_t + t + 1, :]
    otp1, htp1 = model(xtp1, htp1)
    o_future[:, Stop_t + t : Stop_t + t + 1, :] = otp1[:, 1:, :].detach()

o = np.zeros((Ns * SeqN, N))
count = 0
for i in np.arange(Ns):
    for j in np.arange(SeqN):
        o[count, :] = output.detach()[i, j, :]
        # o[count,:] = o_future.detach()[i,j,:]
        count = count + 1

o_re = pca.inverse_transform(o * scale + center)
o_re_reshape = np.zeros((Ns, SeqN, 784))
for i in np.arange(Ns):
    o_re_reshape[i, :, :] = o_re[i * SeqN : (i + 1) * SeqN, :]


# Panel B/C: Reconstruction and Prediction
plt.figure(figsize=(20, Ns))
count = 1
for i in np.arange(Ns):
    for j in np.arange(20):
        plt.subplot(Ns, 20, count)
        plt.imshow(o_re[i * SeqN + j, :].reshape((28, 28)))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        count = count + 1

plt.tight_layout()
plt.savefig(Path + name + "_O_re.pdf")
plt.close()


# recurrent unit activity
ht_seq = torch.zeros((Ns, SeqN, hidden_N))
htp1_seq = torch.zeros((Ns, SeqN, hidden_N))
ht = torch.zeros((1, Ns, hidden_N))
# for predloss
for t in np.arange(SeqN):
    ht = model.tanh(model.input_linear(X_mini[:, t, :]) + model.hidden_linear(ht))
    htp1 = model.tanh(model.hidden_linear(ht))
    ht_seq[:, t, :] = ht.detach()
    htp1_seq[:, t, :] = htp1.detach()

htp1_full = np.zeros((Ns * SeqN, hidden_N))
for i in np.arange(Ns):
    htp1_full[i * SeqN : (i + 1) * SeqN, :] = htp1_seq[i, :, :]


# Panel D: ICA
ICA = FastICA(n_components=10, max_iter=1000)
X_ic = ICA.fit_transform(htp1_full)  # results differ in permutation

IC1_list = np.array([0, 1, 2, 3])
IC2_list = np.array([0, 1, 2, 3])
plt.figure(figsize=(7, 7))
for idx in np.arange(len(IC1_list)):
    IC1 = IC1_list[idx]
    IC2 = IC2_list[idx]
    plt.subplot(2, 2, idx + 1)
    for digit in np.arange(10):
        idx_digit = np.arange(LoopN) * 10 + digit
        plt.plot(X_ic[idx_digit, IC1], X_ic[idx_digit, IC2], ".")
    plt.xlabel("IC{}".format(IC1))
    plt.ylabel("IC{}".format(IC2))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.rc("font", size=12)

plt.tight_layout()
plt.savefig(Path + name + "_htp1_cluster_select.eps")
plt.close()
