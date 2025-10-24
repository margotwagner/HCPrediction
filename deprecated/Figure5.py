import time
import os
import numpy as np
from numpy import loadtxt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy
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
from deprecated.helper import *
from deprecated.IO_plot import *

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

Data_path = "/nadata/cnl/home/yuchen/Documents/HCPrediction/Elman_SGD/GridInput/"
Model_path = (
    "/nadata/cnl/home/yuchen/Documents/HCPrediction/Elman_SGD/GridInput/BatchTraining/"
)

# Load the models
Ns = 50
SeqN = 100
tag = 1  # predictive loss
data_name = "Physical_input/InputNs{}_SeqN{}_StraightTraj_Marcus_v2".format(Ns, SeqN)
N, HN = 200, 500
model_name = (
    "PhysicalInput_v2/repeats/InputNs{}_SeqN{}_Marcus_HN500_H5.0_tp1_rep{}".format(
        Ns, SeqN, 1
    )
)
data = torch.load(Data_path + data_name + ".pth.tar")
X_mini = data["X_mini"]
Target_mini = data["Target_mini"]
Ns, SeqN, N = X_mini.shape
net = torch.load(Model_path + model_name + ".pth.tar", map_location="cuda:0")
model = ElmanRNN_tp1(N, HN, N)
model = ElmanRNN(N, HN, N)

model.act = nn.Sigmoid()
model.tanh = nn.ReLU()
model.load_state_dict(net["state_dict"])
ht_seq = np.zeros((Ns, SeqN, HN))
htp1_seq = np.zeros((Ns, SeqN, HN))
output = np.zeros((Ns, SeqN, N))
ht = torch.zeros((1, Ns, HN))
for t in range(SeqN):
    o, ht = model(X_mini[:, t : t + 1, :], ht)
    htp1 = model.tanh(model.hidden_linear(ht))
    output[:, t : t + 1, :] = o.detach().numpy()
    ht_seq[:, t, :] = ht[0, :, :].detach().numpy()
    htp1_seq[:, t, :] = htp1[0, :, :].detach().numpy()

print(np.mean((output - X_mini.numpy()) ** 2))  # loss per batch
print(np.mean((X_mini.numpy()) ** 2))  # baseline value per batch
print(net["loss"][-1])  # loss per batch

Traj_pre = loadtxt("RealInput/StraightTraj_v2.txt", delimiter="\t", unpack=False)
loc = Traj_pre[3:5, : SeqN * Ns]

select_idx = np.arange(1, HN, 2)
PF_pool = []
for neuron in select_idx:
    ac_pre = ht_seq[:, :, neuron].reshape(Ns * SeqN, 1)
    activity = ac_pre
    PF = Grid_PF(activity, loc, 20)
    PF_pool.append(PF)

PF_pool = []
neuron_list = []
I_list = []
for neuron in range(HN):
    if tag:
        activity = np.abs(htp1_seq[:, :, neuron].reshape(Ns * SeqN, 1))
    else:
        activity = np.abs(ht_seq[:, :, neuron].reshape(Ns * SeqN, 1))
    PF = Grid_PF(activity, loc, 20)
    if PF.mean() < 0.01:
        continue
    I = MI_Grid(activity, loc, 20)
    I_list.append(I)
    neuron_list.append(neuron)
    PF_pool.append(PF)

# Panel B: PFs of most Informative place fields
idx = np.argsort(I_list)[::-1]
plt.figure(figsize=(12, 12))
plt.rc("font", size=BIGGER_SIZE)
for c, i in enumerate(idx[:42]):
    plt.subplot(6, 7, c + 1)
    plt.imshow(PF_pool[i].T)
    # plt.colorbar()
    plt.title("MI={:.2f}".format(I_list[i]))
    plt.gca().invert_yaxis()
    plt.axis("off")

plt.tight_layout()
plt.savefig(Model_path + model_name + "_PF_ht_MI_second_selected.pdf")
plt.close()

torch.save(
    {"I_list": I_list, "neuron_list": neuron_list},
    Model_path + model_name + "_ht_MI_second.pth.tar",
)


# Panel C: MI distribution over repetitive training
Ns, SeqN = 50, 100
tag = 1
# tag = 0
data_name = "Physical_input/InputNs{}_SeqN{}_StraightTraj_Marcus_v2".format(Ns, SeqN)
N, HN = 200, 500

loss_true_rep = []
loss_rep = []
I_rep = []
for trial in np.arange(1, 11):
    if tag:
        model_name = "PhysicalInput_v2/repeats/InputNs{}_SeqN{}_Marcus_HN500_H5.0_tp1_rep{}".format(
            Ns, SeqN, trial
        )
    else:
        model_name = (
            "PhysicalInput_v2/repeats/InputNs{}_SeqN{}_Marcus_HN500_H5.0_rep{}".format(
                Ns, SeqN, trial
            )
        )
    data = torch.load(Data_path + data_name + ".pth.tar")
    X_mini = data["X_mini"]
    Target_mini = data["Target_mini"]
    net = torch.load(Model_path + model_name + ".pth.tar", map_location="cuda:0")
    if tag:
        model = ElmanRNN_tp1(N, HN, N)
    else:
        model = ElmanRNN(N, HN, N)
    model.act = nn.Sigmoid()
    model.tanh = nn.ReLU()
    model.load_state_dict(net["state_dict"])
    ht_seq = np.zeros((Ns, SeqN, HN))
    htp1_seq = np.zeros((Ns, SeqN, HN))
    output = np.zeros((Ns, SeqN, N))
    ht = torch.zeros((1, Ns, HN))
    for t in range(SeqN):
        o, ht = model(X_mini[:, t : t + 1, :], ht)
        htp1 = model.tanh(model.hidden_linear(ht))
        output[:, t : t + 1, :] = o.detach().numpy()
        ht_seq[:, t, :] = ht[0, :, :].detach().numpy()
        htp1_seq[:, t, :] = htp1[0, :, :].detach().numpy()
    print(np.mean((output - X_mini.numpy()) ** 2))  # loss per batch
    print(np.mean((X_mini.numpy()) ** 2))  # baseline value per batch
    print(net["loss"][-1])  # loss per batch
    Traj_pre = loadtxt("RealInput/StraightTraj_v2.txt", delimiter="\t", unpack=False)
    loc = Traj_pre[3:5, : SeqN * Ns]
    # Quantify MI between location and activity
    PF_pool = []
    neuron_list = []
    I_list = []
    for neuron in range(HN):
        if tag:
            activity = np.abs(htp1_seq[:, :, neuron].reshape(Ns * SeqN, 1))
        else:
            activity = np.abs(ht_seq[:, :, neuron].reshape(Ns * SeqN, 1))
        PF = Grid_PF(activity, loc, 20)
        if PF.mean() < 0.01:
            continue
        I = MI_Grid(activity, loc, 20)
        I_list.append(I)
        neuron_list.append(neuron)
        PF_pool.append(PF)
    I_rep.append(I_list)
    loss_rep.append(net["loss"][-1])
    loss_true_rep.append(np.mean((output - X_mini.numpy()) ** 2))

if tag:
    idx = np.where(np.array(loss_rep) < 0.02)[0]
else:
    idx = np.arange(8)

I_list = []
loss_list = []
loss_true_list = []  # mse between input and output
for i in idx:
    I_list = I_list + I_rep[i]
    loss_list.append(loss_rep[i])
    loss_true_list.append(loss_true_rep[i])

if tag:
    save_name = Model_path + model_name[:-2] + "_htp1_MI_second.pth.tar"
else:
    save_name = Model_path + model_name[:-2] + "_ht_MI_second.pth.tar"

torch.save(
    {"I_list": I_list, "loss_list": loss_list, "loss_true_list": loss_true_list},
    save_name,
)


sub_path = "PhysicalInput_v2/repeats/InputNs50_SeqN100_Marcus_HN500_H5.0"
current = torch.load(Model_path + sub_path + "_rep_ht_MI_second.pth.tar")
pred = torch.load(Model_path + sub_path + "_tp1_rep_htp1_MI_second.pth.tar")
Input = torch.load(
    Model_path
    + "PhysicalInput_v2/InputNs50_SeqN100_Marcus_HN500_H5.0_tp1_input_MI_second.pth.tar"
)

print(np.mean(current["I_list"]))
print(np.mean(pred["I_list"]))
print(np.mean(Input["I_list"]))
_, p = scipy.stats.ttest_ind(current["I_list"], pred["I_list"])

plt.figure(figsize=(3, 3.5))
bplot = plt.boxplot(
    [Input["I_list"], current["I_list"], pred["I_list"]],
    showfliers=False,
    patch_artist=True,
    notch=True,
    labels=["Input", "Current", "Predictive"],
)
for patch, color in zip(bplot["boxes"], ["tab:green", "tab:blue", "tab:orange"]):
    patch.set_facecolor(color)

plt.rc("font", size=BIGGER_SIZE)
plt.ylabel("MI")
ylims = plt.ylim()
plt.ylim([ylims[0], ylims[1] * 1.1])
plt.xticks(rotation=45, ha="right")
plt.title("Pvalue={:.2e}".format(p))
plt.tight_layout()
plt.savefig(Model_path + sub_path + "_ReLU_H2.0_MI_second_boxplot2.eps")
plt.close()
