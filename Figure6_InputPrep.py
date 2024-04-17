import time
import os
import numpy as np
from numpy import loadtxt
import matplotlib
matplotlib.use('Agg')
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
from IO_plot import *
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

Path = 'Elman_SGD/predloss/'; 

images = np.load('Elman_SGD/predloss/Rotated/MNIST_X_train.npy')
labels = np.load('Elman_SGD/predloss/Rotated/MNIST_labels.npy')

pca = PCA(n_components=68)
pca.fit(images)
print(np.sum(pca.explained_variance_ratio_))
Im_de = pca.transform(images)
center = (np.max(Im_de)+np.min(Im_de))/2
scale = (np.max(Im_de)-np.min(Im_de))/2
Im_de_scale = (Im_de - center)/scale
Im_re = pca.inverse_transform(Im_de_scale*scale+center)

# Convert MNIST to PCA tensors
images = np.load('Elman_SGD/predloss/Rotated/MNIST_X_train.npy')
labels = np.load('Elman_SGD/predloss/Rotated/MNIST_labels.npy')

pca = PCA(n_components=68)
pca.fit(images)
print(np.sum(pca.explained_variance_ratio_))
Im_de = pca.transform(images)
center = (np.max(Im_de)+np.min(Im_de))/2
scale = (np.max(Im_de)-np.min(Im_de))/2
Im_de_scale = (Im_de - center)/scale
Im_re = pca.inverse_transform(Im_de_scale*scale+center)

im_num = [];
for i in np.arange(10):
	im = Im_de_scale[labels==i,:]
	im_num.append(im)

N = Im_de.shape[1]
SeqN = 100
Ns = 5
LoopN = round(SeqN/10)
tmp = np.zeros((Ns,SeqN,N))
for i in np.arange(Ns):
	for j in np.arange(10):
		idx = np.arange(0,10*LoopN,10) + j
		im_pre = im_num[j][np.arange(LoopN)+i*LoopN]
		tmp[i,idx,:] = im_pre

X_mini = torch.tensor(tmp.astype(np.single))
torch.save({'X_mini':X_mini,'Target_mini': X_mini},
	Path + 'MNIST_68PC_SeqN{}_Ns{}.pth.tar'.format(SeqN,Ns))