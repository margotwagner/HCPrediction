import scipy.linalg as linalg 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
import matplotlib.animation as animation
import math
from mpl_toolkits.axes_grid1 import ImageGrid




def LSE(H,B):
	# H: unit dimension x time length
	# B: hidden dimension x time length
	N,T = H.shape
	M = B.shape[0]
	A = np.zeros((M,N))
	for i in range(M):
		bi = B[i,:]
		P = np.matmul(np.linalg.inv(np.matmul(H,np.transpose(H))),H)
		A[i,:] = np.matmul(P,bi)
	B_hat = np.matmul(A,H)
	MSE = ((B - B_hat)**2).mean()
	return A, MSE



# def ICA_infomax(X,epochs,rate = 1e-7,batch=10,tol=0.01):
# 	# X: input: dim x SampleSize
# 	# Data whitening: Y = whitened x s.t. cov(Y) = I
# 	N,T = X.shape
# 	Batch_N = int(T / batch)
# 	C = np.cov(X)
# 	D,V = linalg.eigh(C)
# 	W_z = linalg.sqrtm((linalg.inv(C)))
# 	Y = W_z.dot(X)
# 	C_Y = np.cov(Y)
# 	# initialize W matrix
# 	W = np.identity(N) 
# 	# W = np.random.normal(0,N,(N,N)) 
# 	# SGD
# 	i_sweep = 0.; length = 1e6
# 	update_count = 1
# 	while i_sweep < epochs and length > tol:
# 	    if i_sweep%100 ==0: print(i_sweep,length)
# 	    # permute Y for each sweep
# 	    idx = np.random.permutation(Y.shape[1])
# 	    Y = Y[:,idx]
# 	    for j in range(batch):
# 	        pieces_for_gradient = Y[:,Batch_N*j:Batch_N*(j+1)]
# 	        delta_W = np.zeros(W.shape)
# 	        for k in range(Batch_N):
# 	            u = W.dot(pieces_for_gradient[:,k])
# 	            # use logistic function g(u) = (1+exp(-u))^{-1}
# 	            y = 1/(np.exp(-u) + 1)
# 	            y_hat = 1-2*y;
# 	            delta_W = delta_W+(np.identity(N)+np.outer(y_hat,u)).dot(W)
# 	        length = np.mean(delta_W**2) / Batch_N
# 	        # print(length)
# 	        W = W+rate*delta_W;
# 	        update_count += 1
# 	    if update_count > 500: rate*=0.5; update_count=0
# 	    i_sweep += 1
# 	W_I = W*W_z
# 	return W_I



def GF_sim(lamda,theta,A,r0):
	# Simulate Grid Field using summation of three cosine functions
	# lamda: spatial frequence (spacing)
	# theta: orientation
	# A: maximum firing rate
	# r0: spatial phase (2-d vector) 
	k = 4*np.pi/(np.sqrt(3)*lamda)
	theta1 = theta + np.pi/12; 
	theta2 = theta + np.pi/12*5; 
	theta3 = theta+np.pi/12*9
	k1 = np.array([np.cos(theta1)+np.sin(theta1),np.cos(theta1)-np.sin(theta1)])
	k2 = np.array([np.cos(theta2)+np.sin(theta2),np.cos(theta2)-np.sin(theta2)])
	k3 = np.array([np.cos(theta3)+np.sin(theta3),np.cos(theta3)-np.sin(theta3)])
	NGrid = 100; x = np.linspace(-1,1,NGrid); y = np.linspace(-1,1,NGrid) 
	xx,yy = np.meshgrid(x,y)
	r = np.vstack((xx.flatten(), yy.flatten())).T - r0[:,];
	phi_flat = (np.cos(k/np.sqrt(2)*r.dot(k1)) + np.cos(k/np.sqrt(2)*r.dot(k2)) \
		+ np.cos(k/np.sqrt(2)*r.dot(k3)))/3
	g_flat = 2/3 * A* (phi_flat + 0.5)
	g = g_flat.reshape(NGrid, NGrid)
	return g




def Grid_PF(activity, loc, grid):
	si = (loc.max()-loc.min())/grid
	PF = np.full((grid,grid),np.nan)
	for i in np.arange(grid):
		for j in np.arange(grid):
			x_lb = i*si+loc.min(); x_ub = (i+1)*si+loc.min()
			y_lb = j*si+loc.min(); y_ub = (j+1)*si+loc.min()
			x_t = (loc[0,:]<=x_ub) & (loc[0,:]>x_lb)
			y_t = (loc[1,:]<=y_ub) & (loc[1,:]>y_lb)
			if np.sum(x_t&y_t) == 0:
				PF[i,j] = np.nan
			else:
				PF[i,j] = np.mean(activity[x_t&y_t])
	return PF

def Grid_px(loc,grid):
	# occupancy probability
	si = (loc.max()-loc.min())/grid
	PF = np.full((grid,grid),np.nan)
	for i in np.arange(grid):
		for j in np.arange(grid):
			x_lb = i*si+loc.min(); x_ub = (i+1)*si+loc.min()
			y_lb = j*si+loc.min(); y_ub = (j+1)*si+loc.min()
			x_t = (loc[0,:]<=x_ub) & (loc[0,:]>x_lb)
			y_t = (loc[1,:]<=y_ub) & (loc[1,:]>y_lb)
			PF[i,j] = np.sum(x_t&y_t)
	return PF


def MI_Grid(activity,loc,Grid_size):
	lamdax = Grid_PF(activity,loc,Grid_size)
	px = Grid_px(loc,Grid_size)
	px = (px / np.sum(px)).reshape((1,-1))
	lamdax = lamdax.reshape((1,-1))
	lamdax_idx = ~np.isnan(lamdax)
	lamda = np.sum(lamdax[lamdax_idx] * px[lamdax_idx])
	idx_nonzero = (lamdax != 0) & (lamdax_idx) 
	I = np.sum(lamdax[idx_nonzero]*np.log2(lamdax[idx_nonzero]/lamda)*px[idx_nonzero])
	return I

def MI_linear(fr,norm=True):
	# fr is firing rate with respect to a linear location
	if fr.mean() < 0.1:
		return 0
	fr_norm = fr/fr.mean()
	nonzero = fr_norm != 0
	if norm:
		return np.sum(fr_norm[nonzero]*np.log2(fr_norm[nonzero])) / fr.shape[0]
	else:
		return np.sum(fr[nonzero]*np.log2(fr_norm[nonzero])) / fr.shape[0]



def GF2PF_connection(A,sigma,N,l_max,l_min,lamda):
	coef = 1/A*2*np.pi*sigma**2*2*np.pi/N*np.log(l_max/l_min)
	W = coef * np.exp(-4/3*np.pi**2*sigma**2/lamda**2)/lamda**2
	return W


def py2mat(loaded):
	# INPUT:
	# 	loaded: model output file (*.pth.tar) from training
	matdict = {};
	for key in list(loaded.keys()):
		if key == 'state_dict':
			for name,p in loaded['state_dict'].items():
				newname = name.replace('.','_')
				matdict[newname] = p.detach().cpu().numpy()
		else:
			if isinstance(loaded[key],np.ndarray) or isinstance(loaded[key],list):
				matdict[key] = loaded[key]
			else:
				matdict[key] = loaded[key].numpy()
	return matdict 


def mnist_reverse(output,pca,scale,center):
	(Ns,SeqN,N) = output.shape
	o = np.zeros((Ns*SeqN,N))
	count = 0;
	for i in np.arange(Ns):
		for j in np.arange(SeqN):
			o[count,:] = output.detach()[i,j,:]
			count = count + 1
	o_re = pca.inverse_transform(o*scale+center)
	o_re_reshape = np.zeros((Ns,SeqN,784))
	for i in np.arange(Ns):
		o_re_reshape[i,:,:] = o_re[i*SeqN:(i+1)*SeqN,:]
	return o_re_reshape


def kmeans_label(k,input_full,label_full):
	# assign labels to kmeans clusters and get accuracy
	# unknown label is -9999 (int)
	# input_full: N (number of samples) x D (number of features)
	# label_full: N (number of samples) x 1
	N,D = input_full.shape
	kmeans = KMeans(n_clusters=k, random_state=0).fit(input_full)
	kmeans_pred = kmeans.labels_
	kmeans_label = np.zeros(N,dtype=int)
	for kmeans_class in range(k):
		class_idx = [i[0] for i in np.argwhere(kmeans_pred==kmeans_class)]
		labels = label_full[class_idx]
		kmeans_class_label = np.bincount(labels[labels!=-9999]).argmax()
		kmeans_label[class_idx] = kmeans_class_label
		Acc = np.sum(label_full == kmeans_label)/N
	return Acc

def negentropy(y):
	# y: one dimensional random variable with its realizations
	z = (y-np.mean(y))/np.sqrt(np.var(y))
	return (-np.mean(np.exp(-1/2*z**2))+np.sqrt(2)/2)**2


def plot_compare(rec, ori, title=" ", rec_title=" ", ori_title=" ", loss=0):
    plt.figure(figsize=[20, 5])
    plt.suptitle(title, fontsize=16)

    plt.subplot(131)
    plt.pcolormesh(rec.T, cmap="viridis")
    plt.title(rec_title)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(132)
    plt.pcolormesh(ori.T, cmap="viridis")
    plt.title(ori_title)
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.clim(0, 1)

    plt.subplot(133)
    plt.pcolormesh((rec - ori).T, cmap="seismic", norm=CenteredNorm())
    plt.title("MSE: " + str(round(loss, 4)))
    plt.xlabel("time")
    plt.ylabel("unit")
    plt.colorbar()
    plt.show()


def plot_input(x, name=""):
    plt.pcolormesh(x.T, cmap="viridis")
    plt.title(name)
    plt.xlabel("time")
    plt.ylabel("neuron #")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()


def plot_weights(model):
    plt.figure(figsize=[20, 5])

    plt.subplot(131)
    plt.pcolormesh(model.W_f.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_f$")
    plt.colorbar()

    plt.subplot(132)
    plt.pcolormesh(model.W_r.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_r$")
    plt.colorbar()

    plt.subplot(133)
    plt.pcolormesh(model.W_g.T, cmap="bwr", norm=CenteredNorm())
    plt.title("$W_g$")
    plt.colorbar()
    plt.show()


def plot_digit(x):
    N = x.shape[0]
    n = int(math.sqrt(N))
    plt.imshow(x.reshape(n, n), cmap="binary_r")
    plt.colorbar()
    plt.show()


def plot_digits(X):
    N = X.shape[0]
    fig,axes = plt.subplots(1,N,figsize=[N*2, 2])
    c_min = np.min(X)
    c_max = np.max(X)
    for ax,n in zip(axes,range(0,N)):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        ax.imshow(X[n].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    im = ax.imshow(X[0].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    fig.colorbar(im,ax=axes.ravel().tolist())
    plt.show(block=True)

def plot_digits_grid(X):
    N = X.shape[0]
    fig = plt.figure(figsize=(N*2,2))
    c_min = 0
    c_max = 1
    grid = ImageGrid(fig, (0,0,N,1),
                    nrows_ncols=(1,N),
                    axes_pad=0.15,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )
    for ax,n in zip(grid,range(0,N)):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        im = ax.imshow(X[n].reshape(28, 28), cmap="binary_r",vmin=c_min,vmax=c_max)
    ax.cax.colorbar(im)
    ax.cax.tick_params(labelsize=20)
    ax.cax.toggle_label(True)
    plt.show()


def plot_compare_digits(X_ori, X_rec):
    N = X_ori.shape[0]
    fig = plt.figure(figsize=[N * 3 + 3, 9])
    row = fig.subfigures(nrows=3, ncols=1)
    # fig.tight_layout()

    # Plot original X
    row[0].suptitle("Original", fontsize=16)
    col0 = row[0].subplots(nrows=1, ncols=N, sharey=True)
    c0 = col0[0].imshow(X_ori[0].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
    row[0].colorbar(c0, ax=col0, shrink=0.8)
    for n in range(1, N):
        col0[n].imshow(X_ori[n].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
        # col0[n].tick_params(left=False, bottom=False)

    # Plot recalled X
    row[1].suptitle("Recalled", fontsize=16)
    col1 = row[1].subplots(nrows=1, ncols=N, sharey=True)
    c1 = col1[0].imshow(X_rec[0].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
    row[1].colorbar(c1, ax=col1, shrink=0.8)
    for n in range(1, N):
        col1[n].imshow(X_rec[n].reshape(28, 28), cmap="viridis", vmin=0, vmax=0.1)
        # col1[n].tick_params(left=False, bottom=False)

    # Plot X error
    row[2].suptitle("Difference", fontsize=16)
    col2 = row[2].subplots(nrows=1, ncols=N, sharey=True)
    c2 = col2[0].imshow(
        (X_ori[0] - X_rec[0]).reshape(28, 28), cmap="bwr", vmin=-0.1, vmax=0.1
    )
    row[2].colorbar(c2, ax=col2, shrink=0.8)
    for n in range(1, N):
        col2[n].imshow(
            (X_ori[n] - X_rec[n]).reshape(28, 28), cmap="bwr", vmin=-0.1, vmax=0.1
        )
        # col2[n].tick_params(left=False, bottom=False)

    # plt.tight_layout()
    plt.show()


def plot_moving_digits(X_ori, X_rec):
    N = X_ori.shape[0]
    fig = plt.figure(figsize=[N * 2, 6])
    row = fig.subfigures(nrows=2, ncols=1)
    # fig.tight_layout()
    c_max = np.max(X_ori) * 2
    D = 64

    # Plot original X
    row[0].suptitle("Original", fontsize=16)
    col0 = row[0].subplots(nrows=1, ncols=N, sharey=True)
    c0 = col0[0].imshow(X_ori[0].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)
    row[0].colorbar(c0, ax=col0, shrink=0.8)
    for n in range(1, N):
        col0[n].imshow(X_ori[n].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)

    # Plot recalled X
    row[1].suptitle("Recalled", fontsize=16)
    col1 = row[1].subplots(nrows=1, ncols=N, sharey=True)
    c1 = col1[0].imshow(X_rec[0].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)
    row[1].colorbar(c1, ax=col1, shrink=0.8)
    for n in range(1, N):
        col1[n].imshow(X_rec[n].reshape(D, D), cmap="viridis", vmin=0, vmax=c_max)

    plt.show()


def animate_imgs(X, title="X", c_max=1, diff=False):
    fig, ax = plt.subplots()
    T = X.shape[0]
    D = 64
    c_min = 0
    # c_max = np.max(X)*2
    c_map = "viridis"
    if diff:
        c_min = -c_max
        c_map = "bwr"

    imgs = []
    for t in range(T):
        im = ax.imshow(X[t, :].reshape(D, D), cmap=c_map, vmin=c_min, vmax=c_max)
        imgs.append([im])

    mov = animation.ArtistAnimation(
        fig, imgs, interval=250, blit=True, repeat_delay=1000
    )
    mov.save(title + ".mp4")


def animate_3_imgs(X, X_rec, title="X"):
    fig, ax = plt.subplots()
    p0 = fig.add_subplot(131)
    p1 = fig.add_subplot(132)
    p2 = fig.add_subplot(133)

    T = X.shape[0]
    D = 64
    c_min = 0
    c_max = np.max(X) * 2

    X_diff = X - X_rec

    imgs = []
    for t in range(T):
        im0 = p0.imshow(X[t, :].reshape(D, D), cmap="viridis", vmin=c_min, vmax=c_max)

        im1 = p1.imshow(
            X_rec[t, :].reshape(D, D), cmap="viridis", vmin=c_min, vmax=c_max
        )

        im2 = p2.imshow(X_diff[t, :].reshape(D, D), cmap="bwr", vmin=-c_max, vmax=c_max)
        imgs.append(ax.get_images())
        ax.clear()

    m = animation.ArtistAnimation(
        fig, imgs, interval=250, blit=False, repeat_delay=1000
    )

    m.save(title + ".mp4")
    plt.show(m)


def plot_lines(X):
    N = X.shape[0]

    plt.figure(figsize=[N * 3, 3])

    for n in range(N):
        idx = 100 + 10 * N + (n + 1)
        plt.subplot(idx)
        plt.imshow(X[n], cmap="binary_r")

    plt.show()
