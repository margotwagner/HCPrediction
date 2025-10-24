# To plot IO for every trained model in the folder
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import argparse
import matplotlib.pyplot as plt
from scipy.special import softmax
from numpy import loadtxt
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN_Class import ElmanRNN_pytorch_module
from RNN_Class import ElmanRNN_pytorch_module_v2
import time
import glob
import os.path

parser = argparse.ArgumentParser(description='Plot IO figure in a folder')
parser.add_argument('--dic', default='', type=str)
global args; args = parser.parse_args()


# dic = 'Elman_SGD/GridInput/s0.3/'
dic = args.dic

def main(dic):
	model_list = []; Ns_list = []; SeqN_list = []
	for file in glob.glob(dic+'*.pth.tar'):
		model = file[len(dic):]; 	
		Ns_list.append(int(model.split('_')[0][2:]))
		SeqN_list.append(int(model.split('_')[1][4:]))
		model_list.append(model)
	for i in np.arange(len(model_list)):
		model = model_list[i]
		f = open(model[:-8]+'.txt','a+')
		if not os.path.exists(dic+model[:-8]+'_IO.png'):
			print('Plotting IO', file = f); print('IO plot:{}'.format(model))
			IO_plot(dic+model_list[i][:-8],Ns_list[i],SeqN_list[i])
		if not os.path.exists(dic+model[:-8]+'_RealTraj.png'):
			print('Traj plot and PF plots: {}'.format(model))
			if 'Straight' in model: 
				print('loading straight trajectory', file = f)
				Traj_pre = loadtxt('RealInput/StraightTraj.txt',delimiter='\t', unpack=False)
			else: Traj_pre = loadtxt('RealInput/RealTraj.txt',delimiter='\t', unpack=False)
			if 'down' in model: 
				T_interval = 5
				print('Trajectory downsampled to {}'.format(T_interval), file = f)
			else: T_interval = 1
			print('Plotting Trajectory', file = f)
			loc = Traj_plot(dic+model_list[i][:-8],Ns_list[i],SeqN_list[i],Traj_pre,T_interval)
			print('Plotting PFs', file = f)
			PF_plots(dic+model_list[i][:-8],Ns_list[i],SeqN_list[i],loc)
		f.close()
		

def IO_plot(name,Ns,SeqN,N=200,HN=200):
	net = torch.load(name+'.pth.tar')
	model = ElmanRNN_pytorch_module_v2(N,HN,N)
	model.load_state_dict(net['state_dict'])
	X_mini = net['X_mini']; Target_mini = net['Target_mini']
	input_pool = []; output_pool = [];
	criterion = nn.MSELoss(reduction='sum')
	for i in np.arange(Ns):
		input_pool.append(X_mini[i:i+1,:,:])
		output,_ = model(input_pool[i],torch.zeros(1,1,HN))
		output_pool.append(output)
	plt.figure(figsize=(15,15))
	for i in np.arange(np.minimum(Ns,3)):
		loss = criterion(output_pool[i],Target_mini[i:i+1,:,:]).item()
		plt.subplot(3,3,1+3*i)
		plt.imshow(input_pool[i].numpy()[0,:,:].T); plt.colorbar();plt.title('Input{}'.format(i+1))
		plt.subplot(3,3,2+3*i)
		plt.imshow(output_pool[i].detach().numpy()[0,:,:].T); plt.colorbar();plt.title('Output{}'.format(i+1))
		plt.subplot(3,3,3+3*i)
		plt.imshow(Target_mini[i,:,:].numpy().T); plt.colorbar(); 
		plt.title('Target{}; Final loss:{:.4f}'.format(i+1,loss))
	plt.tight_layout()
	plt.savefig(name+'_IO.png')
	print('Finish IO plot: {}'.format(name+'_IO.png'))
	plt.close()


def Traj_plot(name,Ns,SeqN,Traj_pre,T_interval=1):
	T_original = Traj_pre.shape[1];
	idx_down = np.arange(0,T_original,T_interval) # for the downsampled trajectories
	loc = Traj_pre[3:5,idx_down][:,:SeqN*Ns] # x,y location, from -1 to 1
	plt.figure(figsize=(5,5))
	plt.plot(loc[0,:],loc[1,:],'-o',markersize=3,linewidth=1)
	plt.title('Arena Trajectory')
	plt.savefig(name+'_RealTraj.png')
	print('Finish Traj plot: {}'.format(name))
	plt.close()
	return loc


def PF_plots(name,Ns,SeqN,loc,N=200,HN=200):
	net = torch.load(name+'.pth.tar')
	model = ElmanRNN_pytorch_module_v2(N,HN,N)
	model.load_state_dict(net['state_dict'])
	X_mini = net['X_mini']; hidden = net['hidden']; 
	y_hat = net['y_hat']; Target_mini = net['Target_mini']
	PF_pool = []; T = y_hat.shape[1]
	for neuron in np.arange(N):
		ac_pre = Target_mini.numpy()[:,:,neuron].reshape(Ns*SeqN,1)
		activity = np.maximum(ac_pre,0) # neuron0, time0
		PF = Grid_PF(activity, loc, 30)
		PF_pool.append(PF)
	plt.figure(figsize=(50,50))
	for i in np.arange(N):
		plt.subplot(15,15,i+1)
		plt.imshow(PF_pool[i].T); plt.xlabel('x loc'); plt.ylabel('y loc')
		plt.colorbar();plt.title('Place field of neuron {}'.format(6*i))
		plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(name+'_PF_target.png')
	plt.close()
	PF_pool = [];
	for neuron in np.arange(N):
		ac_pre = y_hat[:,T-1,:,neuron].reshape(Ns*SeqN,1)
		activity = np.maximum(ac_pre,0) # neuron0, time0
		PF = Grid_PF(activity, loc, 30)
		PF_pool.append(PF)
	plt.figure(figsize=(50,50))
	for i in np.arange(N):
		plt.subplot(15,15,i+1)
		plt.imshow(PF_pool[i].T); plt.xlabel('x loc'); plt.ylabel('y loc')
		plt.colorbar();plt.title('Place field of neuron {}'.format(6*i))
		plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(name+'_PF_yhat_t{}_relu.png'.format(T))
	plt.close()
	PF_pool = [];
	for neuron in np.arange(N):
		ac_pre = X_mini.numpy()[:,:,neuron].reshape(Ns*SeqN,1)
		activity = np.maximum(ac_pre,0) # neuron0, time0
		PF = Grid_PF(activity, loc, 30)
		PF_pool.append(PF)
	plt.figure(figsize=(50,50))
	for i in np.arange(N):
		plt.subplot(15,15,i+1)
		plt.imshow(PF_pool[i].T); plt.xlabel('x loc'); plt.ylabel('y loc')
		plt.colorbar();plt.title('Place field of neuron {}'.format(6*i))
		plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(name+'_PF_input.png')
	plt.close()
	for t in np.append(np.arange(0,T,10),T-1):
		PF_pool = [];
		for neuron in np.arange(N):
			ac_pre = hidden[:,t,:,neuron].reshape(Ns*SeqN,1)
			activity = np.maximum(ac_pre,0) # neuron0, time0
			PF = Grid_PF(activity, loc, 30)
			PF_pool.append(PF)
		plt.figure(figsize=(50,50))
		for i in np.arange(N):
			plt.subplot(15,15,i+1)
			plt.imshow(PF_pool[i].T); plt.xlabel('x loc'); plt.ylabel('y loc')
			plt.colorbar();plt.title('Place field of neuron {}'.format(i))
			plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.savefig(name+'_PF_hidden_t{}_relu.png'.format(t))
		plt.close()




def Grid_PF(activity, loc, grid):
	si = 2/grid
	PF = np.zeros((grid,grid))
	for i in np.arange(grid):
		for j in np.arange(grid):
			x_lb = i*si-1; x_ub = (i+1)*si-1
			y_lb = j*si-1; y_ub = (j+1)*si-1
			x_t = (loc[0,:]<x_ub) & (loc[0,:]>x_lb)
			y_t = (loc[1,:]<y_ub) & (loc[1,:]>y_lb)
			if np.sum(x_t&y_t) == 0:
				PF[i,j] = 0
			else:
				PF[i,j] = np.mean(activity[x_t&y_t])
	return PF


def ITO_plot_formal(X_p,Target_p,output_p,N,SeqN,Path,savename,title_list,vmax_list):
	plt.figure(figsize=(9,4))
	plt.subplot(1,3,1)
	vmax = vmax_list[0]; crange = [0,vmax/2,vmax]; 
	plt.imshow(X_p,vmin=0,vmax=vmax); plt.yticks(np.arange(0,N,N/4)); plt.xticks(np.arange(0,SeqN,SeqN/4))
	cbar = plt.colorbar(ticks=crange); cbar.ax.set_yticklabels(['{:.3f}'.format(item) for item in crange])
	plt.title(title_list[0]); 
	plt.xlabel('Time (a.u.)'); plt.ylabel('Neuron index') 
	plt.subplot(1,3,2)
	vmax = vmax_list[1]; crange = [0,vmax/2,vmax]; 
	plt.imshow(Target_p, vmin=0, vmax=vmax); plt.yticks(np.arange(0,N,N/4)); plt.xticks(np.arange(0,SeqN,SeqN/4))
	cbar = plt.colorbar(ticks=crange); cbar.ax.set_yticklabels(['{:.3f}'.format(item) for item in crange])
	plt.title(title_list[1]); 
	plt.xlabel('Time (a.u.)'); plt.ylabel('Neuron index') 
	plt.subplot(1,3,3)
	vmax = vmax_list[2]; crange = [0,vmax/2,vmax];
	plt.imshow(output_p, vmin=0, vmax=vmax); plt.yticks(np.arange(0,N,N/4)); plt.xticks(np.arange(0,SeqN,SeqN/4))
	cbar = plt.colorbar(ticks=crange); cbar.ax.set_yticklabels(['{:.3f}'.format(item) for item in crange])
	plt.title(title_list[2]); 
	plt.xlabel('Time (a.u.)'); plt.ylabel('Neuron index') 
	plt.tight_layout()
	plt.rc('text',usetex=True)
	plt.rc('font', size=12)          # controls default text sizes
	plt.rc('axes', titlesize=16)     # fontsize of the axes title
	plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
	plt.savefig(Path+savename+'.pdf')
	plt.close()




if __name__ == '__main__':
    main(dic)





