import numpy
import torch
from helper import *
from RNN_Class import *
from IO_plot import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import scipy.signal as signal
font = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14}
matplotlib.rc('font', **font)
plt.rcParams.update({"text.usetex": True})


# Panel A: Inputs: 

Path = 'Elman_SGD/Remap_predloss/'
N,TotalSteps,T = 200,2000,100
X = np.zeros((np.int64(N),np.int64(TotalSteps))) # input N*T
tmp = np.linspace(norm.ppf(0.05),norm.ppf(0.95), np.int64(TotalSteps/2))
BellShape = norm.pdf(tmp) # Bellshape vector
template = np.concatenate((BellShape,np.zeros(np.int64(TotalSteps/2))))
X = np.zeros((np.int64(N),np.int64(TotalSteps)))# time-shifting matrix
for i in np.arange(np.int64(N)):
    X[i,:] = np.roll(template,np.int64(i*(TotalSteps/N)))

Select_T = np.arange(0,TotalSteps,np.int64(TotalSteps/T),dtype=int)
tmp = np.expand_dims((X[:,Select_T].T),axis=0)
tmp = tmp / tmp.max()
X_mini = torch.tensor(tmp.astype(np.single))
torch.save({'X_mini':X_mini, 'Target_mini': X_mini},
	Path+'Ns{}_SeqN{}_1.pth.tar'.format(N,T))

plt.figure()
plt.imshow(X_mini[0,:,:].T.numpy())
plt.colorbar()
plt.title('input 1')
plt.tight_layout()
plt.savefig(Path+'Ns{}_SeqN{}_1.png'.format(N,T))
plt.close()

np.random.seed(2); idx = np.random.permutation(np.arange(N))
X_new = X_mini[:,:,idx]
torch.save({'X_mini':X_new, 'Target_mini': X_new},
	Path+'Ns{}_SeqN{}_2.pth.tar'.format(N,T))

plt.figure()
plt.imshow(X_mini[0,:,:].T.numpy())
plt.colorbar()
plt.title('input 2')
plt.tight_layout()
plt.savefig(Path+'Ns{}_SeqN{}_2.png'.format(N,T))
plt.close()

# Combine two inputs as one env
X_mini2 = torch.stack((X_mini[0,:,:],X_new[0,:,:]),dim=0)
torch.save({'X_mini':X_mini2, 'Target_mini': X_mini2},
	Path+'Ns{}_SeqN{}_2Batch.pth.tar'.format(N,T))


# Panel B:
Path = 'Elman_SGD/Remap_predloss/'
N, SeqN, HN = 200, 100, 200
RecordN = 2000
data_name = 'Ns{}_SeqN{}_1'.format(N,SeqN)
model_name = 'Ns{}_SeqN{}_predloss_full'.format(N,SeqN)
model = ElmanRNN_tp1(N,HN,N)
model.act = nn.Sigmoid()
net = torch.load(Path+model_name+'.pth.tar',map_location='cuda:0')
data = torch.load(Path+data_name+'.pth.tar')
model.load_state_dict(net['state_dict'])
X_mini = data['X_mini'][:,:-1,:]; Target_mini = data['Target_mini'][:,1:,:] 

## Replay
SeqN = 100; N = 200; HN = 200
X_noise = np.random.normal(0,1,X_mini.shape)*X_mini.numpy().max()*0.01
X_noise = torch.tensor(X_noise.astype(np.single))
output_null,_ = model(X_noise,torch.zeros((1,1,HN)))
Target_p = Target_mini[0,:,:].T
output_p = output_null.detach().numpy()[0,:,:].T
title_list = ['Input current ($x_t$)','Target output ($y_t$)','Output after training ($\hat{y_t}$)']
vmax_list = np.array([X_noise.max(),Target_p.max(), output_p.max()])
ITO_plot_formal(X_noise,Target_p,output_p,N,SeqN,Path,model_name+'_ITO',title_list, vmax_list)

## Prediction
X_noise[:,:10,:] = X_mini[:,:10,:]
output_null,_ = model(X_noise,torch.zeros((1,1,HN)))
output_p = output_null.detach().numpy()[0,:,:].T
title_list = ['Input current ($x_t$)','Target output ($y_t$)','Output after training ($\hat{y_t}$)']
vmax_list = np.array([X_noise.max(),Target_p.max(), output_p.max()])
ITO_plot_formal(X_noise,Target_p,output_p,N,SeqN,Path,model_name+'_ITO',title_list, vmax_list)


# Panel C: Weight trace
model_name = 'Ns{}_SeqN{}_2Batch_predloss'.format(N,SeqN)
data_name = 'Ns{}_SeqN{}_2Batch'.format(N,SeqN)
model = ElmanRNN_tp1(N,HN,N)
model.act = nn.Sigmoid()
net = torch.load(Path+model_name+'.pth.tar')
data = torch.load(Path+data_name+'.pth.tar')
X_mini = data['X_mini'][:,:-1,:]
model.load_state_dict(net['state_dict'])
W = net['state_dict']['hidden_linear.weight'].cpu().numpy()

hidden = np.zeros((HN,SeqN-1))
hidden_new = np.zeros((HN,SeqN-1))
output = np.zeros((N,SeqN-1))
output_new = np.zeros((N,SeqN-1))
h = torch.zeros(1,1,HN)
for t in np.arange(SeqN-1):
	o,h = model(X_mini[0:1,t:t+1,:],h)
	hidden[:,t] = h.detach().numpy()[0,0,:]
	output[:,t] = o.detach().numpy()[0,0,:]

h = torch.zeros(1,1,HN)
for t in np.arange(SeqN-1):
	o,h = model(X_mini[1:2,t:t+1,:],h)
	hidden_new[:,t] = h.detach().numpy()[0,0,:]
	output_new[:,t] = o.detach().numpy()[0,0,:]
	

hidden_pool = [hidden,hidden_new]
idx = []
c_max = np.abs(W).max()
plt.figure(figsize=(6,3))
for i in range(2):
	idx.append(np.argsort(np.argmax(hidden_pool[i],axis=1)))
	plt.subplot(1,2,i+1)
	plt.imshow(W[idx[i],:][:,idx[i]],cmap='RdBu_r',vmin=-1.001*c_max,vmax=1.001*c_max)
	crange = [-c_max,0,c_max];cbar = plt      .colorbar(ticks=crange); 
	cbar.ax.set_yticklabels(['{:.3f}'.format(item) for item in crange])
	plt.xlabel('Presynaptic neuron'); 
	plt.ylabel('Postsynaptic neuron')
	plt.title('W: Env{} sorted'.format(i+1))
	plt.rc('font', size=12)   
	plt.rc('axes', titlesize=16, labelsize=16) 

plt.tight_layout()
plt.savefig(Path+model_name+'_W_sorted.eps')
plt.close()

offset_list = np.arange(-N,N+1)
plt.figure(figsize=(4,3))
for i in np.arange(2):
	plt_idx = np.argsort(np.argmax(hidden_pool[i],axis=1))
	trace_list = np.array([np.trace(W[plt_idx,:][:,plt_idx],offset=i) for i in offset_list])
	plt.plot(offset_list,trace_list)
	plt.xlabel('Diagonal offset'); plt.ylabel('Weight avg.')
	plt.rc('font', size=12)   
	plt.rc('axes', titlesize=16, labelsize=16) 

plt.axhline(y=0,color='r',linestyle='--')
plt.tight_layout()
plt.legend(['Env{}'.format(i+1) for i in np.arange(2)])
plt.savefig(Path+model_name+'_W_trace.pdf')
plt.close()


# Panel D: CA1 v.s. CA3 Place fields 
## load trained models and save intermediate variables

### concatenate training stages
novel = 1 # for F -> N remapping
# novel = 0 # for F -> F remapping
Path = 'Elman_SGD/Remap_predloss/'
N, SeqN = 200, 100
HN = 200
if novel:
	noise_level_list = [0.0001]
else:
	noise_level_list = [0.05,0.1,0.2,0.3,0.4,0.5]

for noise_level in noise_level_list:
	env1_name = 'Ns{}_SeqN{}_1'.format(N,SeqN)
	if novel:
		env2_name = 'Ns{}_SeqN{}_2'.format(N,SeqN)
		subpath = 'N{}T{}_relu_fixio/stages/'.format(N,SeqN)
	else:
		env2_name = 'Ns{}_SeqN{}_1_{}per'.format(N,SeqN,int(noise_level*100))
		subpath = 'N{}T{}_relu_fixio/F{}per_stages/'.format(N,SeqN,int(noise_level*100))
	print(subpath)
	model0_name = 'N{}T{}_relu_fixio/pred_relu'.format(N,SeqN)
	net = ElmanRNN_pytorch_module_v2(N,HN,N)
	net.act = nn.Sigmoid()
	net.rnn = nn.RNN(N,HN,1, batch_first=True, nonlinearity = 'relu')
	# Before exposure to env2 
	env1 = torch.load(Path+env1_name+'.pth.tar')
	X_mini = env1['X_mini'][:,:-1,:]
	env2 = torch.load(Path+env2_name+'.pth.tar')
	X_new = env2['X_mini'][:,:-1,:]
	model0 = torch.load(Path+model0_name+'.pth.tar',map_location='cuda:0')
	net.load_state_dict(model0['state_dict'])
	output_old,h_seq_old = net(X_mini,torch.zeros((1,1,HN)))
	output0,h_seq0 = net(X_new,torch.zeros((1,1,HN)))
	y_hat = np.zeros((1,SeqN-1,N))
	hidden = np.zeros((1,SeqN-1,HN))
	y_hat[0,:,:] = output_old.detach().numpy()[0,:,:]
	hidden[0,:,:] = h_seq_old.detach().numpy()[0,:,:]
	epoch,epoch_list = 0,[0]
	previous_ep = 0
	remaps = glob.glob(Path+subpath+'remap_s*.pth.tar')
	for i,remap_name in enumerate(remaps):
		remap = torch.load(remap_name)
		y_hat = np.concatenate((y_hat,remap['y_hat'][:,0,:,:]),axis=0)
		hidden = np.concatenate((hidden,remap['hidden'][:,0,:,:]),axis=0)
		epoch_list += [i+previous_ep for i in range(0,len(remap['loss']),int(len(remap['loss'])/remap['hidden'].shape[0]))]
		previous_ep += len(remap['loss'])
	torch.save({'X_mini':X_mini,'X_new':X_new,'y_hat':y_hat,'hidden':hidden,'epoch_list':epoch_list},
	Path+subpath+'HO_evol.pth.tar')

### generate CA1 and CA3 responses
for noise_level in noise_level_list:
	if novel:
		subpath = 'N200T100_relu_fixio/stages/'
	else:
		subpath = 'N200T100_relu_fixio/F{}per_stages/'.format(int(noise_level*100))
	loaded = torch.load(Path+subpath+'HO_evol.pth.tar')
	X_mini = loaded['X_mini']; X_new = loaded['X_new']
	y_hat = loaded['y_hat']; hidden = loaded['hidden']; epoch_list = loaded['epoch_list']
	err_list = []
	for i in range(len(epoch_list)):
		if i == 0:
			err = (X_mini.numpy()[0,:,:]-y_hat[i,:,:])
		else:
			err = (X_new.numpy()[0,:,:]-y_hat[i,:,:])
		err = 1/2*(err+np.abs(err))
		err_list.append(err.mean())
	print(err_list)
	err_matrix = np.zeros(y_hat.shape)
	for i in range(y_hat.shape[0]):
		if i==0: continue
		tmp = X_new.numpy()[0,:,:]-y_hat[i,:,:]
		err_matrix[i,:] = (np.abs(tmp) + tmp)/2
	CA1_rep = np.concatenate((y_hat,err_matrix),axis=2)
	CA3_rep = hidden
	torch.save({'CA1_rep':CA1_rep,'CA3_rep':CA3_rep},
	Path+subpath+'Field2_evol.pth.tar')


## Panel D: Plot fields in F and N after stablized 
subpath1 = 'N200T100_relu_fixio/F5per_stages/' # F -> F remapping
subpath2 = 'N200T100_relu_fixio/stages/' # F -> N remapping
env1 = torch.load(Path+subpath1+'Field2_evol.pth.tar')
env2 = torch.load(Path+subpath2+'Field2_evol.pth.tar')
CA1_rep = [env1['CA1_rep'][-1,:,:],env2['CA1_rep'][-1,:,:]]
CA3_rep = [env1['CA3_rep'][-1,:,:],env2['CA3_rep'][-1,:,:]]

# Plot the PF and then sort
plt.figure(figsize=(5,6))
plt.subplot(2,3,1)
CA1_MI = np.array([MI_linear(CA1_rep[0][:,i],norm=False) for i in range(CA1_rep[0].shape[1])])
CA1_thres = np.percentile(CA1_MI,50)
PF_idx = np.argwhere(CA1_MI>CA1_thres)
pk_idx = np.argsort(np.argmax(CA1_rep[0],axis=0))
pk_PF_idx = [ele for ele in pk_idx if ele in PF_idx]
X_p = CA1_rep[0].T[pk_PF_idx,:]
plt.imshow(X_p)
plt.title('CA1\nF: F sorted')
plt.subplot(2,3,2)
X_p = CA1_rep[1].T[pk_PF_idx,:]
plt.imshow(X_p)
plt.title('CA1\nN: F sorted')
plt.subplot(2,3,3)
CA1_MI = np.array([MI_linear(CA1_rep[1][:,i],norm=False) for i in range(CA1_rep[1].shape[1])])
PF_idx = np.argwhere(CA1_MI>CA1_thres)
pk_idx = np.argsort(np.argmax(CA1_rep[1],axis=0))
pk_PF_idx = [ele for ele in pk_idx if ele in PF_idx]
X_p = CA1_rep[1].T[pk_PF_idx,:]
plt.imshow(X_p)
plt.title('CA1\nN: N sorted')
plt.subplot(2,3,4)
CA3_MI = np.array([MI_linear(CA3_rep[0][:,i]) for i in range(CA3_rep[0].shape[1])])
CA3_thres = np.percentile(CA3_MI,60)
PF_idx = np.argwhere(CA3_MI>CA3_thres)
pk_idx = np.argsort(np.argmax(CA3_rep[0],axis=0))
pk_PF_idx = [ele for ele in pk_idx if ele in PF_idx]
X_p = CA3_rep[0].T[pk_PF_idx,:]
plt.imshow(X_p,vmax=X_p.max()*0.7)
plt.title('CA3\nF: F sorted')
plt.subplot(2,3,5)
X_p = CA3_rep[1].T[pk_PF_idx,:]
plt.imshow(X_p,vmax=X_p.max()*0.7)
plt.title('CA3\nN: F sorted')
plt.subplot(2,3,6)
CA3_MI = np.array([MI_linear(CA3_rep[1][:,i]) for i in range(CA3_rep[1].shape[1])])
PF_idx = np.argwhere(CA3_MI>CA3_thres)
pk_idx = np.argsort(np.argmax(CA3_rep[1],axis=0))
pk_PF_idx = [ele for ele in pk_idx if ele in PF_idx]
X_p = CA3_rep[1].T[pk_PF_idx,:]
plt.imshow(X_p,vmax=X_p.max()*0.7)
plt.title('CA3\n N: N sorted')
plt.tight_layout()
plt.savefig(Path+'FF_FN_Field2.png')
plt.savefig(Path+'FF_FN_Field2.eps')
plt.close()


## Panel E: Lap onset stats
subpath = 'N200T100_relu_fixio/stages/' # F -> N remapping
# subpath = 'N200T100_relu_fixio/F5per_stages/' # F -> F remapping
loaded = torch.load(Path+subpath+'Field2_evol.pth.tar')
CA3_rep = loaded['CA3_rep']
CA1_rep = loaded['CA1_rep']
ac_prob = 0.5 # Optimal, add activation probability for lap onset probability
dice1 = np.random.binomial(1,ac_prob,size=CA1_rep.shape)
dice2 = np.random.binomial(1,ac_prob,size=CA3_rep.shape)
for i in range(dice.shape[1]):
	dice1[:,i,:] = dice1[:,0,:]
	dice2[:,i,:] = dice2[:,0,:]

CA1_rep = np.multiply(CA1_rep,dice1)
CA3_rep = np.multiply(CA3_rep,dice2)

MI_CA1 = np.zeros((len(epoch_list)-1,CA1_rep.shape[2]))
MI_CA3 = np.zeros((len(epoch_list)-1,HN))
for i in range(len(epoch_list)-1):
	MI_CA1[i,:] = np.array([MI_linear(CA1_rep[i+1,:,neuron_idx],norm=False) for neuron_idx in range(CA1_rep.shape[2])])
	MI_CA3[i,:] = np.array([MI_linear(CA3_rep[i+1,:,neuron_idx],norm=False) for neuron_idx in range(HN)])

thres_CA1 = np.percentile(MI_CA1,80)
thres_CA3 = np.percentile(MI_CA3,80)
onset_idx_CA1 = []
for neuron_idx in range(MI_CA1.shape[1]):
	MI_laps = MI_CA1[:,neuron_idx]
	if not (MI_laps>thres_CA1).sum():
		onset_idx_CA1.append(np.nan)
		continue
	onset_idx_CA1.append(np.argwhere(MI_laps>thres_CA1)[0][0])

onset_idx_CA3 = []
for neuron_idx in range(MI_CA3.shape[1]):
	MI_laps = MI_CA3[:,neuron_idx]
	if not (MI_laps>thres_CA3).sum():
		onset_idx_CA3.append(np.nan)
		continue
	onset_idx_CA3.append(np.argwhere(MI_laps>thres_CA3)[0][0])

torch.save({'onset_idx_CA1':onset_idx_CA1,
			'onset_idx_CA3':onset_idx_CA3}, 
			Path+subpath+'Field2_onset.pth.tar')

FN = torch.load(Path+'N200T100_relu_fixio/stages/Field2_onset.pth.tar')
FF = torch.load(Path+'N200T100_relu_fixio/F5per_stages/Field2_onset.pth.tar')

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(4,6))
plt.subplot(2,2,1)
plt.title('F: CA1')
plt.hist(FF['onset_idx_CA1'],color='blue',edgecolor='black',density=True)
plt.ylim([0,0.3])
plt.subplot(2,2,2)
plt.title('N: CA1')
plt.hist(FN['onset_idx_CA1'],color='red',edgecolor='black',density=True)
plt.ylim([0,0.3])
plt.subplot(2,2,3)
plt.title('F: CA3')
plt.hist(FF['onset_idx_CA3'],color='blue',edgecolor='black',density=True)
plt.ylim([0,0.4])
plt.subplot(2,2,4)
plt.title('N: CA3')
plt.hist(FN['onset_idx_CA3'],color='red',edgecolor='black',density=True)
plt.ylim([0,0.4])
fig.text(0.5, 0.04, 'Field onset lap', ha='center')
fig.text(0, 0.5, 'Fration of place fields', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig(Path+'Field2_onset_lap.png')
plt.savefig(Path+'Field2_onset_lap.pdf')
plt.close()



## Panel F: CA1 v.s. CA3 activity correlation
noise_level1 = 0.05
noise_level2_list = [0.1,0.2,0.3,0.4,0.5]
for noise_level2 in noise_level2_list:
	subpath1 = 'N200T100_relu_fixio/F{}per_stages/'.format(int(noise_level1*100))
	subpath2 = 'N200T100_relu_fixio/F{}per_stages/'.format(int(noise_level2*100))
	env1 = torch.load(Path+subpath1+'Field2_evol.pth.tar')
	env2 = torch.load(Path+subpath2+'Field2_evol.pth.tar')
	corr_CA3 = np.array([np.corrcoef(env1['CA3_rep'][-1,:,neuron_idx], env2['CA3_rep'][-1,:,neuron_idx])[0,1] for neuron_idx in range(CA3_rep.shape[2])])
	corr_CA1 = np.array([np.corrcoef(env1['CA1_rep'][-1,:,neuron_idx], env2['CA1_rep'][-1,:,neuron_idx])[0,1] for neuron_idx in range(CA1_rep.shape[2])])
	torch.save({'corr_CA1':corr_CA1,'corr_CA3':corr_CA3},
		Path+'env{}_env{}_Field2_corr.pth.tar'.format(int(noise_level1*100),int(noise_level2*100)))

CA1_list = []
CA3_list = []
noise_level2_list = [0.1,0.2,0.3,0.4]
for noise_level2 in noise_level2_list:
	tmp = torch.load(Path+'env5_env{}_Field2_corr.pth.tar'.format(int(100*noise_level2)))
	CA1_list.append(tmp['corr_CA1'])
	CA3_list.append(tmp['corr_CA3'])

CA1_mean = [np.nanmean(ele) for ele in CA1_list]
CA1_std = [np.nanstd(ele) for ele in CA1_list]
CA1_numbers = [np.sum(~np.isnan(ele)) for ele in CA1_list]
CA1_sem = np.array(CA1_std) / np.sqrt(CA1_numbers)
CA3_mean = [np.nanmean(ele) for ele in CA3_list]
CA3_std = [np.nanstd(ele) for ele in CA3_list]
CA3_number = [np.sum(~np.isnan(ele)) for ele in CA3_list]
CA3_sem = np.array(CA3_std) / np.sqrt(CA3_number)
plt.figure(figsize=(4,4))
plt.plot(noise_level2_list,CA1_mean, marker='o', linestyle='--',label='CA1',color='red')
plt.errorbar(noise_level2_list,CA1_mean,yerr=np.array(CA1_sem)/2,fmt='o',color='red')
plt.plot(noise_level2_list,CA3_mean, marker='o', linestyle='--',label='CA3',color='blue')
plt.errorbar(noise_level2_list,CA3_mean,yerr=np.array(CA3_sem)/2,fmt='o',color='blue')
plt.legend()
plt.ylabel('Correlation (mean $\pm$ sem)'); plt.xlabel('Fog level')
plt.tight_layout()
plt.savefig(Path+'Field2_corr.png')
plt.savefig(Path+'Field2_corr.eps',transparent=True)
plt.close()





