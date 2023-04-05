#  Open arena exploration test
# 1. Generate trajectories
# 2. Generate input based on sensory input and save for model training
# 3. Load trained model
# 4. model analysis and plotting



# Part 1: Generate straight trajectories
# Refer to Traj_Generate.m for the generation code


# Part 2: Generate nonlinear expansion of sensory input
# loading trajectory - version 2
Traj_pre = loadtxt("StraightTraj_v2.txt", delimiter="\t", unpack=False)
Traj_pre[1,0] = 2*np.pi
# row0: distance traveled since last wall hit
# row1: world frame angle wrt (0,0) in (0,2pi)
# row2: head direction orthogonal to the hitted wall \in (-0.5pi, 0.5pi)
# row3: x
# row4: y

loc_norm = np.vstack((np.cos(Traj_pre[0,:]*2*np.pi/np.sqrt(8)),np.sin(Traj_pre[0,:]*2*np.pi/np.sqrt(8)),
 np.cos(Traj_pre[1,:]), np.sin(Traj_pre[1,:]),
 np.cos(Traj_pre[2,:]*2), np.sin(Traj_pre[2,:]*2)))

# nonlinear random expansion
N = 200
np.random.seed(0)
W_mix = np.random.normal(size=(N,loc_norm.shape[0])) # standard normal distribution
X = W_mix.dot(loc_norm)
X = (X>=0).astype(np.int)

SeqN = 100; 
Ns = 50
tmp = np.zeros((Ns,SeqN,N)); 
for i in np.arange(Ns):
	tmp[i,:,:] = X[:,i*SeqN:(i+1)*SeqN].T

X_mini = torch.tensor(tmp.astype(np.single))

torch.save({'X_mini':X_mini, 'Target_mini': X_mini,'W_mix':W_mix},
		'InputNs{}_SeqN{}_StraightTraj_Marcus_v2.pth.tar'.format(Ns,SeqN))


