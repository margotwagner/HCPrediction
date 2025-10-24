import numpy as np
import pickle, sys, gzip
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append("../")
from scipy.special import softmax
from deprecated.helper import *
from sklearn.decomposition import PCA
from Main import *
import pickle
from sklearn.preprocessing import MinMaxScaler


# load model
flc = open("../mnist6_local", "rb")
dlc = pickle.load(flc)
flc.close()

# load input
src = "data/train-images-idx3-ubyte.gz"
with gzip.open(src, "rb") as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
X = data.reshape(data.shape[0], 28 * 28).astype(np.float64)[:100]
first_ten_idx = [1, 3, 5, 7, 9, 0, 13, 15, 17, 4]

pca = PCA(n_components=68)
X_pca = pca.fit_transform(X)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_pca_scaled = scaler.fit_transform(X_pca)
data_pca = X_pca_scaled[first_ten_idx].T
x = data_pca
print(x.shape)


# panel C: reconstructed figures
X_rec_pca_lc = dlc["output_rep"][-1].T
X_rec_lc = pca.inverse_transform(X_rec_pca_lc)
plot_digits_grid(X_rec_lc)


# Fig. S7
lr = 0.05
net = dlc["net"]
grad_list = dlc["grad_list"]
U = net.U.copy()
V = net.V.copy()
r2_list = []
for i in range(1, len(grad_list[0])):
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        U.flatten(), V.T.flatten()
    )
    r2_list.append(r_value**2)
    dLdU = grad_list[2][-i]
    dLdV = grad_list[1][-i]
    U += lr * dLdU
    V += lr * dLdV

r2_list = r2_list[::-1]
slope, intercept, r_value, p_value, std_err = stats.linregress(
    U.flatten(), V.T.flatten()
)
x = np.linspace(np.min(U.flatten()), np.max(U.flatten()), 100)
y = slope * x + intercept

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Set fontsize for the entire figure
plt.rcParams.update({"font.size": 14})

# Plot imshow(net.U)
im1 = axs[0, 0].imshow(net.U, cmap="bwr")
axs[0, 0].set_title("U", fontsize=16)
axs[0, 0].set_xlabel("Columns", fontsize=14)
axs[0, 0].set_ylabel("Rows", fontsize=14)
plt.colorbar(im1, ax=axs[0, 0])

# Plot imshow(net.V.T)
im2 = axs[0, 1].imshow(net.V.T, cmap="bwr")
axs[0, 1].set_title("V^T", fontsize=16)
axs[0, 1].set_xlabel("Columns", fontsize=14)
axs[0, 1].set_ylabel("Rows", fontsize=14)
plt.colorbar(im2, ax=axs[0, 1])

# Plot linear regression plot with fitted line
axs[1, 0].plot(net.U.flatten(), net.V.T.flatten(), "o", label="Data")
axs[1, 0].plot(x, y, color="black", label="Fitted Line", linewidth=2)
axs[1, 0].set_xlabel("U elements", fontsize=14)
axs[1, 0].set_ylabel("V^T elements", fontsize=14)
axs[1, 0].set_title("Linear Regression Plot", fontsize=16)
# Calculate R-squared value
# Display R-squared and p-value
axs[1, 0].text(
    0.05,
    0.95,
    f"R-squared: {r_value**2:.4f}\n p-value: {p_value:.2e}",
    transform=axs[1, 0].transAxes,
    fontsize=12,
    verticalalignment="top",
)

# Plot the code in axs[1,1]
axs[1, 1].plot(range(len(r2_list)), r2_list)
axs[1, 1].set_xlabel("Epochs")
axs[1, 1].set_ylabel("R-squared Value")

plt.tight_layout()
plt.show()
fig.savefig("figure.eps")
