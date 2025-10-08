# This is a set of RNN classes defined by Y.C. (cyusi@ucsd.edu)


import numpy as np
from scipy.stats import norm
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class ElmanRNN:
    """
    Elman RNN: numpy based
    tanh and softmax nonlinearity
    """

    def __init__(self, W, V, N, bptt_depth):
        self.N = N
        self.bptt_depth = bptt_depth
        self.W = W
        self.V = V

    def forward_propagation(self, X, Target, h0):
        h = np.zeros((np.int(self.N), np.int(self.bptt_depth)))
        h[:, 0] = h0
        y_hat = np.zeros((np.int(self.N), np.int(self.bptt_depth)))
        for i in np.arange(1, np.int(self.bptt_depth)):  # loop from 1:latest
            h[:, i] = np.tanh(np.matmul(self.W, h[:, i - 1]) + X[:, i])
            y_hat[:, i] = softmax(np.matmul(self.V, h[:, i]))
        delta = y_hat - Target
        L = np.sum(np.power(delta, 2))
        return h, y_hat, L

    def bptt(self, X, Target, h0):
        h, y_hat, L = self.forward_propagation(X, Target, h0)
        delta = y_hat - Target
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], np.int(self.bptt_depth)))
        dLdh = np.zeros((h.shape[0], 1, np.int(self.bptt_depth)))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], np.int(self.bptt_depth)))
        for t in np.arange(1, np.int(self.bptt_depth - 1))[
            ::-1
        ]:  # loop from the latest-1:1
            dLdo = delta[:, t] * y_hat[:, t] * (1 - y_hat[:, t])
            dLdV[:, :, t] = np.matmul(
                dLdo.reshape(np.int(self.N), 1), (h[:, t].reshape(1, np.int(self.N)))
            )
            dLdh[:, :, t] = np.matmul(
                np.matmul(self.W.T, np.diag(1 - np.power(h[:, t + 1], 2))),
                dLdh[:, :, t + 1],
            ) + np.matmul(self.V.T, dLdo.reshape(np.int(self.N), 1))
            dLdW[:, :, t] = np.matmul(
                np.matmul(np.diag(1 - np.power(h[:, t], 2)), dLdh[:, :, t]),
                h[:, t - 1].reshape(1, np.int(self.N)),
            )
        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        return dLdV_accum, dLdW_accum


class ElmanRNN_pytorch_module(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_pytorch_module, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnn = nn.RNN(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act = nn.Softmax(2)  # activation function

    def forward(self, x, h0):
        batch_size = x.size(0)
        # h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # if cuda: h0 = h0.cuda()
        z, hidden = self.rnn(x, h0)
        out = self.act(self.linear(z))
        return out, hidden


class ElmanRNN_pytorch_module_v2(nn.Module):
    # v2: change the 2nd output variable to rnn output:
    # hidden activity (BatchN * SeqN * HiddenN)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_pytorch_module_v2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnn = nn.RNN(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act = nn.Softmax(2)  # activation function

    def forward(self, x, h0):
        z, _ = self.rnn(x, h0)
        out = self.act(self.linear(z))
        return out, z


class SymAsymRNN(nn.Module):
    """
    W = λ S + (1-λ) A, with λ ∈ [0,1] via sigmoid(lambda_raw).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, init_lambda: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # symmetric + antisymmetric bases
        self.S = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.A = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.orthogonal_(self.S)
        nn.init.orthogonal_(self.A)

        # set λ(0) = init_lambda via logit
        lam0 = float(init_lambda)
        lam0 = min(max(lam0, 1e-6), 1.0 - 1e-6)  # clamp away from {0,1}
        logit = math.log(lam0 / (1.0 - lam0))
        self.lambda_raw = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        # input/output
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

        # keep these for Main_s4 compatibility
        self.rnn = None
        self.act = nn.Softmax(2)
        self.tanh = torch.tanh

    def effective_lambda(self):
        return torch.sigmoid(self.lambda_raw)

    def effective_W(self):
        S_sym = 0.5 * (self.S + self.S.T)
        A_asym = 0.5 * (self.A - self.A.T)
        lam = self.effective_lambda()
        return lam * S_sym + (1.0 - lam) * A_asym

    def forward(self, x, h0):
        W = self.effective_W()
        T = x.size(1)
        h = h0[0] if h0.ndim == 3 else h0
        outputs, hiddens = [], []
        for t in range(T):
            h = self.tanh(self.input_linear(x[:, t, :]) + h @ W.T)
            y = self.act(self.linear(h))
            outputs.append(y)
            hiddens.append(h)
        return torch.stack(outputs, 1), torch.stack(hiddens, 1)


class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.act = nn.Softmax(2)  # activation functions

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            z[:, t, :] = ht
        out = self.act(self.linear3(z))
        return out, ht


class ElmanRNN_pred(nn.Module):
    # output prediction value: y_{t+1|t} at time t
    # y_{t+1|t} = sigma(W h_{t+1})
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_pred, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act = nn.Softmax(2)  # activation functions
        self.xavier_var = 0.005015

        # initialize input/output weights to xavier
        self.init_xavier_io_weights()

        # initialize hidden weights
        # self.init_xavier_hidden_weights()
        # self.init_orthog_weights()
        # self.init_identity_weights()
        # self.init_tridiag_weights()
        # self.init_cyclic_tridiag_weights()
        # self.init_shift_weights()
        # self.init_cyclic_shift_weights()
        # self.init_mh_weights()
        # self.init_cyclic_mh_weights()
        # self.init_cyclic_weights()
        # self.init_custom_mh()
        # self.init_unitary_circulant()

        # initialize hidden weights with noise
        # self.init_mh_noise()
        self.init_shift_noise()
        # self.init_orthog_noise()
        # self.init_cyclic_tridiag_noise(
        #    diag_val=1, off_diag_val=-1, noise_mode="structure", noise_scale=1.# 5)

    def init_xavier_io_weights(self):
        """initialize input/output weights using Xavier initialization."""
        # input weights
        nn.init.xavier_uniform_(
            self.input_linear.weight, gain=nn.init.calculate_gain("tanh")
        )
        nn.init.zeros_(self.input_linear.bias)

        # output weights
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

        # set hidden weight bias to zero
        nn.init.zeros_(self.hidden_linear.bias)

    def init_xavier_hidden_weights(self):
        """initialize hidden weights to Xavier as baseline."""
        nn.init.xavier_uniform_(self.hidden_linear.weight)

        W = self.hidden_linear.weight.data
        variance = W.var().item()
        print(f"Xavier variance: {variance: .6f}")

    def init_orthog_weights(self):
        # hidden weights (H->H) with orthogonal init
        print("Initializing hidden_linear with orthogonal weights")
        W = torch.empty(self.hidden_dim, self.hidden_dim)
        nn.init.orthogonal_(W)
        current_var = W.var().item()
        print(f"Orthogonal variance before scaling: {current_var:.6f}")

        scaling_factor = (self.xavier_var / current_var) ** 0.5
        W_scaled = W * scaling_factor
        print(f"Orthogonal variance before scaling: {W_scaled.var().item():.6f}")

        with torch.no_grad():
            self.hidden_linear.weight.copy_(W_scaled)

        # Check orthogonality
        WTW = W.T @ W
        identity = torch.eye(W.shape[0])
        is_orthogonal = torch.allclose(WTW, identity, atol=1e-5)

        print("Is orthogonal:", is_orthogonal)
        WTW = W_scaled.T @ W_scaled
        lambda_max = WTW.diag().mean().item()  # estimate of alpha^2
        approx_identity = torch.eye(W_scaled.shape[0]) * lambda_max
        is_scaled_orthogonal = torch.allclose(WTW, approx_identity, atol=1e-5)

        print("Is scaled orthogonal:", is_scaled_orthogonal)

        # optional: check
        """
        orth = W_scaled @ W_scaled.T
        print(
            "Is orthogonal?",
            torch.allclose(orth, torch.eye(W_scaled.shape[0]), atol=1e-5),
        )
        """

    def init_mh_weights(self):
        """Generate a 1D Mexican hat vector of given size with the maximum along the main diagonal. Non-cyclic"""
        with torch.no_grad():
            size = self.hidden_dim
            sigma = size / 10
            center = size // 2

            # 1D mexican hat kernel
            x = np.arange(-center, size - center)
            mh = (1 - (x**2) / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
            mh = torch.tensor(mh, dtype=torch.float32)

            # construct Toeplitz matrix
            mh_matrix = torch.empty((size, size))
            for i in range(size):
                for j in range(size):
                    idx = i - j + center
                    if 0 <= idx < size:
                        mh_matrix[i, j] = mh[idx]
                    else:
                        mh_matrix[i, j] = 0  # outside bounds

            # compute empirical variance
            mean = mh_matrix.mean()
            var = ((mh_matrix - mean) ** 2).mean()
            print(f"Variance prior to scaling: {var:.6f}")

            # scale to match target variance
            scale = (self.xavier_var / var).sqrt()
            mh_matrix *= scale
            print(f"Variance after scaling: {mh_matrix.var().item():.6f}")

            self.hidden_linear.weight.copy_(mh_matrix)

    def init_cyclic_mh_weights(self):
        """Initialize weights with a cyclic Maxican hat kernel scaled to Xavier variance."""
        with torch.no_grad():
            size = self.hidden_dim
            sigma = size / 10

            # 1D cyclic Mexican hat kernel
            x = np.arange(size)
            x = np.minimum(x, size - x)  # cyclic distance from center
            mh = (1 - (x**2) / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
            mh = torch.tensor(mh, dtype=torch.float32)

            # construct cyclic Toeplitz matrix
            mh_matrix = torch.empty((size, size))
            for i in range(size):
                for j in range(size):
                    dist = min(abs(i - j), size - abs(i - j))  # cyclic distance
                    mh_matrix[i, j] = mh[dist]

            # compute empirical variance
            mean = mh_matrix.mean()
            var = ((mh_matrix - mean) ** 2).mean()
            print(f"Variance prior to scaling: {var:.6f}")

            # scale to match target variance
            scale = (self.xavier_var / var).sqrt()
            mh_matrix *= scale
            print(f"Variance after scaling: {mh_matrix.var().item():.6f}")

            self.hidden_linear.weight.copy_(mh_matrix)

    def init_identity_weights(self, value=1):
        print("Initializing hidden weights to scaled identity matrix")
        alpha = (self.hidden_dim * self.xavier_var) ** 0.5
        with torch.no_grad():
            self.hidden_linear.weight.copy_(torch.eye(self.hidden_dim) * alpha)
        print(f"Hidden weight variance: {self.hidden_linear.weight.var().item():.6f}")

    def init_custom_mh(self):
        sigma = 10
        with torch.no_grad():
            n = self.hidden_dim
            m = n / 2

            # generate 1d mexican hat vector
            c = np.array(
                [
                    (2 / (np.sqrt(3 * np.pi) * sigma))
                    * (1 - ((i - m) / sigma) ** 2)
                    * np.exp(-((i - m) ** 2) / (2 * sigma**2))
                    for i in range(n)
                ]
            )

            # generate circulant matrix
            C = np.stack([np.roll(c, i) for i in range(n)], axis=1)
            var = np.var(C)
            print("Mexican hat matrix variance:", var)

            # scale to Xavier variance
            C *= np.sqrt(self.xavier_var / var)
            print("Variance after scaling:", np.var(C))

            # assign to layer weights
            self.hidden_linear.weight.copy_(torch.tensor(C, dtype=torch.float32))

    def init_cyclic_weights(self):
        """Initialize a square weight matrix as a cyclic (circulant) matrix."""
        with torch.no_grad():
            size = self.hidden_dim  # assume square: (hidden_dim, hidden_dim)
            base_row = torch.randn(size)  # random 1D vector

            # build circulant matrix by shifting base_row
            circulant_matrix = torch.stack(
                [base_row.roll(shifts=i) for i in range(size)]
            )

            # compute empirical variance
            mean = circulant_matrix.mean()
            var = ((circulant_matrix - mean) ** 2).mean()
            print(f"Variance prior to scaling: {var:.6f}")

            # normalize to control spectral norm (optional but useful)
            circulant_matrix /= torch.norm(circulant_matrix, p=2)

            # scale to match target variance
            scale = (self.xavier_var / var).sqrt()
            circulant_matrix *= scale
            print(f"Variance after scaling: {circulant_matrix.var().item():.6f}")

            self.hidden_linear.weight.copy_(circulant_matrix)

    def init_unitary_circulant(self):
        """Initialize hidden weights with real-valued circulant matrix whose eigenvalues lie on the complex unit circle (unitary), scaled to Xavier variance."""
        with torch.no_grad():
            # random complex phases on a unit circle
            theta = np.random.uniform(0, 2 * np.pi, self.hidden_dim)
            eigenvalues = np.exp(1j * theta)

            # inverse FFT to get base rows (complex vector)
            base_row = np.fft.ifft(eigenvalues)

            # take real part (gives real circulant matrix)
            base_row_real = base_row.real

            # build circulant matrix
            C = np.stack(
                [np.roll(base_row_real, i) for i in range(self.hidden_dim)], axis=0
            )

            # scale to Xavier variance
            var = np.var(C)
            print("variance before scaling:", var)
            scale = np.sqrt(self.xavier_var / var)
            C *= scale
            print("Variance after scaling:", np.var(C))

            # copy to hidden weights
            self.hidden_linear.weight.copy_(torch.tensor(C, dtype=torch.float32))

    def init_cyclic_mh_weights(self):
        """Initialize weights with a cyclic Maxican hat kernel scaled to Xavier variance."""
        with torch.no_grad():
            size = self.hidden_dim
            sigma = size / 10

            # 1D cyclic Mexican hat kernel
            x = np.arange(size)
            x = np.minimum(x, size - x)  # cyclic distance from center
            mh = (1 - (x**2) / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
            mh = torch.tensor(mh, dtype=torch.float32)

            # construct cyclic Toeplitz matrix
            mh_matrix = torch.empty((size, size))
            for i in range(size):
                for j in range(size):
                    dist = min(abs(i - j), size - abs(i - j))  # cyclic distance
                    mh_matrix[i, j] = mh[dist]

            # compute empirical variance
            mean = mh_matrix.mean()
            var = ((mh_matrix - mean) ** 2).mean()
            print(f"Variance prior to scaling: {var:.6f}")

            # scale to match target variance
            scale = (self.xavier_var / var).sqrt()
            mh_matrix *= scale
            print(f"Variance after scaling: {mh_matrix.var().item():.6f}")

            self.hidden_linear.weight.copy_(mh_matrix)

    def init_mh_noise(self, noise_mode="structure", noise_scale=1.0):
        """Initialize hidden layers with a noisy Mexican hat circulant matrix scaled to match Xavier variance. Noise scale 0.1-0.5*base_variance is standard.
        - noise_scale 0.1-0.25: prioritizes structure/interpretability
        - 0.25 - 0.5: balanced structure + performance
        - 0.5 - 1+: emphasis on trainability (ignore structure)
        """
        with torch.no_grad():
            # mexican hat kernel
            sigma = self.hidden_dim / 10

            # 1D cyclic Mexican hat kernel
            x = np.arange(self.hidden_dim)
            x = np.minimum(x, self.hidden_dim - x)  # cyclic distance from center
            mh = (1 - (x**2) / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
            mh = torch.tensor(mh, dtype=torch.float32)

            # construct cyclic Toeplitz matrix
            mh_matrix = torch.empty((self.hidden_dim, self.hidden_dim))
            for i in range(self.hidden_dim):
                for j in range(self.hidden_dim):
                    dist = min(abs(i - j), self.hidden_dim - abs(i - j))
                    mh_matrix[i, j] = mh[dist]

            # compute variance
            print(f"Variance before noise: {mh_matrix.var().item():.6f}")

            # add noise
            print(f"Scaling noise to {noise_mode} at {noise_scale}")
            if noise_mode == "structure":
                noise_std = noise_scale * mh_matrix.std()
            elif noise_mode == "xavier":
                noise_std = noise_scale * (self.xavier_var**0.5)
            elif noise_mode == "fixed":
                noise_std = 0.1
            else:
                raise ValueError("noise_mode must be 'structure', 'xavier', or 'fixed'")
            noise = torch.randn_like(mh_matrix) * noise_std
            mh_matrix += noise

            # re-compute variance
            mean = mh_matrix.mean()
            var = ((mh_matrix - mean) ** 2).mean()
            print(f"Variance after noise (before scaling): {var:.6f}")

            # scale to Xavier variance
            scale = (self.xavier_var / var).sqrt()
            mh_matrix *= scale
            print(f"Variance after scaling: {mh_matrix.var().item():.6f}")

            self.hidden_linear.weight.copy_(mh_matrix)

    def init_shift_noise(
        self, off_diag_val=1, noise_scale=1.0, noise_mode="structure", cyclic=False
    ):
        # create shift matrix (upper off-diagonal)
        W = torch.zeros((self.hidden_dim, self.hidden_dim))
        idx = torch.arange(self.hidden_dim - 1)
        W[idx, idx + 1] = off_diag_val

        if cyclic:
            W[-1, 0] = off_diag_val  # wrap-around element for cyclic shift
        print(f"Using {'cyclic' if cyclic else 'standard'} shift matrix")
        print(f"Variance before noise {W.var().item():.6f}")

        # add noise
        print(f"Adding noise_mode of {noise_mode} with noise_scale of {noise_scale}")
        if noise_mode == "structure":
            noise_std = noise_scale * W.std()
        elif noise_mode == "xavier":
            noise_std = noise_scale * (self.xavier_var**0.5)
        elif noise_mode == "fixed":
            noise_std = 0.05
        else:
            raise ValueError("noise_mode must be 'structure', 'xavier', or 'fixed'")

        noise = torch.randn_like(W) * noise_std
        W_noisy = W + noise

        # compute variance
        mean = W_noisy.mean()
        var = ((W_noisy - mean) ** 2).mean()
        print(f"Variance before scaling: {var:.6f}")

        # scale
        scale = (self.xavier_var / var).sqrt()
        W_scaled = W_noisy * scale
        print(f"Variance after scaling: {W_scaled.var().item():.6f}")

        # copy into layer
        with torch.no_grad():
            self.hidden_linear.weight.copy_(W_scaled)

    def init_orthog_noise(self, noise_scale=0.01):
        """Initialize orthogonal matrix with noise"""
        with torch.no_grad():
            # create orthogonal matrix
            W = torch.empty(self.hidden_dim, self.hidden_dim)
            nn.init.orthogonal_(W)

            # add Gaussian noise
            noise = torch.randn_like(W) * noise_scale
            W_noisy = W + noise

            # re-orthogonalize using QR decomposition
            Q, _ = torch.qr(W_noisy)  # Q is orthogonal

            # scale to match Xavier variance
            var = Q.var()
            scale = (self.xavier_var / var).sqrt()
            W_scaled = Q * scale
            print(f"Variance after orthog and scaling: {W_scaled.var().item():.6f}")

            self.hidden_linear.weight.copy_(W_scaled)

    def init_cyclic_tridiag_noise(
        self, diag_val=1, off_diag_val=1, noise_scale=0.1, noise_mode="xavier"
    ):
        W = torch.zeros((self.hidden_dim, self.hidden_dim))

        # set main diagonal
        W[torch.arange(self.hidden_dim), torch.arange(self.hidden_dim)] = diag_val

        # set upper diagonal (wrap around)
        W[
            torch.arange(self.hidden_dim - 1), torch.arange(1, self.hidden_dim)
        ] = off_diag_val
        W[0, self.hidden_dim - 1] = off_diag_val
        print(
            f"Created cyclic tridiagonal matrix with diag={diag_val}, off-diag={off_diag_val}"
        )

        # add noise
        print(f"Adding {noise_mode} noise scaled to {noise_scale}")
        if noise_mode == "structure":
            noise_std = noise_scale * W.std()
        elif noise_mode == "xavier":
            noise_std = noise_scale * (self.xavier_var**0.5)
        elif noise_mode == "fixed":
            noise_std = noise_scale
        else:
            raise ValueError("noise_mode must be 'structure', 'xavier', or 'fixed'")
        noise = torch.randn_like(W) * noise_std
        W_noisy = W + noise

        # rescale to Xavier variance
        var = W_noisy.var()
        scale = (self.xavier_var / var).sqrt()
        W_scaled = W_noisy * scale
        print(f"Variance after scaling: {W_scaled.var().item():.6f}")

        with torch.no_grad():
            self.hidden_linear.weight.copy_(W_scaled)

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN - 1):
            # ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            # htp1 = self.tanh(self.hidden_linear(ht))  # predict one-step ahead
            ht = self.input_linear(x[:, t, :]) + self.hidden_linear(ht)
            htp1 = self.hidden_linear(ht)  # predict one-step ahead

            z[:, t + 1, :] = htp1  # z_{t+1} = h_{t+1|t}
        out = self.act(self.linear3(z))  # y_{t+1|t}
        return out, ht


class ElmanRNN_tp1(nn.Module):
    # output prediction value: y_{t+1|t} at time t
    # y_{t+1|t} = sigma(W h_{t+1})
    # change the size of output
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_tp1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.act = nn.Softmax(2)  # activation functions

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            htp1 = self.tanh(self.hidden_linear(ht))  # predict one-step ahead
            z[:, t, :] = htp1  # z_{t} = h_{t+1|t}
        out = self.act(self.output_linear(z))  # y_{t+1|t}
        return out, ht


class ElmanRNN_pred_v2(nn.Module):
    # output prediction value: y_{t+1|t} at time t
    # y_{t+1|t} = sigma(W h_{t+1})
    # Change second output to htp1_seq (batchN*SeqN*hiddenN)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_pred_v2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.act = nn.Softmax(2)  # activation functions

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        htp1_seq = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN - 1):
            ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            htp1 = self.tanh(self.hidden_linear(ht))  # predict one-step ahead
            z[:, t + 1, :] = htp1  # z_{t+1} = h_{t+1|t}
        out = self.act(self.linear3(z))  # y_{t+1|t}
        return out, z


class ElmanRNN_pred_feedback(nn.Module):
    # Need to be exactly auto-encoder structure
    # output prediction value: y_{t+1|t} at time t
    # y_{t+1|t} = sigma(W h_{t+1}+U y_{t|t})
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_pred_feedback, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.act = nn.Softmax(2)  # activation functions

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN - 1):
            ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            yt = self.act(self.linear3(ht))
            htp1 = self.tanh(
                self.input_linear(yt) + self.hidden_linear(ht)
            )  # predict one-step ahead
            z[:, t + 1, :] = htp1  # z_{t+1} = h_{t+1|t}
        out = self.act(self.linear3(z))
        return out, ht


class ElmanRNN_pred_v3(nn.Module):
    # output prediction value: y_{t+k|t} at time t
    # y_{t+k|t} = sigma(W h_{t+k}s)
    def __init__(self, input_dim, hidden_dim, output_dim, k):
        super(ElmanRNN_pred_v3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.tanh = nn.Tanh()
        self.act = nn.Softmax(2)  # activation functions
        self.k = k

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN - self.k):
            ht = self.tanh(self.input_linear(x[:, t, :]) + self.hidden_linear(ht))
            for j in range(self.k):
                ht = self.tanh(self.hidden_linear(ht))  # predict one-step ahead
            z[:, t + self.k, :] = ht  # z_{t+k} = h_{t+k|t}
        out = self.act(self.linear3(z))  # y_{t+k|t}
        return out, ht


class ElmanRNN_sparse(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super(ElmanRNN_sparse, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.dropout = dropout
        self.rnn = nn.RNN(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act = nn.Softmax(2)  # activation function

    def forward(self, x, h0):
        batch_size, SeqN, _ = x.shape
        ht = h0
        z = torch.zeros(x.shape).to(x.device)
        mask = (
            torch.zeros(1, batch_size, self.hidden_dim)
            .bernoulli_(1 - self.dropout)
            .to(x.device)
        )
        for t in range(SeqN):
            # print('output shape: {}; when t={}'.format(ht.shape,t))
            ht_pre, _ = self.rnn(x[:, t : t + 1, :], ht)
            ht_pre = ht_pre.transpose(1, 0)
            # print('RNN direct output shape: {}; mask shape: {}'.format(ht_pre.shape, mask.shape))
            ht = ht_pre * mask
            z[:, t, :] = ht[0, :, :]
        out = self.act(self.linear(z))
        return out, ht


class ElmanRNN_v3(nn.Module):
    # To compare with RateRNN
    # Different output calculation: remove the softmax function
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanRNN_v3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnn = nn.RNN(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, h0):
        z, hidden = self.rnn(x, h0)
        out = self.linear(z)
        return out, z


class RateRNN(nn.Module):
    """
    Define rate network used in Kim PNAS 2019
    No Dale's law; not trainable synaptic constant; batch size=1
    MODEL INPUT:
        input_dim:
        hidden_dim
        output_dim:
        tau: membrane constant
        dt: Euler integration constant
    INPUT:
        x: input signal: BatchN*SeqN*HiddenN
        h: initial hidden unit states: BatchN*LayerN*HiddenN
    OUTPUT:
        out: output signal
        hidden: hidden unit states across time
    """

    def __init__(self, input_dim, hidden_dim, output_dim, tau, dt):
        super(RateRNN, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dt = dt
        self.linear_ho = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_ih = nn.Linear(self.input_dim, self.hidden_dim, bias=None)
        self.linear_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=None)
        self.act = nn.Sigmoid()  # activation function

    def forward(self, x, h):
        batch_size, SeqN, _ = x.shape
        # hidden = []
        hidden = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            # hidden.append(h[0:1,0,:])
            h_ = (
                (1 - self.dt / self.tau) * h
                + self.dt
                / self.tau
                * (self.linear_hh(self.act(h)) + self.linear_ih(x[:, t : t + 1, :]))
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h = h_
            hidden[:, t, :] = h[:, 0, :]
        # hidden = torch.stack(hidden,1)
        out = self.linear_ho(self.act(hidden))
        return out, hidden


class RateRNN_v2(nn.Module):
    """
    Based on RateRNN_v2:
        Apply softmax transformation to the output node
    MODEL INPUT:
        input_dim:
        hidden_dim
        output_dim:
        tau: membrane constant
        dt: Euler integration constant
    INPUT:
        x: input signal: BatchN*SeqN*HiddenN
        h: initial hidden unit states: BatchN*LayerN*HiddenN
    OUTPUT:
        out: output signal
        hidden: hidden unit states across time
    """

    def __init__(self, input_dim, hidden_dim, output_dim, tau, dt):
        super(RateRNN_v2, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dt = dt
        self.linear_ho = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_ih = nn.Linear(self.input_dim, self.hidden_dim, bias=None)
        self.linear_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=None)
        self.act = nn.Sigmoid()  # activation function
        self.act2 = nn.Softmax(
            2
        )  # activation function, normalization across hidden units

    def forward(self, x, h):
        batch_size, SeqN, _ = x.shape
        # hidden = []
        hidden = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            # hidden.append(h[0:1,0,:])
            h_ = (
                (1 - self.dt / self.tau) * h
                + self.dt
                / self.tau
                * (self.linear_hh(self.act(h)) + self.linear_ih(x[:, t : t + 1, :]))
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h = h_
            hidden[:, t, :] = h[:, 0, :]
        # hidden = torch.stack(hidden,1)
        out = self.act2(self.linear_ho(self.act(hidden)))
        return out, hidden


class RateRNN_v2_tp1(nn.Module):
    """
    Define rate network used in Kim PNAS 2019
    No Dale's law; not trainable synaptic constant; batch size=1
    Apply nonlinear transformation to the output node (default softmax)
    predict one step ahead
    MODEL INPUT:
        input_dim:
        hidden_dim
        output_dim:
        tau: membrane constant
        dt: Euler integration constant
    INPUT:
        x: input signal: BatchN*SeqN*HiddenN
        h: initial hidden unit states: BatchN*LayerN*HiddenN
    OUTPUT:
        out: output signal
        hidden: hidden unit states across time
    """

    def __init__(self, input_dim, hidden_dim, output_dim, tau, dt):
        super(RateRNN_v2_tp1, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dt = dt
        self.linear_ho = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_ih = nn.Linear(self.input_dim, self.hidden_dim, bias=None)
        self.linear_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=None)
        self.act = nn.Sigmoid()  # activation function
        self.act2 = nn.Softmax(
            2
        )  # activation function, normalization across hidden units

    def forward(self, x, h):
        batch_size, SeqN, _ = x.shape
        # hidden = []
        hidden = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            # hidden.append(h[0:1,0,:])
            h_ = (
                (1 - self.dt / self.tau) * h
                + self.dt
                / self.tau
                * (self.linear_hh(self.act(h)) + self.linear_ih(x[:, t : t + 1, :]))
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h_tp1 = (
                (1 - self.dt / self.tau) * h_
                + self.dt / self.tau * (self.linear_hh(self.act(h_)))
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h = h_
            hidden[:, t, :] = h_tp1[:, 0, :]
        # hidden = torch.stack(hidden,1)
        out = self.act2(self.linear_ho(self.act(hidden)))  # out_t = prediction{t+1|t}
        return out, hidden


class RateRNN_dale(nn.Module):
    """
    Define rate network used in Kim PNAS 2019
    Apply Dale's rule: W wrapped by ReLu
    Apply softmax transformation to the output node
    MODEL INPUT:
        input_dim:
        hidden_dim
        output_dim:
        tau: membrane constant
        dt: Euler integration constant
        P_inh: Percentage of inhibitory neurons
    INPUT:
        x: input signal: BatchN*SeqN*HiddenN
        h: initial hidden unit states: BatchN*LayerN*HiddenN
    OUTPUT:
        out: output signal
        hidden: hidden unit states across time
    """

    def __init__(self, input_dim, hidden_dim, output_dim, tau, dt, P_inh=0.2):
        super(RateRNN_dale, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dt = dt
        self.linear_ho = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_ih = nn.Linear(self.input_dim, self.hidden_dim, bias=None)
        self.linear_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=None)
        self.m = nn.ReLU()
        self.act = nn.Sigmoid()  # activation function
        self.act2 = nn.Softmax(2)  # activation function
        self.P_inh = P_inh
        self.inh = np.random.rand(self.hidden_dim, 1) < self.P_inh
        self.mask = np.eye(self.hidden_dim, dtype=np.float32)
        self.mask[np.where(self.inh == True)[0], np.where(self.inh == True)[0]] = -1

    def forward(self, x, h):
        batch_size, SeqN, _ = x.shape
        mask = torch.tensor(self.mask.astype(np.single)).to(x.device)
        hidden = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            # hidden.append(h[0:1,0,:])
            h_ = (
                (1 - self.dt / self.tau) * h
                + self.dt
                / self.tau
                * (
                    torch.matmul(self.m(self.linear_hh(self.act(h))), mask)
                    + self.linear_ih(x[:, t : t + 1, :])
                )
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h = h_
            hidden[:, t, :] = h[:, 0, :]
        # hidden = torch.stack(hidden,1)
        out = self.act2(self.linear_ho(self.act(hidden)))
        return out, hidden


class RateRNN_dale_v2(nn.Module):
    """
    Apply Dale's rule: W wrapped by ReLu
    Apply softmax transformation to the output node
    More distinct EI functions:
        1. extend Dale's rule to ih and ho weights
        2. external 8Hz oscillatory signals injected to inhibitory neurons
        3. inhibitory hidden neurons don't contribute to output signal
        4. inhibitory indices are at the last
        5. by default, no bias term !
    MODEL INPUT:
        input_dim:
        hidden_dim
        output_dim:
        tau: membrane constant
        dt: Euler integration constant
        theta: magnitude of oscillatory input
        P_inh: percentage of inhibitory neurons
    INPUT:
        x: input signal: BatchN*SeqN*HiddenN
        h: initial hidden unit states: BatchN*LayerN*HiddenN
    OUTPUT:
        out: output signal
        hidden: hidden unit states across time
    """

    def __init__(self, input_dim, hidden_dim, output_dim, tau, dt, theta, P_inh=0.2):
        super(RateRNN_dale_v2, self).__init__()
        # Defining some parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dt = dt
        self.P_inh = P_inh
        self.theta = theta
        self.inh_dim = np.round(self.hidden_dim * self.P_inh).astype(int)
        self.exc_dim = self.hidden_dim - self.inh_dim
        self.linear_ho = nn.Linear(self.exc_dim, self.output_dim, bias=None)
        self.linear_ih = nn.Linear(self.input_dim, self.exc_dim, bias=None)
        self.linear_hh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=None)
        self.m = nn.ReLU()
        self.act = nn.Sigmoid()  # activation function
        self.act2 = nn.Softmax(2)  # activation function
        self.inh = np.arange(self.exc_dim, self.hidden_dim)
        self.mask = np.eye(self.hidden_dim, dtype=np.float32)
        self.mask[self.inh, self.inh] = -1

    def forward(self, x, h):
        batch_size, SeqN, _ = x.shape
        mask = torch.tensor(self.mask.astype(np.single)).to(x.device)
        hidden = torch.zeros((batch_size, SeqN, self.hidden_dim)).to(x.device)
        # I_record = torch.zeros((batch_size,SeqN,self.hidden_dim)).to(x.device)
        for t in range(SeqN):
            I_recurrent = torch.matmul(self.m(self.linear_hh(self.act(h))), mask)
            I_exc = self.m(self.linear_ih(x[:, t : t + 1, :]))
            # print(I_exc.shape)
            I_inh = self.theta * (np.sin(16 * np.pi * t * self.dt) + 1)
            I_inh = np.expand_dims(np.repeat(I_inh, self.inh_dim), axis=(0, 1))
            I_inh = torch.tensor(I_inh.astype(np.single)).to(x.device)
            I = I_recurrent + torch.cat((I_exc, I_inh), dim=2)
            h_ = (
                (1 - self.dt / self.tau) * h
                + self.dt / self.tau * (I)
                + torch.randn(h.shape).to(x.device) * 0.01
            )
            h = h_
            hidden[:, t, :] = h[:, 0, :]
            # I_record[:,t,:] = I[:,0,:]
        out = self.act2(self.linear_ho(self.act(hidden[:, :, 0 : self.exc_dim])))
        return out, hidden


# add conv layer at decoding and reconstruction
class ConvRNN(nn.Module):
    """
    INPUT:
        x: BatchSize * Channel_in(=1) * TimeLength* H_in * W_in (needs to be normalized between 0 and 1)
        x_conv: BatchSize * Channel_out(=1) * H_out * W_out
        x_conv_vec: BatchSize * TimeLength * (H_out*W_out)
        z: BatchSize * TimeLength * (H_out*W_out)
    OUTPUT:
        h: 1 * (H_out*W_out)
        out: BatchSize * Channel_in(=1) * TimeLength * H_in * W_in
    """

    def __init__(self, kernel, hidden_dim, H_out, W_out):
        super(ConvRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.H_out = H_out
        self.W_out = W_out
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.act = nn.Sigmoid()
        self.rnn = nn.RNN(
            self.H_out * self.W_out,
            self.hidden_dim,
            1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.output = nn.Linear(self.hidden_dim, self.H_out * self.W_out, bias=None)

    def forward(self, x, h0):
        N, Ch, T, H_in, W_in = x.shape
        x_conv_vec = torch.zeros((N, T, self.H_out * self.W_out)).to(x.device)
        for t in range(T):
            x_conv = self.conv(x[:, :, t, :, :])
            x_conv_vec[:, t, :] = torch.reshape(x_conv, (N, 1, -1))[:, 0, :]
        z, h = self.rnn(x_conv_vec, h0)
        z = self.act(self.output(z))
        out = torch.zeros(x.shape).to(x.device)
        for t in range(T):
            out[:, :, t, :, :] = self.deconv(
                torch.reshape(z[:, t, :], (N, 1, self.H_out, self.W_out))
            )
        return out, h


class ConvRNN_v2(nn.Module):
    """
    v2: make the second output to record all ht
    INPUT:
        x: BatchSize * Channel_in(=1) * TimeLength* H_in * W_in (needs to be normalized between 0 and 1)
        x_conv: BatchSize * Channel_out(=1) * H_out * W_out
        x_conv_vec: BatchSize * TimeLength * (H_out*W_out)
        z: BatchSize * TimeLength * (H_out*W_out)
    OUTPUT:
        z: TimeLength * (H_out*W_out)
        out: BatchSize * Channel_in(=1) * TimeLength * H_in * W_in
    """

    def __init__(self, kernel, hidden_dim, H_out, W_out):
        super(ConvRNN_v2, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.H_out = H_out
        self.W_out = W_out
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.act = nn.Sigmoid()
        self.rnn = nn.RNN(
            self.H_out * self.W_out,
            self.hidden_dim,
            1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.output = nn.Linear(self.hidden_dim, self.H_out * self.W_out, bias=None)

    def forward(self, x, h0):
        N, Ch, T, H_in, W_in = x.shape
        x_conv_vec = torch.zeros((N, T, self.H_out * self.W_out)).to(x.device)
        for t in range(T):
            x_conv = self.conv(x[:, :, t, :, :])
            x_conv_vec[:, t, :] = torch.reshape(x_conv, (N, 1, -1))[:, 0, :]
        z, h = self.rnn(x_conv_vec, h0)
        z = self.act(self.output(z))
        out = torch.zeros(x.shape).to(x.device)
        for t in range(T):
            out[:, :, t, :, :] = self.deconv(
                torch.reshape(z[:, t, :], (N, 1, self.H_out, self.W_out))
            )
        return out, z


class ConvRNN_tp1(nn.Module):
    """
    *_tp1: Iterate one more step for future prediction
    INPUT:
        x: BatchSize * Channel_in(=1) * TimeLength* H_in * W_in (needs to be normalized between 0 and 1)
        x_conv: BatchSize * Channel_out(=1) * H_out * W_out
        x_conv_vec: BatchSize * TimeLength * (H_out*W_out)
        z: BatchSize * TimeLength * (H_out*W_out)
    OUTPUT:
        h: 1 * (H_out*W_out)
        out: BatchSize * Channel_in(=1) * TimeLength * H_in * W_in
    """

    def __init__(self, kernel, hidden_dim, H_out, W_out):
        super(ConvRNN_tp1, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.H_out = H_out
        self.W_out = W_out
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.input_linear = nn.Linear(self.H_out * self.W_out, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.H_out * self.W_out)
        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h0):
        N, Ch, T, H_in, W_in = x.shape
        ht = h0[0, :, :]
        z = torch.zeros((N, T, self.hidden_dim)).to(x.device)
        for t in range(T):
            x_conv = self.conv(x[:, :, t, :, :])
            ht = self.tanh(
                self.input_linear(torch.reshape(x_conv[:, 0, :, :], (N, -1)))
                + self.hidden_linear(ht)
            )
            htp1 = self.tanh(self.hidden_linear(ht))
            z[:, t, :] = htp1  # z_t = h_{t+1|t}
        z = self.act(self.output_linear(z))  # y_{t+1|t}
        out = torch.zeros(x.shape).to(x.device)
        for t in range(T):
            out[:, :, t, :, :] = self.deconv(
                torch.reshape(z[:, t, :], (N, 1, self.H_out, self.W_out))
            )
        return out, ht


# needs to be constructed
class Conv_compress(nn.Module):
    """
    Compress the 2d images into a 1d array (between 0 and 1)
    INPUT:
        x: BatchSize * Channel_in(=1) * H_in * W_in (needs to be normalized between 0 and 1)
        x_conv: BatchSize * Channel_out(=1) * H_out * W_out
    OUTPUT:
        z: BatchSize * (H_out*W_out)
        out: BatchSize * Channel_in(=1) * H_in * W_in
    """

    def __init__(self, kernel, hidden_dim, H_out, W_out):
        super(ConvRNN_tp1, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.H_out = H_out
        self.W_out = W_out
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=self.kernel, bias=False
        )
        self.input_linear = nn.Linear(self.H_out * self.W_out, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.H_out * self.W_out)
        self.act = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h0):
        N, Ch, H_in, W_in = x.shape
        x_conv = self.conv(x)
        z = torch.reshape(x_conv, (N, -1))
        out = self.deconv(x_conv)
        return out, z


def Train_SGD(X_mini, Target_mini, lr, n_epochs, net, PATH, RecordEp=1000):
    """
    Use SGD to train neural network and save nn object to PATH
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            lr: learning rate
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            PATH: dir to save model to *.pt
        OUTPUT:
            net: save and return
    """
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    params = list(net.parameters())
    print("{} parameters to optimize".format(len(params)))
    loss_list = np.ones((1, 1)) * torch.sum(Target_mini**2).item() / 2
    y_hat = np.zeros((np.int(n_epochs / RecordEp), X_mini.shape[1], X_mini.shape[2]))
    start = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        output, hidden = net(X_mini, torch.zeros(1, 1, X_mini.shape[2]))
        loss = criterion(output, Target_mini)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly
        loss_list = np.append(loss_list, loss.item())
        if epoch % RecordEp == 0:
            end = time.time()
            print(end - start)
            start = time.time()
            print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
            print("Loss: {:.4f}".format(loss.item()))
            y_hat[np.int(epoch / RecordEp), :, :] = output.detach().numpy()[0, :, :]
    torch.save(net, PATH)
    return net, loss_list, y_hat


# RNN class defined by Mia


class BaseRNN:
    """
    Base RNN: numpy based
    tanh and softmax nonlinearity
    """

    def __init__(self, N, hidden_N, bptt_depth):
        self.N = N
        self.hidden_N = hidden_N
        self.bptt_depth = bptt_depth  # number of time-steps
        self.U = np.random.uniform(-0.1, 0.1, size=(hidden_N, N))
        self.W = np.random.uniform(
            -0.1, 0.1, size=(hidden_N, hidden_N)
        )  # recurrent weight matrix
        self.V = np.random.uniform(
            -0.1, 0.1, size=(N, hidden_N)
        )  # output weight matrix
        self.b = np.random.uniform(-0.1, 0.1, size=(hidden_N))
        self.c = np.random.uniform(-0.1, 0.1, size=(N))

    def forward_propagation(self, y, h0, closed=False):
        """
        X	  : np array of size (N, bptt_depth) with the input sequence
        h0	 : np array of size (hidden_N) with starting state of h
        """
        h = np.zeros((self.hidden_N, self.bptt_depth))
        # initialize hidden state

        y_hat = np.zeros((self.N, self.bptt_depth))
        h[:, 0] = h0
        y_hat[:, 0] = y[:, 0]

        for i in range(1, self.bptt_depth):  # loop from 1:latest
            if closed:
                y_ = y_hat[:, i - 1]
            else:
                y_ = y[:, i - 1]
            u = self.W @ h[:, i - 1] + self.b + (self.U @ y_)
            h[:, i] = np.tanh(u)
            o = self.V @ h[:, i] + self.c
            y_hat[:, i] = np.tanh(o)
        delta = y_hat[1::] - y[1::]  # loss vector
        L = np.sum(np.power(delta, 2))  # MSE loss function (scalar)
        return h, y_hat, L

    def softmax_jacobian(self, x):
        # J = -1 * np.outer(x, x)
        # J += np.diag(x)
        J = np.diag(1 - np.power(x, 2))
        return J

    def gradient(self, y, h0):
        raise NotImplementedError

    def update_weights(self, dLdV, dLdW, dLdU, dLdb, dLdc, lr=0.05):
        self.V += -lr * dLdV
        self.W += -lr * dLdW
        self.U += -lr * dLdU
        self.b += -lr * dLdb
        self.c += -lr * dLdc
        return


class BPTTRNN(BaseRNN):
    """
    BPTT RNN: numpy based
    Inherits from BaseRNN
    """

    def __init__(self, N, hidden_N, bptt_depth):
        super().__init__(N, hidden_N, bptt_depth)

    def gradient(self, y, h0):
        h, y_hat, L = self.forward_propagation(y, h0)
        delta = y_hat - y
        L = np.sum(np.power(delta, 2))

        # initialize jacobians used in error computation
        dLdU = np.zeros((self.U.shape[0], self.U.shape[1], self.bptt_depth))
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], self.bptt_depth))
        dLdh = np.zeros((h.shape[0], self.bptt_depth))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], self.bptt_depth))
        dLdb = np.zeros((self.b.shape[0], self.bptt_depth))
        dLdc = np.zeros((self.c.shape[0], self.bptt_depth))

        # last element
        dLdo = self.softmax_jacobian(y_hat[:, -1]) @ delta[:, -1]
        dLdV[:, :, -1] = np.outer(dLdo, h[:, -1])
        dLdh[:, -1] = self.V.T @ dLdo
        dLdW[:, :, -1] = np.outer(
            np.diag(1 - np.power(h[:, -1], 2)) @ dLdh[:, -1], h[:, -2]
        )
        dLdU[:, :, -1] = np.outer(dLdh[:, -1], y[:, -1])
        dLdb[:, -1] = np.diag(1 - np.power(h[:, -1], 2)) @ dLdh[:, -1]
        dLdc[:, -1] = dLdo

        for t in range(self.bptt_depth - 2, 0, -1):
            dLdo = self.softmax_jacobian(y_hat[:, t]) @ delta[:, t]
            dLdV[:, :, t] = np.outer(dLdo, h[:, t])
            dLdh[:, t] = (
                self.W.T @ dLdh[:, t + 1] @ np.diag(1 - np.power(h[:, t + 1], 2))
                + self.V.T @ dLdo
            )
            dLdW[:, :, t] = np.outer(
                (np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]), h[:, t - 1]
            )
            dLdU[:, :, t] = np.outer(dLdh[:, t], y[:, t])
            dLdb[:, t] = np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]
            dLdc[:, t] = dLdo

        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        dLdU_accum = np.sum(dLdU, 2)
        dLdb_accum = np.sum(dLdb, 1)
        dLdc_accum = np.sum(dLdc, 1)
        return L, dLdV_accum, dLdW_accum, dLdU_accum, dLdb_accum, dLdc_accum


# Modified recirculation-trained Elman network
class LocalRNN(BaseRNN):
    """
    Modified-recircuation based RNN: numpy based
    Inherits from BaseRNN
    """

    def __init__(self, N, hidden_N, bptt_depth):
        super().__init__(N, hidden_N, bptt_depth)

    def gradient(self, y, h0):
        h, y_hat, L = self.forward_propagation(y, h0)
        delta = y_hat - y
        L = np.sum(np.power(delta, 2))

        # initialize jacobians used in error computation
        dLdU = np.zeros((self.U.shape[0], self.U.shape[1], self.bptt_depth))
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], self.bptt_depth))
        dLdh = np.zeros((h.shape[0], self.bptt_depth))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], self.bptt_depth))
        dLdb = np.zeros((self.b.shape[0], self.bptt_depth))
        dLdc = np.zeros((self.c.shape[0], self.bptt_depth))

        for t in range(1, self.bptt_depth):
            dLdo = self.softmax_jacobian(y_hat[:, t]) @ delta[:, t]
            dLdV[:, :, t] = np.outer(dLdo, h[:, t])
            dLdh[:, t] = self.U @ dLdo  # truncated
            dLdW[:, :, t] = np.outer(
                (np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]), h[:, t - 1]
            )
            dLdU[:, :, t] = np.outer(dLdh[:, t], y[:, t])
            dLdb[:, t] = np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]
            dLdc[:, t] = dLdo

        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        dLdU_accum = np.sum(dLdU, 2)
        dLdb_accum = np.sum(dLdb, 1)
        dLdc_accum = np.sum(dLdc, 1)
        return L, dLdV_accum, dLdW_accum, dLdU_accum, dLdb_accum, dLdc_accum


# Modified recirculation-trained Elman network
class PredRec(BaseRNN):
    """
    Predictive-recircuation RNN: numpy based
    Uses the auxillary-loss implementation, rather than assuming V = U.T
    Inherits from BaseRNN
    """

    def __init__(self, N, hidden_N, bptt_depth):
        super().__init__(N, hidden_N, bptt_depth)

    def forward_propagation(self, y, h0):
        """
        X	  : np array of size (N, bptt_depth) with the input sequence
        h0	 : np array of size (hidden_N) with starting state of h
        """
        h = np.zeros((self.hidden_N, self.bptt_depth))
        # initialize hidden state
        h[:, 0] = self.U @ y[:, 0]

        y_hat = np.zeros((self.N, self.bptt_depth))
        y_hat[:, 0] = y[:, 0]

        for i in np.arange(1, self.bptt_depth):  # loop from 1:latest
            u = self.W @ h[:, i - 1] + self.b  # closed-loop recurrence
            h[:, i] = np.tanh(u)
            o = self.V @ h[:, i] + self.c
            y_hat[:, i] = softmax(o)
        delta = y_hat[1::] - y[1::]  # loss vector
        L = np.sum(np.power(delta, 2))  # MSE loss function (scalar)
        return h, y_hat, L

    def gradient(self, y, h0):
        h = np.zeros((self.hidden_N, self.bptt_depth))
        h[:, 0] = self.U @ y[:, 0]
        y_hat = np.zeros((self.N, self.bptt_depth))
        y_hat[:, 0] = y[:, 0]

        # initialize jacobians used in error computation
        dLdU = np.zeros((self.U.shape[0], self.U.shape[1], self.bptt_depth))
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], self.bptt_depth))
        dLdh = np.zeros((h.shape[0], self.bptt_depth))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], self.bptt_depth))
        dLdb = np.zeros((self.b.shape[0], self.bptt_depth))
        dLdc = np.zeros((self.c.shape[0], self.bptt_depth))

        for t in range(1, self.bptt_depth):
            u_tilde = self.W @ h[:, t - 1] + self.b
            h_tilde = np.tanh(u_tilde)

            u = self.U @ y[:, t]
            h[:, t] = np.tanh(u)
            o = self.V @ h[:, t] + self.c
            y_hat[:, t] = softmax(o)

            u_hat = self.U @ y_hat[:, t]
            h_hat = np.tanh(u_hat)

            # output weights
            dLdo = self.softmax_jacobian(y_hat[:, t]) @ (y[:, t] - y_hat[:, t])
            dLdV[:, :, t] = -1 * np.outer(dLdo, h[:, t])
            dLdc[:, t] = -1 * dLdo

            # inputs weights
            dLdU[:, :, t] = -1 * np.outer((h[:, t] - h_hat), y[:, t])

            # recurrent weights
            dLdh[:, t] = h[:, t] - h_tilde
            dLdW[:, :, t] = -1 * np.outer(
                (np.diag(1 - np.power(h_tilde, 2)) @ dLdh[:, t]), h[:, t - 1]
            )
            dLdb[:, t] = -1 * np.diag(1 - np.power(h_tilde, 2)) @ dLdh[:, t]

        delta = y - y_hat
        L = np.sum(np.power(delta, 2))

        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        dLdU_accum = np.sum(dLdU, 2)
        dLdb_accum = np.sum(dLdb, 1)
        dLdc_accum = np.sum(dLdc, 1)
        return L, dLdV_accum, dLdW_accum, dLdU_accum, dLdb_accum, dLdc_accum
