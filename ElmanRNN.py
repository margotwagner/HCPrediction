import torch.nn as nn
import numpy as np, torch
import torch.nn.functional as F


class ElmanRNN_circulant(nn.Module):
    """
    Drop-in sibling of ElmanRNN_pytorch_module_v2:
      - same IO shape + second output is hidden sequence (Batch, Seq, H)
      - tanh by default; supports act override (relu/none)
      - hidden->hidden is circular convolution by learnable kernel
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, rnn_act: str = "tanh"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_act = rnn_act

        # I/O layers (keep exact structure)
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act_output = nn.Softmax(2)  # match v2 default

        # Hidden recurrence: circulant via Conv1d
        self.hh_circ = CirculantHH(hidden_dim)

        # Nonlinearity
        if self.rnn_act == "relu":
            self.h_act = nn.ReLU()
        elif self.rnn_act == "linear":
            self.h_act = nn.Identity()
        else:
            self.h_act = nn.Tanh()

    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        """
        x:  (B, T, N)
        h0: (1, B, H)   # match your caller; we’ll use h = h0[0]
        returns: (out, z) where z is hidden sequence (B, T, H) like *_v2
        """
        B, T, _ = x.shape
        h = h0[0] if h0.ndim == 3 else h0  # (B,H)
        z_seq = torch.zeros(B, T, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            pre = self.input_linear(x[:, t, :]) + self.hh_circ(h)
            h = self.h_act(pre)
            z_seq[:, t, :] = h

        out = self.act_output(self.output_linear(z_seq))
        return out, z_seq


class CirculantHH(nn.Module):
    """
    Hidden->Hidden = circular 1D convolution by a learnable kernel (first row).
    Supports short kernels (banded circulant) by choosing kernel_size = K <= H.
    Maintains Frobenius-norm standardization via projection.
    """

    def __init__(self, hidden_dim: int, target_fro: float = None):
        super().__init__()
        self.H = int(hidden_dim)
        # default target Frobenius norm (Xavier-for-tanh-like variance): sqrt(H/3)
        if target_fro is None:
            target_fro = ((self.H * self.H) * (1.0 / (3.0 * self.H))) ** 0.5
        self.target_fro = float(target_fro)

        # start with full-length kernel of tiny random values
        K = self.H
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=K,
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.zero_()  # safe default until init_from_row0 is called

    def _minimal_circular_window(self, row0: np.ndarray, tol: float = 1e-8):
        """
        Return (window, start_idx). start_idx is the index in [0..H-1] where the window begins.
        Finds the shortest circularly-contiguous window covering all |row0| > tol.
        """
        r = np.asarray(row0, dtype=np.float32).copy()
        H = self.H
        if r.size != H:
            raise ValueError("row0 must have length H for auto-compress")

        nz = np.flatnonzero(np.abs(r) > tol)
        if nz.size == 0:
            raise ValueError("row0 is effectively all zeros under the given tolerance.")

        # handle singleton support correctly
        if nz.size == 1:
            start = int(nz[0])
            K = 1
            window = np.array([r[start]], dtype=np.float32)
            return window, start

        # General case (>=2 taps): minimal closing arc = complement of largest circular gap
        srt = np.sort(nz)
        gaps = np.diff(np.r_[srt, srt[0] + H])  # circular gaps
        j = int(np.argmax(gaps))  # largest gap ends at srt[j]
        start = int((srt[j] + gaps[j]) % H)  # window starts after the largest gap
        K = H - int(gaps[j])

        window = np.array([r[(start + i) % H] for i in range(K)], dtype=np.float32)
        return window, start

    def init_from_row0(
        self, row0: np.ndarray, tol: float = 1e-8, edge_trim: bool = True
    ):
        """
        Accepts:
          - length H vector, possibly banded with wrap -> auto-compress to minimal K
          - length K<=H short kernel -> use directly
        Kernel is flipped before assignment because Conv1d implements correlation.
        """

        r = np.asarray(row0, dtype=np.float32).reshape(-1)
        if r.size == self.H:
            rK, start = self._minimal_circular_window(r, tol=tol)
        elif r.size < self.H:
            rK, start = r, 0
        else:
            raise ValueError(f"row0 length {r.size} > H={self.H}")

        # --- optional edge-trim of the compressed band ---
        if edge_trim and rK.size > 1:
            left = 0
            while left < rK.size - 1 and abs(rK[left]) <= tol:
                left += 1
            right = rK.size - 1
            while right > left and abs(rK[right]) <= tol:
                right -= 1
            if right >= left:
                if left > 0 or right < rK.size - 1:
                    rK = rK[left : right + 1]
                    start = (start + left) % self.H

        # roll so that the original lag-0 (index 0 in row0) sits at position 0
        # In terms of the compressed window starting at `start`, lag-0 is at offset:
        offset0 = (-start) % self.H
        if rK.size <= self.H and offset0 < rK.size:
            rK = np.roll(rK, -offset0)
        # (if lag-0 wasn’t inside the compressed band, leaving as-is is fine)

        K = int(rK.size)

        # safe-guard K = 0 case
        if K < 1:
            raise ValueError(
                f"[circulant] Derived kernel length K=0. "
                f"row0_len={len(row0)}, H={self.H}, "
                f"maybe all taps ≤ tol or edge_trim removed everything. "
                f"Try lowering tol or provide a row0 with at least one |tap|>tol."
            )

        if self.conv.kernel_size[0] != K:
            self.conv = nn.Conv1d(
                1, 1, K, padding=0, padding_mode="circular", bias=False
            )
        with torch.no_grad():
            # Convert first, then flip in Torch to avoid NumPy negative-stride views
            w = torch.as_tensor(rK, dtype=torch.float32)  # (K,)
            w = (
                torch.flip(w, dims=[0]).contiguous().view(1, 1, K)
            )  # flip for correlation
            self.conv.weight.copy_(w)
        self.project_fro_to_target()

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_t: (batch, H) -> returns (batch, H) after circular conv by kernel
        """
        x = h_t.unsqueeze(1)  # (B,1,H)
        K = int(self.conv.kernel_size[0])
        # symmetric "same" padding for stride=1, dilation=1
        pad_left = K // 2
        pad_right = K - 1 - pad_left
        x_pad = F.pad(x, (pad_left, pad_right), mode="circular")
        y = self.conv(x_pad)  # (B,1,H)
        return y.squeeze(1)  # (B,H)

    def current_kernel(self) -> torch.Tensor:
        # return unflipped kernel (first row)
        w = self.conv.weight[0, 0].detach().clone()
        return torch.flip(w, dims=[0])  # reverse back

    def project_fro_to_target(self):
        """
        Enforce ||W||_F = target_fro by scaling kernel L2 to target_fro / sqrt(H).
        Since ||W||_F = sqrt(H)*||w||_2 for circulant matrices, this keeps the implicit
        matrix’s Frobenius norm controlled during/after training. Works on old Torch.
        """
        with torch.no_grad():
            # Get unflipped convolution kernel (convolution orientation)
            w = torch.flip(self.conv.weight[0, 0], dims=[0])  # (K,)
            # L2 without torch.linalg
            w_sq_sum = (w * w).sum()
            # clamp to avoid divide-by-zero
            w_norm = torch.sqrt(torch.clamp(w_sq_sum, min=1e-12))
            target_w_norm = self.target_fro / (self.H**0.5)
            scale = target_w_norm / w_norm
            w_scaled = w * scale
            # write back (flip again because Conv1d stores correlation kernel)
            self.conv.weight[0, 0].copy_(torch.flip(w_scaled, dims=[0]))


class ElmanRNN_pytorch_module_v2(nn.Module):
    # v2: change the 2nd output variable to rnn output:
    # hidden activity (BatchN * SeqN * HiddenN)
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_act: str = "tanh"):
        super(ElmanRNN_pytorch_module_v2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnn_act = rnn_act
        # nn.RNN only accepts 'tanh' or 'relu'; for 'linear' we still
        # construct an RNN module to reuse its parameters, but we will
        # implement the recurrence manually in forward().
        nonlin = "relu" if rnn_act == "relu" else "tanh"
        self.rnn = nn.RNN(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            nonlinearity=nonlin,
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.act_output = nn.Softmax(2)  # activation function

    def forward(self, x, h0):
        # x:  (B, T, input_dim)
        # h0: (1, B, H) or (B, H)
        if self.rnn_act == "linear":
            B, T, _ = x.shape
            h = h0[0] if h0.ndim == 3 else h0  # (B, H)

            # Reuse the underlying nn.RNN parameter tensors
            W_ih = self.rnn.weight_ih_l0  # (H, input_dim)
            W_hh = self.rnn.weight_hh_l0  # (H, H)
            b_ih = getattr(self.rnn, "bias_ih_l0", None)
            b_hh = getattr(self.rnn, "bias_hh_l0", None)

            z_seq = []
            for t in range(T):
                x_t = x[:, t, :]  # (B, input_dim)
                pre = x_t @ W_ih.t() + h @ W_hh.t()
                if b_ih is not None:
                    pre = pre + b_ih
                if b_hh is not None:
                    pre = pre + b_hh
                # Linear/identity hidden: no nonlinearity
                h = pre
                z_seq.append(h.unsqueeze(1))

            z = torch.cat(z_seq, dim=1)  # (B, T, H)
        else:
            # Tanh or ReLU handled by built-in nn.RNN nonlinearity
            z, _ = self.rnn(x, h0)

        out = self.act_output(self.linear(z))
        return out, z
