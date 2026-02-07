import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_o = nn.Linear(input_size + hidden_size, hidden_size)

    def lstm_forward(self, x, h_prev, c_prev):
        x_full = torch.cat([h_prev, x], dim=0)
        f_t = F.sigmoid(self.w_f(x_full))  # Forget
        i_t = F.sigmoid(self.w_i(x_full))  # Input
        c_t = F.tanh(self.w_c(x_full))  # Candidate
        o_t = F.sigmoid(self.w_o(x_full))  # Out

        c_new = f_t * c_prev + i_t * c_t
        h_new = o_t * F.tanh(c_new)
        return h_new, c_new

    def seq_forward(self, x):
        T = x.size(0)

        h_fwd = torch.zeros((T, self.hidden_size), device=x.device)
        h = torch.zeros((self.hidden_size,), device=x.device)
        c = torch.zeros((self.hidden_size,), device=x.device)
        for i in range(T):
            h, c = self.lstm_forward(x[i], h, c)
            h_fwd[i] = h

        h_bck = torch.zeros((T, self.hidden_size), device=x.device)
        h = torch.zeros((self.hidden_size,), device=x.device)
        c = torch.zeros((self.hidden_size,), device=x.device)
        for i in reversed(range(T)):
            h, c = self.lstm_forward(x[i], h, c)
            h_bck[i] = h

        return torch.cat([h_fwd, h_bck], dim=1)

    def forward(self, x):
        B, T, _ = x.shape

        outputs = []
        for b in range(B):
            output = self.seq_forward(x[b])
            outputs.append(output)

        return torch.stack(outputs)
