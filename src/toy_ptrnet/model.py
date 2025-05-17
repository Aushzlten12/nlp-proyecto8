import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressivePointerNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.vt = nn.Linear(hidden_dim, 1)

    def forward(self, x, target_len):
        batch_size, seq_len, _ = x.size()
        enc_out, (h, c) = self.encoder(x)
        mask = torch.zeros(batch_size, seq_len, device=x.device)
        pointers, log_probs = [], []

        # entrada inicial al decoder: vector cero
        dec_input = torch.zeros(batch_size, x.size(-1), device=x.device)
        hx, cx = h[0], c[0]

        for _ in range(target_len):
            hx, cx = self.decoder_cell(dec_input, (hx, cx))

            query = self.W2(hx).unsqueeze(1)
            keys = self.W1(enc_out)
            scores = self.vt(torch.tanh(keys + query)).squeeze(-1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))
            log_prob = F.log_softmax(scores, dim=1)

            idx = log_prob.argmax(dim=1)
            pointers.append(idx)
            log_probs.append(log_prob[torch.arange(batch_size), idx])

            mask.scatter_(1, idx.unsqueeze(1), 1)
            dec_input = x[torch.arange(batch_size), idx]

        pointers = torch.stack(pointers, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return pointers, log_probs
