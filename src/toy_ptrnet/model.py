import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNetwork(nn.Module):
    """
    Pointer Network toy para sumar subconjuntos.
    Conditioned on S (suma objetivo).
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Encoder bidireccional
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # Inicializar decoder
        self.init_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        self.decoder_cell = nn.LSTMCell(embed_dim, hidden_dim)

        # Atención (pointer)
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # Proyección de la suma
        self.sum_proj = nn.Linear(1, embed_dim)

    def forward(self, x: torch.LongTensor, S: torch.LongTensor, max_output_len: int):

        B, seq_len = x.size()
        # Encode
        emb = self.embedding(x)
        emb = self.dropout(emb)
        enc_out, (h_n, c_n) = self.encoder(emb)
        enc_out = self.dropout(enc_out)
        # Init decoder
        h0 = torch.cat([h_n[-2], h_n[-1]], dim=1)
        c0 = torch.cat([c_n[-2], c_n[-1]], dim=1)
        h = self.init_linear(h0)
        c = self.init_linear(c0)
        # sum embedding
        s_emb = self.sum_proj(S.unsqueeze(-1).float())
        inp = self.dropout(s_emb)
        # Decode steps
        ptr_dists = []
        for _ in range(max_output_len):
            h, c = self.decoder_cell(inp, (h, c))
            # Atención como puntero
            w1 = self.W1(enc_out)
            w2 = self.W2(h).unsqueeze(1)
            u = self.v(torch.tanh(w1 + w2)).squeeze(-1)
            a = F.softmax(u, dim=1)
            ptr_dists.append(a)
            # embed pointed element
            idx = a.argmax(dim=1)
            inp = emb.gather(1, idx.view(B, 1, 1).expand(-1, -1, emb.size(2)))
            inp = inp.squeeze(1)
            inp = self.dropout(inp)
        return torch.stack(ptr_dists, dim=1)
