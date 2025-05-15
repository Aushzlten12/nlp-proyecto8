import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from data import get_dataloader
from model import PointerNetwork

# Hiperparámetros
epochs = 10
batch_size = 64
seq_len = 20
max_val = 20
max_subset = 3
vocab_size = max_val + 1
embed_dim = 64
hidden_dim = 128
lr = 1e-3
alpha = 0.1
train_size = 10000
val_size = 2000

# DataLoaders
train_loader = get_dataloader(
    batch_size, train_size, seq_len, max_val, max_subset, shuffle=True
)
val_loader = get_dataloader(
    batch_size, val_size, seq_len, max_val, max_subset, shuffle=False
)

# Modelo y optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointerNetwork(seq_len, vocab_size, embed_dim, hidden_dim).to(device)
opt = Adam(model.parameters(), lr=lr)

for ep in range(1, epochs + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    for xs, idxs, Ss in train_loader:
        xs, idxs, Ss = xs.to(device), idxs.to(device), Ss.to(device)
        ptr = model(xs, Ss, max_output_len=idxs.size(1))  # [B, K, seq_len]
        # Cross-entropy
        B, K, L = ptr.size()
        ce = 0
        for t in range(K):
            ce += F.nll_loss(torch.log(ptr[:, t, :] + 1e-8), idxs[:, t])
        ce = ce / K
        # Término de suma
        expected_sum = (ptr * xs.unsqueeze(1).float()).sum(dim=(1, 2))
        ls = F.l1_loss(expected_sum, Ss.float())
        loss = ce + alpha * ls
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xs, idxs, Ss in val_loader:
            xs, idxs = xs.to(device), idxs.to(device)
            ptr = model(xs, Ss, max_output_len=idxs.size(1))
            pred = ptr.argmax(dim=-1)
            for b in range(xs.size(0)):
                if set(pred[b].tolist()) == set(idxs[b].tolist()):
                    correct += 1
                total += 1
    t1 = time.time()
    print(
        f"Epoch {ep:02d} - Train Loss: {train_loss/len(train_loader):.3f} "
        f"Val Acc: {correct/total:.2%} "
        f"Time: {t1-t0:.1f}s"
    )

# Guardar modelo
torch.save(model.state_dict(), "best_ptrnet.pth")
print("Modelo guardado en best_ptrnet.pth")
