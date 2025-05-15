# eval.py
import time
import torch
from data import get_dataloader
from model import PointerNetwork

# --- Configuración (idéntica a train.py) ---
seq_len = 20
max_val = 20
max_subset = 3
batch_size = 64
vocab_size = max_val + 1
embed_dim = 64
hidden_dim = 128
model_path = "best_ptrnet.pth"

# --- Carga del modelo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointerNetwork(
    seq_len=seq_len, vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

test_loader = get_dataloader(
    batch_size=batch_size,
    num_examples=200,
    seq_len=seq_len,
    max_val=max_val,
    max_subset=max_subset,
    shuffle=False,
)


correct = total = 0
times = []
with torch.no_grad():
    for xs, idxs, Ss in test_loader:
        xs, idxs, Ss = xs.to(device), idxs.to(device), Ss.to(device)
        t0 = time.time()
        ptr = model(xs, Ss, max_output_len=idxs.size(1))  # [B, K, seq_len]
        times.append((time.time() - t0) / xs.size(0))
        pred = ptr.argmax(dim=-1)  # [B, K]
        for b in range(xs.size(0)):
            if set(pred[b].tolist()) == set(idxs[b].tolist()):
                correct += 1
            total += 1

print(f"Test Accuracy: {correct/total:.2%}")
print(f"Avg inference time per sequence: {sum(times)/len(times)*1000:.2f} ms")
