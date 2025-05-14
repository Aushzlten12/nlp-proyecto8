import random
import torch.nn as nn
import torch.nn.functional as F


class PtrNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()


def obtener_ejemplo(n=10, min_val=1, max_val=9, max_subset=4):
    x = [random.randint(min_val, max_val) for _ in range(n)]
    # Elegimos aleatoriamente un subconjunto de tamaño k
    k = random.randint(1, max_subset)
    idxs = random.sample(range(n), k)
    S = sum(x[i] for i in idxs)
    return x, idxs, S


x, idxs, S = obtener_ejemplo(n=12)
print("x =", x)
print("subset indices =", idxs, "sum =", S)
