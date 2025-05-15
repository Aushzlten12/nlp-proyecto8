import random
import torch
from torch.utils.data import Dataset, DataLoader


class SubsetSumDataset(Dataset):
    """
    Dataset para la tarea toy de suma de subconjuntos.
    Cada muestra es:
      - x: secuencia de enteros de longitud fija
      - idxs: índices del subconjunto cuya suma es S
      - S: suma objetivo
    """

    def __init__(self, num_examples: int, seq_len: int, max_val: int, max_subset: int):
        self.seq_len = seq_len
        self.max_val = max_val
        self.max_subset = max_subset
        self.data = []
        for _ in range(num_examples):
            # Generar entrada
            x = [random.randint(1, max_val) for _ in range(seq_len)]
            k = random.randint(1, max_subset)
            idxs = random.sample(range(seq_len), k)
            S = sum(x[i] for i in idxs)
            self.data.append((x, idxs, S))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, idxs, S = self.data[i]
        return x, idxs, S


def collate_fn(batch):
    xs, idxs_list, Ss = zip(*batch)
    B = len(xs)
    seq_len = len(xs[0])
    max_k = max(len(idxs) for idxs in idxs_list)
    # Construir tensores
    x_tensor = torch.tensor(xs, dtype=torch.long)  # [B, seq_len]
    idxs_tensor = torch.zeros(B, max_k, dtype=torch.long)
    for i, idxs in enumerate(idxs_list):
        padded = idxs + [idxs[-1]] * (max_k - len(idxs))
        idxs_tensor[i] = torch.tensor(padded, dtype=torch.long)
    S_tensor = torch.tensor(Ss, dtype=torch.long)  # [B]
    return x_tensor, idxs_tensor, S_tensor


def get_dataloader(
    batch_size: int,
    num_examples: int,
    seq_len: int,
    max_val: int,
    max_subset: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = SubsetSumDataset(num_examples, seq_len, max_val, max_subset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
