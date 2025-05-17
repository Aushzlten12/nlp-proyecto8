import numpy as np


def generate_sequence_target_data(num_samples=1000, seq_len=10, max_val=20):
    """
    Genera un dataset de secuencias de enteros, sus subconjuntos y la suma objetivo.
    """
    inputs, outputs, target_sums = [], [], []
    for _ in range(num_samples):
        seq = np.random.randint(1, max_val, size=seq_len)
        num_choices = np.random.randint(1, seq_len)
        subset_indices = sorted(
            np.random.choice(seq_len, size=num_choices, replace=False).tolist()
        )
        target_sum = int(seq[subset_indices].sum())

        inputs.append(seq)
        outputs.append(subset_indices)
        target_sums.append(target_sum)

    return inputs, outputs, target_sums
