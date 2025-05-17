import time
import torch


def evaluate_model(model, inputs, outputs, target_sums):
    """
    Evalúa MAE y tiempo medio de inferencia usando la longitud real de cada subconjunto.
    """
    model.eval()
    total_error = 0.0
    total_time = 0.0
    n = len(inputs)

    with torch.no_grad():
        for seq, subset, tgt in zip(inputs, outputs, target_sums):
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
            length = len(subset)

            start = time.time()
            pred_indices, _ = model(x, target_len=length)
            elapsed = time.time() - start

            pred = pred_indices[0].tolist()
            pred_sum = sum(seq[j] for j in pred)

            total_error += abs(pred_sum - tgt)
            total_time += elapsed

    mae = total_error / n
    avg_time = total_time / n
    return mae, avg_time
