from data import generate_sequence_target_data
from model import AutoregressivePointerNet
from train import train_model
from evaluate import evaluate_model
import torch


def demo():
    seq_len = 10
    # Generar datos de entrenamiento y validación
    train_data = generate_sequence_target_data(100, seq_len)
    val_inputs, val_outputs, val_sums = generate_sequence_target_data(20, seq_len)
    val_data = (val_inputs, val_outputs, val_sums)

    # Inicializar el modelo
    model = AutoregressivePointerNet(input_dim=1, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    train_model(model, optimizer, train_data, val_data=val_data, num_epochs=15)

    # evaluate
    test_inputs, test_outputs, test_sums = generate_sequence_target_data(5, seq_len)
    mae, avg_time = evaluate_model(model, test_inputs, test_outputs, test_sums)

    print("\n=== Evaluación Final ===")
    print(f"Test MAE: {mae:.4f}")
    print(f"Avg inference time: {avg_time*1000:.2f} ms")

    # ejemplos
    model.eval()
    with torch.no_grad():
        for seq, subset, tgt in zip(test_inputs, test_outputs, test_sums):
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
            pred_indices, _ = model(x, target_len=len(subset))
            pred = pred_indices[0].tolist()
            pred_sum = sum(seq[j] for j in pred)

            print(f"Input: {seq.tolist()}")
            print(f"Ground-truth indices: {subset}, sum={tgt}")
            print(f"Predicted indices: {pred}, sum={pred_sum}")
            print("-" * 30)


if __name__ == "__main__":
    demo()
