import torch

from data import generate_sequence_target_data
from model import AutoregressivePointerNet
from evaluate import evaluate_model


def train_model(model, optimizer, data, val_data=None, num_epochs=10):
    train_inputs, train_outputs, train_sums = data

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for seq, subset in zip(train_inputs, train_outputs):
            x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
            y = torch.LongTensor(subset)

            optimizer.zero_grad()
            _, log_probs = model(x, target_len=len(y))
            loss = -log_probs[0].sum()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_inputs)

        if val_data is not None:
            val_inputs, val_outputs, val_sums = val_data
            mae, avg_time = evaluate_model(model, val_inputs, val_outputs, val_sums)
            print(
                f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f} — MAE: {mae:.4f} — Time: {avg_time*1000:.2f} ms"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")
