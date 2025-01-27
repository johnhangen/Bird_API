from configs.config import Config

import torch
import wandb
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer, config: Config):

    for epoch in range(config.Train.Epoch):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_size = labels.size(0)

            correct_predictions += batch_correct
            total_samples += labels.size(0)

            wandb.log(
                {"batch_loss": loss.item(), 
                 "batch_accuracy": (batch_correct/batch_size)*100.0}
            )
            print(f"batch: {i_batch}, Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        wandb.log(
            {"epoch_loss": average_loss, 
             "epoch_accuracy": accuracy, 
             "epoch": epoch + 1}
        )
        print(f"epoch: {epoch + 1}")
        print(f"epoch_accuracy: {accuracy}")

    return model
