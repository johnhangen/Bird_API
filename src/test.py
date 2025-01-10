from configs.config import Config

import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, dataloader, criterion, config: Config):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Assuming classification: get predicted classes
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Log loss and accuracy per batch
            wandb.log(
                {"test_loss": loss.item(), 
                 "batch_accuracy": correct_predictions / total_samples}
            )

    # Calculate and log overall metrics
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    wandb.log(
        {"average_test_loss": average_loss, 
         "test_accuracy": accuracy}
    )

    return average_loss, accuracy
