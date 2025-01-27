from configs.config import Config

import torch
import wandb
import time
from torchmetrics.classification import MulticlassF1Score

def train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, config: Config):
    since = time.time()
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(dataloaders['train'].dataset.classes)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)

    scaler = torch.amp.GradScaler()

    for epoch in range(config.Train.Epoch):
        print(f'Epoch {epoch}/{config.Train.Epoch - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_num_correct = 0.0
            f1_metric.reset()

            for images, label in dataloaders[phase]:
                images = images.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train') and torch.autocast(device_type=device, dtype=torch.float16):
                    outputs = model(images)
                    assert outputs.dtype is torch.float16

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label)
                    assert loss.dtype is torch.float32

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update() 

                running_loss += loss.item() * images.size(0)
                running_num_correct += torch.sum(preds == label.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_num_correct.double() / dataset_sizes[phase]
            epoch_f1 = f1_metric.compute()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if config.Train.WandB:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Phase": phase,
                        "Loss": epoch_loss,
                        "Accuracy": epoch_acc,
                         "F1 Score": epoch_f1,
                    })

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model.save(r'/content/drive/MyDrive/Projects/ResNet.pt')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load(r'/content/drive/MyDrive/Projects/ResNet.pt')

    return model
