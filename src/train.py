from configs.config import Config
import torch
import wandb
import time
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score

def train(model, dataloaders, trainset, dataset_sizes, criterion, optimizer, scheduler, config: Config):
    if config.DataLoader.deepLake:
        deeplake_classes = set([int(cls) for cls in trainset.labels.numpy()])
        cls_to_idx = {cls: i for i, cls in enumerate(sorted(deeplake_classes))}

    since = time.time()
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 555
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)

    scaler = torch.amp.GradScaler()

    for epoch in range(config.Train.Epoch):
        print(f'Epoch {epoch}/{config.Train.Epoch - 1}')
        print('-' * 10)
        f1_metric.reset()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_num_correct = 0

            for images, labels in tqdm(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)
                if config.DataLoader.deepLake:
                    labels = torch.tensor([cls_to_idx[cls.item()] for cls in labels.cpu()], dtype=torch.long, device=device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if phase == 'train' else torch.no_grad():
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * images.size(0)
                running_num_correct += torch.sum(preds == labels).item()
                f1_metric.update(preds.detach(), labels.detach())                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_num_correct / dataset_sizes[phase]
            epoch_f1 = f1_metric.compute().item()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Time: {time.time() - since}')

            if config.Train.WandB:
                if phase == 'train':
                    wandb.log(
                        {
                            "train Epoch": epoch,
                            "train Loss": epoch_loss,
                            "train Accuracy": epoch_acc,
                            "train F1 Score": epoch_f1,
                        }
                    )
                if phase == 'val':
                    wandb.log(
                        {
                            "val Epoch": epoch,
                            "val Loss": epoch_loss,
                            "val Accuracy": epoch_acc,
                            "val F1 Score": epoch_f1,
                        }
                    )

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), config.Model.Path)

        torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load(config.Model.Path))

    return model