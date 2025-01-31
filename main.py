from src.model import BirdClassifierResNet
from src.train import train
from configs.config import Config

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
import torchvision
import deeplake


def main():
    config = Config.load_config(
        "/Bird_API/configs/default_configs.yaml"
        )
    
    if config.Model.Debug:
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0)) 
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        config={
            "BatchSize": config.DataLoader.BatchSize,
            "Epochs": config.Train.Epoch,
            "Pretrained": config.Model.Pretrained,
            "lr": config.Optimizer.lr,
            "Momentum": config.Optimizer.momentum
        },
        mode="disabled",
    )
    train_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    
    val_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])


    if config.DataLoader.deepLake:
        train_dataset = deeplake.load('hub://activeloop/nabirds-dataset-train')
        val_dataset = deeplake.load('hub://activeloop/nabirds-dataset-val')

        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset),
        }

        train_dataloader = train_dataset.pytorch(
                transform ={'images': train_transform, 'labels': None},
                batch_size=config.DataLoader.BatchSize,
                shuffle=config.DataLoader.shuffle,
                num_workers=config.DataLoader.num_workers,
                pin_memory=config.DataLoader.pin_memory,
                prefetch_factor=1, 
                decode_method = {"images":"pil"}
                )
        
        val_dataloader = val_dataset.pytorch(
                transform ={'images': val_transform, 'labels': None},
                batch_size=config.DataLoader.BatchSize,
                shuffle=config.DataLoader.shuffle,
                num_workers=config.DataLoader.num_workers,
                pin_memory=config.DataLoader.pin_memory,
                prefetch_factor=1, 
                decode_method = {'images': 'pil'}
                )
        
        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
    else:
        dataset = torchvision.datasets.ImageFolder(
        root=config.DataLoader.Path,
        transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_sizes = {
            'train': train_size,
            'val': val_size
        }

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.DataLoader.BatchSize,
                shuffle=config.DataLoader.shuffle,
                num_workers=config.DataLoader.num_workers,
                pin_memory=config.DataLoader.pin_memory,
                persistent_workers=config.DataLoader.persistent_workers
                )
        
        val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.DataLoader.BatchSize,
                shuffle=config.DataLoader.shuffle,
                num_workers=config.DataLoader.num_workers,
                pin_memory=config.DataLoader.pin_memory,
                persistent_workers=config.DataLoader.persistent_workers
                )
        
        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
            }
    
    # model
    ResNet = BirdClassifierResNet(
        num_classes=555,
        pretrained=config.Model.Pretrained
    )
    #ResNet.load(config.Model.Path)
    ResNet = ResNet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet.parameters(), lr=config.Optimizer.lr, momentum=config.Optimizer.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.Scheduler.step_size, gamma=config.Scheduler.gamma)

    ResNet.save(config.Model.Path)
    
    ResNet = train(
                model=ResNet,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                trainset=train_dataset,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config
                )
    
    ResNet.save(config.Model.Path)

if __name__ == "__main__":
    main()