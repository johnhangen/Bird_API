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

def main():
    config = Config.load_config(
        "/content/Bird_API/configs/default_configs.yaml"
        )
    
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
    
    #Create datasets
    dataset = torchvision.datasets.ImageFolder(
      root='/content/drive/MyDrive/Projects/data/nabirds/nabirds/images',
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
    val_size = (len(dataset) - train_size)//2
    test_size = (len(dataset) - train_size)//2
    dataset_sizes = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.DataLoader.BatchSize,
            shuffle=True,
            num_workers=config.DataLoader.num_workers,
            pin_memory=True,
            persistent_workers=True
            )
    
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.DataLoader.BatchSize,
            shuffle=True,
            num_workers=config.DataLoader.num_workers,
            pin_memory=True,
            persistent_workers=True
            )
    
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.DataLoader.BatchSize,
            shuffle=True,
            num_workers=config.DataLoader.num_workers,
            pin_memory=True,
            persistent_workers=True
            )
    
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
    
    # model
    ResNet = BirdClassifierResNet(
        num_classes=len(dataset.find_classes('/content/drive/MyDrive/Projects/data/nabirds/nabirds/images')[0]),
        pretrained=config.Model.Pretrained
    )
    ResNet.load(r'/content/drive/MyDrive/Projects/ResNet.pt')
    ResNet = ResNet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet.parameters(), lr=config.Optimizer.lr, momentum=config.Optimizer.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.Sch, gamma=0.1)

    ResNet.save(r'/content/drive/MyDrive/Projects/ResNet.pt')
    
    ResNet = train(
                model=ResNet,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config
                )
    
    ResNet.save(r'/content/drive/MyDrive/Projects/ResNet.pt')

if __name__ == "__main__":
    main()