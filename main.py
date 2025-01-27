from src.BirdDataset import BirdDataset
from src.model import BirdClassifierResNet
from src.train import train
from src.test import test
from configs.config import Config

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
import torchvision

def main():
    config = Config.load_config(
        "/content/Bird_API/configs/default_configs.yaml"
        )
    
    wandb.init(
        config={
            "BatchSize": config.DataLoader.BatchSize,
            "Epochs": config.Train.Epoch,
            "Pretrained": config.Model.Pretrained,
            "lr": config.Optimizer.lr,
            "Momentum": config.Optimizer.momentum
        },
        #mode="disabled",
    )
    
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
    
    dataloader = DataLoader(
            dataset,
            batch_size=config.DataLoader.BatchSize,
            shuffle=True,
            num_workers=config.DataLoader.num_workers,
            pin_memory=True,
            persistent_workers=True
            )
    
    ResNet = BirdClassifierResNet(
        num_classes=len(dataset.find_classes('/content/drive/MyDrive/Projects/data/nabirds/nabirds/images')[0]),
        pretrained=config.Model.Pretrained
    )
    ResNet = ResNet.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet.parameters(), lr=config.Optimizer.lr, momentum=config.Optimizer.momentum)

    ResNet.save(r'/content/drive/MyDrive/Projects/ResNet.pt')
    
    ResNet = train(
                model=ResNet,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                config=config
                )
    
    ResNet.save(r'/content/drive/MyDrive/Projects/ResNet.pt')
    
    _, _ = test(
                model=ResNet,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                config=config
                )
    
    ResNet.save(r'/content/drive/MyDrive/Projects/ResNet.pt')


if __name__ == "__main__":
    main()