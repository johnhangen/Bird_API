from src.BirdDataset import BirdDataset, Rescale, ToTensor
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
        mode="disabled",
    )
    
    dataset = BirdDataset(
                        path="/content/drive/MyDrive/Projects/data/nabirds/nabirds",
                        transform=transforms.Compose([
                            Rescale(128),
                            ToTensor()
                            ])
                          )

    dataloader = DataLoader(
                        dataset, 
                        batch_size=config.DataLoader.BatchSize, 
                        shuffle=config.DataLoader.shuffle,
                        num_workers=config.DataLoader.num_workers
                        )
    
    ResNet = BirdClassifierResNet(
        num_classes=dataset.num_bird_class(),
        pretrained=config.Model.Pretrained
    )
    ResNet = ResNet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet.parameters(), lr=config.Optimizer.lr, momentum=config.Optimizer.momentum)

    ResNet.save(r'C:\Users\jthan\OneDrive\Desktop\2024\Projects\Bird_API\model')
    
    ResNet = train(
                model=ResNet,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                config=config
                )
    
    _, _ = test(
                model=ResNet,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                config=config
                )
    
    ResNet.save()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    main()