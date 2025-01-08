from src.BirdDataset import BirdDataset, Rescale, ToTensor
from src.model import BirdClassifierResNet
from src.train import train
from configs.config import Config

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

def main():
    config = Config.load_config(
        "configs\default_configs.yaml"
        )
    
    dataset = BirdDataset(
                        path=config.DataLoader.Path,
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
    
    ResNet = train(
                model=ResNet,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                config=config
                )
    
    ResNet.save()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()