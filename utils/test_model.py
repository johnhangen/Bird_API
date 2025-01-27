from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch
import wandb
import matplotlib.pyplot as plt

from src.model import BirdClassifierResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(mode="disabled")

transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

dataset = torchvision.datasets.ImageFolder(
    root='/content/drive/MyDrive/Projects/data/nabirds/nabirds/images',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


ResNet = BirdClassifierResNet(
    num_classes=len(dataset.find_classes('/content/drive/MyDrive/Projects/data/nabirds/nabirds/images')[0])
    )
ResNet.load(r'/content/drive/MyDrive/Projects/ResNet.pt')
ResNet.to(device)


for test_images, test_labels in dataloader:
    sample_image = test_images.to(device)
    sample_label = test_labels

    pred = ResNet(sample_image)
    _, pred = torch.max(pred, 1)
    break

plt.imshow(sample_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu())
plt.title(f"Pred: {pred.item()}, Actual: {sample_label.item()}")
plt.savefig('test1.png')