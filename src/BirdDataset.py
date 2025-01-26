import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass

@dataclass
class BirdImgObj:
    img_path: str
    img_label: str
    class_name: str
    idx: int

class BirdDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.transform = transform
        self.paths = {}
        self.names = {}
        self.bird_imgs = []

        self._load_class_names()
        self._load_image_paths()
        
        for i, img_path in enumerate(self.paths):
            self.bird_imgs.append(
                BirdImgObj(
                    img_path=self.paths[img_path],
                    img_label=img_path,
                    class_name=self.names[str(int(self.paths[img_path].split('/')[0]))],
                    idx=i
                )
            )

        del self.paths
        self.path = os.path.join(self.path, 'images')

    def _load_class_names(self) -> None:
        with open(os.path.join(self.path, 'classes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = pieces[0]
                self.names[class_id] = ' '.join(pieces[1:])

    def _load_image_paths(self) -> None:
        with open(os.path.join(self.path, 'images.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                image_id = pieces[0]
                self.paths[image_id] = pieces[1]

    def __len__(self) -> int:
        return len(self.bird_imgs)

    def num_bird_class(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple:
        img_name = os.path.join(self.path, self.bird_imgs[idx].img_path)
        image = Image.open(img_name).convert('RGB')

        label_name = self.bird_imgs[idx].class_name
        label = list(self.names.values()).index(label_name)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = BirdDataset(
        path=r'data\nabirds\nabirds',
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    for i, (images, labels) in enumerate(dataloader):
        print(labels)
        print(f"Batch {i}:")
        print(f" - Images shape: {images.shape}")
        print(f" - Labels shape: {labels.shape}")
        
        if i == 2:
            break

if __name__ == "__main__":
    main()
