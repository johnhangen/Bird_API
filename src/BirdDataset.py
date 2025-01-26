import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass

@dataclass
class BirdImgObj:
    img_path: str
    img_label: int 
    class_name: str
    idx: int

class BirdDataset(Dataset):
    def __init__(self, path: str, transform=None, preload_images=False):
        self.path = os.path.join(path, "images")
        self.transform = transform
        self.preload_images = preload_images

        self.class_names = self._load_class_names()
        self.bird_imgs = self._load_image_paths()

        # Preload images if enabled
        if self.preload_images:
            print("Preloading images into RAM...")
            self.loaded_images = {obj.idx: self._load_image(obj.img_path) for obj in self.bird_imgs}

    def _load_class_names(self) -> dict:
        class_names = {}
        with open(os.path.join(self.path, "../classes.txt")) as f:
            for line in f:
                class_id, *name = line.strip().split()
                class_names[class_id] = " ".join(name)
        return class_names

    def _load_image_paths(self) -> list:
        bird_imgs = []
        with open(os.path.join(self.path, "../images.txt")) as f:
            for idx, line in enumerate(f):
                image_id, image_path = line.strip().split()
                class_id = image_path.split("/")[0]
                class_name = self.class_names.get(class_id, "Unknown")
                bird_imgs.append(BirdImgObj(img_path=image_path, img_label=int(class_id), class_name=class_name, idx=idx))
        return bird_imgs

    def _load_image(self, img_name: str):
        img_path = os.path.join(self.path, img_name)
        return Image.open(img_path).convert("RGB") 

    def __len__(self) -> int:
        return len(self.bird_imgs)

    def __getitem__(self, idx: int):
        bird_obj = self.bird_imgs[idx]
        
        if self.preload_images:
            image = self.loaded_images[idx]
        else:
            image = self._load_image(bird_obj.img_path)

        label = bird_obj.img_label

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = BirdDataset(
        path=r'data/nabirds/nabirds',
        transform=transform,
        preload_images=True  # Enable preloading
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count() - 1,  # Optimize worker count
        pin_memory=True,
        persistent_workers=True  # Keep workers alive
    )

    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i}: Images {images.shape}, Labels {labels.shape}")
        if i == 2:
            break

if __name__ == "__main__":
    main()
