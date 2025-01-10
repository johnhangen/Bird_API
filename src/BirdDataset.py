import os
import torch
from skimage import io, transform
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
    def __init__(self, path:str, transform=None):
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
                    img_path = self.paths[img_path],
                    img_label = img_path,
                    class_name = self.names[str(int(self.paths[img_path].split('/')[0]))],
                    idx = i
                )
            )
        
        del self.paths

        self.path = self.path + r'\images'

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

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.path,
            self.bird_imgs[idx].img_path
        )

        image = io.imread(img_name)

        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        label_name = self.bird_imgs[idx].class_name
        label = list(self.names.values()).index(label_name)

        if self.transform:
            sample = self.transform({'image': image, 'label': label})
            image, label = sample['image'], sample['label']

        return image, label
    
class Rescale(object):

    def __init__(self, output_size) -> None:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        img = transform.resize(image, self.output_size, anti_aliasing=True)

        return {'image': img, 'label': label}
    
class ToTensor(object):

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': label}


def main():
    dataset = BirdDataset(
                        path=r'data\nabirds\nabirds',
                        transform=transforms.Compose([
                            Rescale(128),
                            ToTensor()
                            ])
                          )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for i, (images, labels) in enumerate(dataloader):
        print(labels)
        print(f"Batch {i}:")
        print(f" - Images shape: {images.shape}")
        print(f" - Labels shape: {labels.shape}")
        
        if i == 2:
            break   

if __name__ == "__main__":
    main()