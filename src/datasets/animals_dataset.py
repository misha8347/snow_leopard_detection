import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

class AnimalsDataset(Dataset):
    def __init__(self, path_to_csv, transform=None):
        super().__init__()
        self.items = pd.read_csv(path_to_csv).to_dict('records')
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        image = cv2.imread(item['image_path'], cv2.IMREAD_UNCHANGED)
        label = torch.tensor([item['label']])

        if self.transform:
            image = self.transform(image=image)['image']
        return image, label
