import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2


def sample_label(group, max_samples):
    if len(group) > max_samples:
        return group.sample(max_samples)
    else:
        return group


class AnimalsDataset(Dataset):
    def __init__(self, path_to_csv, transform, max_samples):
        super().__init__()
        self.items = pd.read_csv(path_to_csv)
        self.items = self.items[~self.items['label'].isin(['noise', 'fox'])]
        self.items = self.items.groupby('label').apply(sample_label, max_samples=max_samples).reset_index(drop=True)
        
        self.animal_mapping = {'bear': 0,
                             'bird': 1,
                             'gazelle': 2,
                             'goat': 3,
                             'hare': 4,
                             'horse': 5,
                             'marten': 6,
                             'onager': 7,
                             'snow leopard': 8,
                             'wolf': 9}
        
        self.items = self.items.to_dict('records')
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        image = cv2.imread(item['image_path'], cv2.IMREAD_UNCHANGED)
        label = torch.tensor(self.animal_mapping[item['label']])

        if self.transform:
            image = self.transform(image=image)['image']
        return image, label
