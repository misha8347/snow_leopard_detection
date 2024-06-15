import torch
import torch.nn as nn
import click
from addict import Dict
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
import os

from src.datasets import AnimalsDataset
from src.models import DinoVisionTransformerClassifier
from src.trainers import train_fn
from src.utils import load_yaml

@click.command()
@click.argument('cfg_path', type=click.Path(), default='src/configs/config.yaml')
def main(cfg_path: str):
    cfg = Dict(load_yaml(cfg_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DinoVisionTransformerClassifier(cfg.model.num_classes, cfg.model.num_features, 
                                            cfg.model.s, cfg.model.m)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate) 
    criterion = nn.CrossEntropyLoss()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        ToTensorV2()
    ])
    df_train = AnimalsDataset(cfg.df_train.path_to_data, transform)
    df_val = AnimalsDataset(cfg.df_val.path_to_data, transform)
    
    train_loader = DataLoader(df_train, batch_size=cfg.train_loader.batch_size, 
                              shuffle=cfg.train_loader.shuffle, pin_memory=cfg.train_loader.pin_memory,
                              num_workers=cfg.train_loader.num_workers)
    
    val_loader = DataLoader(df_val, batch_size=cfg.val_loader.batch_size, 
                              shuffle=cfg.val_loader.shuffle, pin_memory=cfg.val_loader.pin_memory,
                              num_workers=cfg.val_loader.num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    os.makedirs('checkpoints', exist_ok=True)
    train_fn(model, device, dataloaders, optimizer, criterion, cfg.train_fn.epochs, cfg.train_fn.model_path)

if __name__ == '__main__':
    main()