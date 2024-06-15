import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.util import save_model

def train_epoch(model, device, loader, optimizer, criterion):
    model.to(device)
    model.train()

    epoch_loss, epoch_acc = 0.0, 0.0
    num_elements = 0

    for idx, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += (labels == logits.argmax(dim=1)).sum()
        num_elements += images.shape[0]


    epoch_loss /= num_elements
    epoch_acc /= num_elements

    return epoch_loss, epoch_acc


def test_epoch(model, device, loader, criterion):
    model.to(device)
    model.eval()

    epoch_loss, epoch_acc = 0.0, 0.0
    num_elements = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            epoch_loss += loss.item()
            epoch_acc += (labels == logits.argmax(dim=1)).sum()
            num_elements += images.shape[0]
        
    epoch_loss /= num_elements
    epoch_acc /= num_elements

    return epoch_loss, epoch_acc


def train_fn(model, device, dataloaders, optimizer, criterion, epochs, model_path):
    print('Initialization of the training process...')
    best_val_acc = 0.0
    for i_epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, device, dataloaders['train'], optimizer, criterion)
        val_loss, val_acc = test_epoch(model, device, dataloaders['val'], criterion)

        if val_acc > best_val_acc:
            print(f'logging model from accuracy {best_val_acc} to {val_acc}')
            best_val_acc = val_acc
            save_model(model, model_path)


