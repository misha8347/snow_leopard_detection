import torch
import yaml

def save_model(model, path):
    torch.save(
        {
            'model_state_dict': model.state_dict()
        }, 
        path
    )


def load_yaml(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg