import torch

def resolve_device(device):
    if not device:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda' or None")
    return device