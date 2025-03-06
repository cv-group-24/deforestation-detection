import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get device to use (GPU if available, else CPU).
    
    Returns:
        torch.device: The device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Validation loss
        path: Path to save to
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        path: Path to load from
        model: Model to load into
        optimizer: Optional optimizer to load state into
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint