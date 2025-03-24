# Import the visualization functions
from utils.visualization import visualize_predictions, visualize_batch_predictions
import torch

import argparse
import json
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config.default_config import DEFAULT_CONFIG
from data.dataloader import create_data_loaders
from models.helpers import get_model

from utils.helpers import set_seed, get_device, save_checkpoint
from utils.metrics import evaluate_model
from utils.visualization import plot_losses

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the ForestNet dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            # Update default config with user config
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    # Set random seed
    set_seed(config["training"]["seed"])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    data = create_data_loaders(config)
    test_loader = data["test_loader"]
    
    
    # Create model
    num_classes = len(data["label_to_index"])
    model = get_model(
        config["model"]["type"], 
        num_classes
    )
    model = model.to(device)

    # Load your best model
    model.load_state_dict(torch.load("outputs/best_model_yaren.pth")["model_state_dict"])
    model.eval()

    # Class colors: Black = Background, Red = Deforestation
    class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # 1. Visualize individual samples from the dataset
    visualize_predictions(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        output_dir="outputs/predictions",
        num_samples=10,
        class_colors=class_colors
    )

    # 2. Visualize a batch from the test loader
    visualize_batch_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir="outputs/batch_predictions",
        batch_idx=0,  # First batch
        max_samples=8,
        class_colors=class_colors
    )
    

if __name__ == "__main__":
    main()