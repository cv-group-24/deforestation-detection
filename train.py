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
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    
    # Create model
    num_classes = len(data["label_to_index"])
    model = get_model(
        config["model"]["type"], 
        num_classes
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Training variables
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # For tracking losses
    training_losses = []
    validation_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        validation_losses.append(val_metrics["loss"])
        
        # Test phase (optional during training)
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_metrics["loss"])
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {epoch_loss:.4f}")
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")

        # # Plot training history
        plot_losses(training_losses, validation_losses, test_losses)
        
        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            
            if not os.path.exists("outputs"):
                    os.makedirs("outputs")
            
            # Save best model
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join("outputs", "best_model.pth")
            )
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break
        
        # Calculate and display timing info
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_remaining = epoch_duration * remaining_epochs
        print(f"  Epoch Time: {epoch_duration:.2f}s")
        print(f"  Estimated time remaining: {estimated_time_remaining:.2f}s\n")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss")
    
    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print("\nFinal Test Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Plot training history
    plot_losses(training_losses, validation_losses, test_losses)
    
    print("Training complete!")

if __name__ == "__main__":
    main()