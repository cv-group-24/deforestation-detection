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
    
    num_classes = len(data["label_to_index"])
    multi_modal_size = config["model"]["multi_modal_size"]
    
    # Create CNN model
    cnn_model = None
    if (config["model"]["train_cnn"]): 
        cnn_model = get_model(
        config["model"]["type"], 
        num_classes, 
        multi_modal_size
        )
        cnn_model = cnn_model.to(device)


    # If the multi_modal_model parameter is not set then this will be set to None
    multi_modal_model = None
    if (config["model"["train_multi_modal_model"]]):
        multi_modal_model = get_model(
            config["model"]["multi_modal_model"], 
            num_classes, 
            multi_modal_size=multi_modal_size
        )

        multi_modal_model = multi_modal_model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=config["training"]["learning_rate"])
    multi_modal_optimizer = optim.Adam(multi_modal_model.parameters(), lr=config["training"]["learning_rate"])
    
    # Training variables
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["early_stopping_patience"]

    cnn_best_val_loss = float('inf')
    cnn_epochs_without_improvement = 0
    cnn_best_model_state = None

    multi_modal_best_val_loss = float('inf')
    multi_modal_epochs_without_improvement = 0
    multi_modal_best_model_state = None
    
    # For tracking losses of each model
    cnn_training_losses = []
    cnn_validation_losses = []
    cnn_test_losses = []

    # For tracking losses of each model
    multi_modal_training_losses = []
    multi_modal_validation_losses = []
    multi_modal_test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        if (cnn_model):
            cnn_model.train()

        if (multi_modal_model):
            multi_modal_model.train()

            
        # by defaults both losses are 0 but this is overwidden by the training below. Ie. it is 0 if the models are not set to be training
        cnn_running_loss = 0.0
        multi_modal_running_loss = 0.0 
        for images, multi_modal_features, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            multi_modal_features = multi_modal_features.to(device)

            # TRAIN CNN
            if (config["model"]["train_cnn"]):
                cnn_optimizer.zero_grad()
                outputs = cnn_model(images, multi_modal_features)
                loss = criterion(outputs, labels)
                loss.backward()
                cnn_optimizer.step()
                cnn_running_loss = cnn_running_loss + loss.item() * images.size(0)

            # TRAIN MULTI MODAL 
            if (multi_modal_model):
                multi_modal_optimizer.zero_grad()
                outputs = multi_modal_model(multi_modal_features)
                loss = criterion(outputs, labels)
                loss.backward()
                multi_modal_optimizer.step()
                multi_modal_running_loss = loss.item() * images.size(0)
    
        
        cnn_epoch_loss = cnn_running_loss / len(train_loader.dataset)
        cnn_training_losses.append(cnn_epoch_loss)

        multi_modal_epoch_loss = multi_modal_running_loss / len(train_loader.dataset)
        multi_modal_training_losses.append(cnn_epoch_loss)
        
        # Validation phase
        cnn_val_metrics = evaluate_model(cnn_model, val_loader, criterion, device)
        cnn_validation_losses.append(cnn_val_metrics["loss"])

        multi_modal_val_metrics = evaluate_model(multi_modal_model, val_loader, criterion, device)
        multi_modal_validation_losses.append(multi_modal_val_metrics["loss"])
        
        # Test phase (optional during training)
        cnn_test_metrics = evaluate_model(cnn_model, test_loader, criterion, device)
        cnn_test_losses.append(cnn_test_metrics["loss"])

        multi_modal_test_metrics = evaluate_model(multi_modal_model, test_loader, criterion, device)
        multi_modal_test_losses.append(multi_modal_test_metrics["loss"])
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss     CNN: {cnn_epoch_loss:.4f}     Multi Modal: {multi_modal_epoch_loss}")
        print(f"  Validation Loss     CNN: {cnn_val_metrics['loss']:.4f}     Multi Modal: {multi_modal_val_metrics['loss']}")
        print(f"  Test Loss     CNN: {cnn_test_metrics['loss']:.4f}     Multi Modal: {multi_modal_test_metrics['loss']}")
        print(f"  Validation Accuracy: CNN: {cnn_val_metrics['accuracy']:.4f}     Multi Modal: {multi_modal_test_metrics['accuracy']}")
        
        # Early stopping check for CNN
        if cnn_val_metrics["loss"] < cnn_best_val_loss:
            cnn_best_val_loss = cnn_val_metrics["loss"]
            cnn_epochs_without_improvement = 0
            cnn_best_model_state = copy.deepcopy(cnn_model.state_dict())
            
            if not os.path.exists("outputs"):
                    os.makedirs("outputs")
            
            # Save best model
            save_checkpoint(
                cnn_model, cnn_optimizer, epoch, cnn_best_val_loss,
                os.path.join("outputs", "cnn_best_model.pth")
            )
        else:
            cnn_epochs_without_improvement += 1
            print(f" CNN No improvement for {cnn_epochs_without_improvement} epoch(s)")
            
            if cnn_epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break

        # Early stopping check MultiModal
        if multi_modal_val_metrics["loss"] < multi_modal_best_val_loss:
            multi_modal_best_val_loss = multi_modal_val_metrics["loss"]
            multi_modal_epochs_without_improvement = 0
            multi_modal_best_model_state = copy.deepcopy(multi_modal_model.state_dict())
            
            if not os.path.exists("outputs"):
                    os.makedirs("outputs")
            
            # Save best model
            save_checkpoint(
                multi_modal_model, multi_modal_optimizer, epoch, multi_modal_best_val_loss,
                os.path.join("outputs", "multi_modal_best_model.pth")
            )
        else:
            multi_modal_epochs_without_improvement += 1
            print(f" Multi Modal No improvement for {multi_modal_epochs_without_improvement} epoch(s)")
            
            if multi_modal_epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break
        
        # Calculate and display timing info
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_remaining = epoch_duration * remaining_epochs
        print(f"  Epoch Time: {epoch_duration:.2f}s")
        print(f"  Estimated time remaining: {estimated_time_remaining:.2f}s\n")

        plot_losses(cnn_training_losses, cnn_validation_losses, cnn_test_losses, output_path='outputs/cnn_losses.png')
        plot_losses(multi_modal_training_losses, multi_modal_validation_losses, multi_modal_test_losses, output_path='outputs/multi_modal_losses.png')
    
    # Load best model for final evaluation
    if cnn_best_model_state is not None:
        cnn_model.load_state_dict(cnn_best_model_state)
        print("Loaded best model based on validation loss")
    
    # Final evaluation on test set
    test_metrics = evaluate_model(cnn_model, test_loader, criterion, device)
    print("\nFinal Test Metrics CNN:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")


    # Load best model for final evaluation
    if multi_modal_best_model_state is not None:
        multi_modal_model.load_state_dict(multi_modal_best_model_state)
        print("Loaded best model based on validation loss")
    
    # Final evaluation on test set
    test_metrics = evaluate_model(multi_modal_model, test_loader, criterion, device)
    print("\nFinal Test Metrics Multi Modal:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()