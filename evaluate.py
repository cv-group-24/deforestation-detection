import argparse
import json
import os
import torch
import torch.nn as nn

from config.default_config import DEFAULT_CONFIG
from data.dataloader import create_data_loaders
from models.helpers import get_model
from utils.helpers import get_device, load_checkpoint
from utils.metrics import evaluate_model
from utils.visualization import plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the ForestNet dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    data = create_data_loaders(config)
    test_loader = data["test_loader"]
    index_to_label = data["index_to_label"]
    
    # Create model
    num_classes = len(data["label_to_index"])
    model = get_model(
        config["model"]["type"], 
        num_classes
    )
    model = model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Print classification report
    class_report = test_metrics["classification_report"]
    
    print("\nClassification Report:")
    print(class_report)
    for class_idx, metrics in class_report.items():
        if isinstance(class_idx, str) and class_idx.isdigit():  # Skip averages
            label = index_to_label.get(int(class_idx), class_idx)
            print(f"  Class {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")
    
    # Plot confusion matrix
    class_names = [index_to_label[i] for i in range(num_classes)]
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names)

if __name__ == "__main__":
    main()