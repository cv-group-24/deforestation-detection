import argparse
import json
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

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


def performance_metrics():
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
    
    # Print classification report
    class_report = test_metrics["classification_report"]

    with open("outputs/classification_report.txt", "w") as f:
        f.write("Test Metrics:\n")
        f.write(f"  Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")

        f.write("\nClassification Report:\n")
        for class_idx, metrics in class_report.items():
            if isinstance(class_idx, str) and class_idx.isdigit():  # Skip averages
                label = index_to_label.get(int(class_idx), class_idx)
                f.write(f"\nClass {label}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
        f.write("\n")

        # Write Macro Average
        f.write("\nMacro Average:\n")
        f.write(f"  Precision: {class_report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall: {class_report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-Score: {class_report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  Support: {class_report['macro avg']['support']}\n")

        # Write Weighted Average
        f.write("\nWeighted Average:\n")
        f.write(f"  Precision: {class_report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall: {class_report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score: {class_report['weighted avg']['f1-score']:.4f}\n")
        f.write(f"  Support: {class_report['weighted avg']['support']}\n")
    
    # Plot confusion matrix
    class_names = [index_to_label[i] for i in range(num_classes)]
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names)


def metamorphic_testing():
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
    metamorphic_test_loader = data["metamorphic_test_loader"]
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

    compare_predictions(model, test_loader, metamorphic_test_loader, device, index_to_label)


def compare_predictions(model, test_loader, metamorphic_test_loader, device, index_to_label):
    """
    Compare predictions between the test_loader and test_augmented_loader.
    This function will print how many times the predicted labels are the same or different
    and plot the ratio of same predictions per class.

    Args:
        model: The trained model.
        test_loader: The original test data loader.
        metamorphic_test_loader: The augmented test data loader.
        device: The device (CPU or GPU).
        index_to_label: Mapping from index to label names (for plotting classes).
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize counters for same and different predictions per class
    same_count_per_class = {i: 0 for i in range(len(index_to_label))}
    total_count_per_class = {i: 0 for i in range(len(index_to_label))}

    # Loop through the data from both loaders
    for (image_orig, label_orig), (image_aug, label_aug) in zip(test_loader, metamorphic_test_loader):
        # Move images and labels to the device
        image_orig, label_orig = image_orig.to(device), label_orig.to(device)
        image_aug, label_aug = image_aug.to(device), label_aug.to(device)

        # Get predictions from both original and augmented data
        with torch.no_grad():
            pred_orig = torch.argmax(model(image_orig), dim=1)
            pred_aug = torch.argmax(model(image_aug), dim=1)

        # Compare the predictions for the current image
        for o, a, label in zip(pred_orig, pred_aug, label_orig):
            total_count_per_class[label.item()] += 1
            if o == a:
                same_count_per_class[label.item()] += 1

    # Calculate the ratio of same predictions per class
    same_ratio_per_class = {class_id: same_count_per_class[class_id] / total_count_per_class[class_id]
                            if total_count_per_class[class_id] > 0 else 0
                            for class_id in same_count_per_class}

    # Print the results
    print(f"\nPredictions comparison:")
    for class_id in same_ratio_per_class:
        print(f"Class {index_to_label[class_id]}: Same Prediction Ratio: {same_ratio_per_class[class_id]:.4f}")

    # Plotting the ratio of same predictions per class
    plt.figure(figsize=(10, 6))
    bars = plt.bar(same_ratio_per_class.keys(), same_ratio_per_class.values(),
                   tick_label=[index_to_label[class_id] for class_id in same_ratio_per_class],
                   width=0.4)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=10)  # Add text above each bar
    plt.xlabel('Classes')
    plt.ylabel('Ratio of Same Predictions')
    plt.title('Ratio of Same Predictions for Each Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/metamorphic_testing.png")


if __name__ == "__main__":
    # performance_metrics()
    metamorphic_testing()

