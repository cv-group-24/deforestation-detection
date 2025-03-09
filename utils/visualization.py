import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_losses(training_losses, validation_losses, test_losses=None, output_path='outputs/losses.png', is_done_training = False):
    """
    Plot training, validation, and optionally test losses.
    
    Args:
        training_losses: List of training losses
        validation_losses: List of validation losses
        test_losses: Optional list of test losses
    """
    epochs = range(1, len(training_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs, validation_losses, marker='s', linestyle='-', color='r', label='Validation Loss')
    
    if test_losses:
        plt.plot(epochs, test_losses, marker='^', linestyle='-', color='g', label='Test Loss')
    
    plt.title("Training and Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)

    if (is_done_training):
        plt.show()

def plot_confusion_matrix(conf_matrix, class_names, output_path='outputs/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    # plt.show()

def visualize_samples(dataset, class_names, num_samples=5):
    """
    Visualize sample images from dataset.
    
    Args:
        dataset: Dataset
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    # TODO Implementation for visualizing samples, is it worth? might be useful for debugging
    # ...
    pass