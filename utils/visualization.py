import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_losses(training_losses, validation_losses, test_losses=None, output_path='outputs/losses_yaren.png'):
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
    # plt.show()

def plot_confusion_matrix(conf_matrix, class_names, output_path='outputs/confusion_matrix_yaren.png'):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 17})  # Adjust color bar size
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Increase font size for color bar (legend)
    cbar = ax.collections[0].colorbar  # Access color bar
    cbar.ax.tick_params(labelsize=17)  # Increase font size of color bar ticks
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

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.colors import ListedColormap


def visualize_predictions(model, dataset, device, output_dir, num_samples=5, class_colors=None):
    """
    Visualize semantic segmentation predictions.
    
    Args:
        model: The trained model
        dataset: Dataset containing images and ground truth masks
        device: Device to run the model on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        class_colors: List of RGB tuples for each class, e.g. [(0,0,0), (255,0,0), ...]
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default colors if not provided
    if class_colors is None:
        # Default colors: background (black), deforestation (red)
        class_colors = [(0, 0, 0), (255, 0, 0)]
        
        # Generate additional colors if needed
        num_classes = len(class_colors)
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            num_classes = model.fc.out_features
        
        # Generate more colors if needed
        if num_classes > len(class_colors):
            import random
            random.seed(42)  # For reproducibility
            for _ in range(num_classes - len(class_colors)):
                class_colors.append((random.randint(0, 255), 
                                    random.randint(0, 255),
                                    random.randint(0, 255)))
    
    # Create colormap for visualization
    cmap = ListedColormap([(r/255, g/255, b/255) for r, g, b in class_colors])
    
    # Set model to evaluation mode
    model.eval()
    
    # Choose samples to visualize
    sample_indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    # Process each sample
    for i, idx in enumerate(sample_indices):
        # Get image and mask
        image, mask = dataset[idx]
        
        # Prepare image for model
        input_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Generate prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Convert image tensor to numpy array for plotting
        if isinstance(image, torch.Tensor):
            # Denormalize if needed
            if image.max() <= 1.0:
                # Assume normalized image
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                # Standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                # Already in 0-255 range
                image_np = image.cpu().numpy().transpose(1, 2, 0) / 255.0
        else:
            # Already a numpy array
            image_np = image
        
        # Get ground truth mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Resize prediction if needed to match ground truth
        if pred.shape != mask_np.shape:
            from PIL import Image
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_img = pred_img.resize((mask_np.shape[1], mask_np.shape[0]), Image.NEAREST)
            pred = np.array(pred_img)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot predicted mask
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'segmentation_sample_{i}.png'), dpi=200)
        plt.close()
    
    print(f"Saved {num_samples} visualization samples to {output_dir}")


def create_segmentation_overlay(image, mask, alpha=0.5, class_colors=None):
    """
    Create an overlay of segmentation mask on the original image.
    
    Args:
        image: RGB image as numpy array
        mask: Segmentation mask with class indices
        alpha: Transparency of the overlay (0-1)
        class_colors: List of RGB tuples for each class
    
    Returns:
        Overlay image as numpy array
    """
    if class_colors is None:
        class_colors = [(0, 0, 0), (255, 0, 0)]  # Black for background, red for deforestation
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_idx, (r, g, b) in enumerate(class_colors):
        colored_mask[mask == class_idx] = [r, g, b]
    
    # Convert image to proper format if needed
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize if needed
        if image_np.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
        
        image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        image_np = image
    
    # Resize image if needed
    if image_np.shape[:2] != mask.shape:
        from PIL import Image
        image_pil = Image.fromarray(image_np)
        image_pil = image_pil.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
        image_np = np.array(image_pil)
    
    # Create overlay
    overlay = image_np.copy()
    for i in range(3):
        overlay[:,:,i] = image_np[:,:,i] * (1 - alpha) + colored_mask[:,:,i] * alpha
    
    return overlay


def visualize_batch_predictions(model, dataloader, device, output_dir, batch_idx=0, max_samples=8,
                               class_colors=None):
    """
    Visualize predictions for a batch of samples.
    
    Args:
        model: The trained model
        dataloader: DataLoader containing batches of images and masks
        device: Device to run the model on
        output_dir: Directory to save visualizations
        batch_idx: Which batch to visualize
        max_samples: Maximum number of samples to visualize from the batch
        class_colors: List of RGB tuples for each class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default colors
    if class_colors is None:
        class_colors = [(0, 0, 0), (255, 0, 0)]  # Black for background, red for deforestation
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch
    for i, (images, masks) in enumerate(dataloader):
        if i == batch_idx:
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Limit number of samples to visualize
            num_samples = min(images.shape[0], max_samples)
            
            # Create a grid of visualizations
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
            
            for j in range(num_samples):
                # Get image, mask and prediction
                image = images[j].cpu()
                mask = masks[j].cpu().numpy()
                pred = preds[j]
                
                # Convert image tensor to numpy array for plotting
                if image.max() <= 1.0:
                    # Denormalize
                    image_np = image.numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
                else:
                    # Already in 0-255 range
                    image_np = image.numpy().transpose(1, 2, 0) / 255.0
                
                # Resize prediction if needed
                if pred.shape != mask.shape:
                    from PIL import Image
                    pred_img = Image.fromarray(pred.astype(np.uint8))
                    pred_img = pred_img.resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
                    pred = np.array(pred_img)
                
                # Create colormap
                cmap = ListedColormap([(r/255, g/255, b/255) for r, g, b in class_colors])
                
                # Plot
                if num_samples == 1:
                    axes[0].imshow(image_np)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                else:
                    axes[j, 0].imshow(image_np)
                    axes[j, 0].set_title(f'Image {j+1}')
                    axes[j, 0].axis('off')
                    
                    axes[j, 1].imshow(mask, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
                    axes[j, 1].set_title(f'Ground Truth {j+1}')
                    axes[j, 1].axis('off')
                    
                    axes[j, 2].imshow(pred, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
                    axes[j, 2].set_title(f'Prediction {j+1}')
                    axes[j, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'batch_{batch_idx}_predictions.png'), dpi=200)
            plt.close()
            break
    
    print(f"Saved batch visualization to {output_dir}")