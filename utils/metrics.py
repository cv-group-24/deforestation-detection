import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to use
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Generate detailed metrics
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "predictions": all_preds,
        "labels": all_labels
    }


def evaluate_semantic_segmentation_model(model, data_loader, criterion, device):
    """
    Evaluate a semantic segmentation model on a dataset.
    
    Args:
        model: The segmentation model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to use
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    
    # Initialize metrics
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_preds = []
    all_labels = []
    class_ious = {}  # IoU for each class
    
    num_classes = 0  # Will be determined from predictions
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Resize masks to match output dimensions if needed
            if masks.shape[-2:] != outputs.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks.float().unsqueeze(1),
                    size=outputs.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
            
            # Calculate loss
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Convert to numpy for metric calculation
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Update maximum number of classes
            num_classes = max(num_classes, outputs.size(1))
            
            # Calculate IoU and Dice score for each sample
            for i in range(preds_np.shape[0]):  # For each sample in the batch
                # Calculate metrics per sample
                sample_iou, sample_dice, sample_acc = calculate_segmentation_metrics(
                    preds_np[i], masks_np[i], num_classes
                )
                iou_scores.append(sample_iou)
                dice_scores.append(sample_dice)
                pixel_accuracies.append(sample_acc)
                
                # Store predictions and ground truth for later analysis
                all_preds.append(preds_np[i])
                all_labels.append(masks_np[i])
    
    # Calculate class-wise IoU if needed
    class_ious = calculate_class_ious(all_preds, all_labels, num_classes)
    
    # Calculate mean metrics
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    mean_pixel_accuracy = np.mean(pixel_accuracies)
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(data_loader.dataset)
    
    return {
        "loss": epoch_loss,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mean_pixel_accuracy": mean_pixel_accuracy,
        "class_ious": class_ious,
        "iou_scores": iou_scores,
        "dice_scores": dice_scores,
        "pixel_accuracies": pixel_accuracies,
    }

def calculate_segmentation_metrics(pred, target, num_classes):
    """
    Calculate IoU, Dice coefficient, and pixel accuracy for a single prediction.
    
    Args:
        pred: Predicted segmentation mask (2D array with class indices)
        target: Ground truth mask (2D array with class indices)
        num_classes: Number of classes
        
    Returns:
        tuple: (IoU, Dice coefficient, Pixel accuracy)
    """
    # Calculate IoU (Jaccard index)
    iou_sum = 0
    # ignore background class (0)
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        # Handle case where the class doesn't exist in this image
        if union == 0:
            continue
        
        iou = intersection / union
        iou_sum += iou
    
    # Calculate mean IoU only for classes that exist in the ground truth
    num_relevant_classes = len(np.unique(target)) - (1 if 0 in target else 0)
    mean_iou = iou_sum / max(1, num_relevant_classes)
    
    # Calculate Dice coefficient (F1 score)
    dice_sum = 0
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        intersection = (pred_cls * target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        
        # Handle case where the class doesn't exist in this image
        if total == 0:
            continue
        
        dice = 2 * intersection / total
        dice_sum += dice
    
    # Calculate mean Dice only for classes that exist in the ground truth
    mean_dice = dice_sum / max(1, num_relevant_classes)
    
    # Calculate pixel accuracy
    pixel_accuracy = (pred == target).mean()
    
    return mean_iou, mean_dice, pixel_accuracy

def calculate_class_ious(predictions, targets, num_classes):
    """
    Calculate IoU for each class across the dataset.
    
    Args:
        predictions: List of prediction masks
        targets: List of target masks
        num_classes: Number of classes
        
    Returns:
        dict: Dictionary with class IoU values
    """
    class_intersections = np.zeros(num_classes)
    class_unions = np.zeros(num_classes)
    
    for pred, target in zip(predictions, targets):
        for cls in range(num_classes):
            pred_cls = (pred == cls).astype(np.float32)
            target_cls = (target == cls).astype(np.float32)
            
            intersection = np.logical_and(pred_cls, target_cls).sum()
            union = np.logical_or(pred_cls, target_cls).sum()
            
            class_intersections[cls] += intersection
            class_unions[cls] += union
    
    # Calculate IoU for each class
    class_ious = {}
    for cls in range(num_classes):
        if class_unions[cls] == 0:
            class_ious[f"class_{cls}"] = 0.0
        else:
            class_ious[f"class_{cls}"] = class_intersections[cls] / class_unions[cls]
    
    return class_ious