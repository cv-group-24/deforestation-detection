import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, data_loader, criterion, device):
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

            # Resize labels to match output dimensions
            if labels.shape[-2:] != outputs.shape[-2:]:
                labels = torch.nn.functional.interpolate(
                    labels.float().unsqueeze(1), 
                    size=outputs.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()

            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    # accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # # Generate detailed metrics
    # class_report = classification_report(all_labels, all_preds, output_dict=True)
    # conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        "loss": epoch_loss,
        # "accuracy": accuracy,
        # "classification_report": class_report,
        # "confusion_matrix": conf_matrix,
        # "predictions": all_preds,
        # "labels": all_labels
    }