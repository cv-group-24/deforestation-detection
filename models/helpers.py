from models.cnn import SimpleCNN, EnhancedCNN
from models.multi_modal_models import MultiModalMLP, DecisionTreeWrapper, RandomForestWrapper
from models.transferlearning import ResNetTransferLearning, EfficientNetTransferLearning, DenseNetTransferLearning


def get_model(model_type, num_classes, multi_modal_size):
    """
    Factory function to get a model.
    
    Args:
        model_type: Type of model ('SimpleCNN', 'EnhancedCNN',
          'ResNetTransferLearning', 'EfficientNetTransferLearning',
            'DenseNetTransferLearning')
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        torch.nn.Module: The model
    """
    if model_type == "SimpleCNN":
        return SimpleCNN(num_classes, multi_modal_size)
    elif model_type == "EnhancedCNN":
        return EnhancedCNN(num_classes, multi_modal_size)
    elif model_type == "ResNetTransferLearning":
        return ResNetTransferLearning(num_classes, multi_modal_size)
    elif model_type == "EfficientNetTransferLearning":
        return EfficientNetTransferLearning(num_classes, multi_modal_size)
    elif model_type == "DenseNetTransferLearning":
        return DenseNetTransferLearning(num_classes, multi_modal_size)
    elif model_type == "MLP":
        return MultiModalMLP(input_size=multi_modal_size, num_classes=num_classes)
    elif model_type == "DT":
        return DecisionTreeWrapper(input_size=multi_modal_size, num_classes=num_classes)
    elif model_type == "RF":
        return RandomForestWrapper(input_size=multi_modal_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")