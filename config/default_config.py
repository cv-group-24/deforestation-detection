# Default configuration settings
DEFAULT_CONFIG = {
    "data": {
        "dataset_path": r"C:\Users\chris\Desktop\University\Code\ComputerVision\ForestNetDataset",
        "batch_size": 32,
        "num_workers": 0,
        "use_masking": False,
        "sample_data": True,
        "sample_size": 100
    },
    "training": {
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "early_stopping_patience": 5,
        "seed": 42
    },
    "model": {
        "type": "DenseNetTransferLearning",  # or "EnhancedCNN"
        "dropout_rate": 0.5, 
        "multi_modal_size": 2,
    },
    "transforms": {
        "resize": "small", ## or 'small'
        "spatial_augmentation": "affine", ## or 'affine'
        "pixel_augmentation": "all" ## or 'all'
    }
}