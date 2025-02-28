# Default configuration settings
DEFAULT_CONFIG = {
    "data": {
        "dataset_path": "data/raw/ForestnetDataset",
        "batch_size": 32,
        "num_workers": 0,
        "use_masking": True,
        # "sample_data": True,
        # "sample_size": 100
    },
    "training": {
        "num_epochs": 1,
        "learning_rate": 1e-4,
        "early_stopping_patience": 5,
        "seed": 42
    },
    "model": {
        "type": "EnhancedCNN",  # or "SimpleCNN"
        "dropout_rate": 0.5
    },
    "transforms": {
        "resize": "none", ## or 'small'
        "spatial_augmentation": "none", ## or 'affine'
        "pixel_augmentation": "none" ## or 'all'
    }
}