# Default configuration settings
DEFAULT_CONFIG = {
    "data": {
        "dataset_path": r"C:\Users\yaren\OneDrive\Desktop\Yaren's\TU Delft\Masters\CV\deep\downloads\ForestNetDataset",
        "batch_size": 32,
        "num_workers": 0,
        "use_masking": False,
        "use_augmentation": False,
        # "sample_data": True,
        # "sample_size": 100
    },
    "training": {
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "early_stopping_patience": 5,
        "seed": 42
    },
    "model": {
        "type": "UNet",  # or "SimpleCNN"
        "problem_type": "semantic_segmentation",  # or "classification"
        "dropout_rate": 0.5
    },
    "transforms": {
        "resize": "small", ## or 'small'
        "spatial_augmentation": "affine", ## or 'affine'
        "pixel_augmentation": "all" ## or 'all'
    },
    "testing": {
        "is_testing": True,
        "resize": "small", ## or 'small'
        "spatial_augmentation": "affine", ## or 'affine'
        "pixel_augmentation": "all" ## or 'all'
    }
}