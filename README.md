# deforestation-detection

This project implements deep learning models for deforestation classification from satellite imagery using the ForestNet dataset.

## Setup

1. **Create a Conda Environment**:
   
   Run the following command to create a new Conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Environment**:
   
   Activate the newly created environment with:
   ```bash
   conda activate deforestation-detection
   ```

3. **Download and Place the Dataset**:
   
   Download the dataset from [this link](https://stanfordmlgroup.github.io/projects/forestnet/) and place it in the `data/raw` directory. You can also update the configuration file to point to a different location if preferred. Make sure that the 'ForestNetDataset' folder is being pointed to.

## Project Structure

```
deforestation-detection/
│
├── config/                   # Configuration files
│   └── default_config.py     # Default configuration settings
│
├── data/                     # Data handling utilities
│   ├── dataloader.py         # Data loading utilities
│   ├── dataset.py            # Dataset class definitions
│   ├── transforms.py         # Data transformation functions
│   └── raw/                  # Raw dataset files (not tracked in git)
│       └── ForestNetDataset/ # The downloaded dataset goes here
│
├── models/                   # Model architectures
│   ├── helpers.py            # Model loading utilities
│   └── [model files]         # Various model implementations
│
├── utils/                    # Utility functions
│   ├── helpers.py            # General helper functions
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Visualization utilities
│
├── outputs/                  # Training outputs (not tracked in git)
│   ├── best_model.pth        # Saved model checkpoints
│   └── [other outputs]       # Various outputs, logs, and visualizations
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── environment.yml           # Conda environment specification
└── README.md                 # This file
```

## Usage

### Training

To train a model using the default configuration:

```bash
python train.py
```

You can specify a custom configuration file:

```bash
python train.py --config path/to/config.json
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint outputs/best_model.pth
```

### Configuration

The project uses JSON configuration files to set hyperparameters and other settings. The default configuration is defined in `config/default_config.py`, and you can override it by providing a custom configuration file.

Example configuration:

```json
{
    "data": {
        "dataset_path": r"data\raw\ForestNetDataset",
        "batch_size": 32,
        "num_workers": 0,
        "use_masking": False,
        "use_augmentation": True,
    },
    "training": {
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "early_stopping_patience": 3,
        "seed": 42
    },
    "model": {
        "type": "EnhancedCNN",
        "dropout_rate": 0.5
    },
    "transforms": {
        "resize": "small",
        "spatial_augmentation": "affine",
        "pixel_augmentation": "all"
    },
    "testing": {
        "is_testing": True,
        "resize": "small",
        "spatial_augmentation": "affine",
        "pixel_augmentation": "all"
    }
}
```


## Models

The project supports multiple model architectures for classification tasks:

- **Classification Pre-trained**: ResNet50, EfficientNetB0, DenseNet121 (see `models/transferlearning.py`)
- **Classification from Scratch**: SimpleCNN, EnhancedCNN (Our own CNNs, see `models/cnn.py` )

## Acknowledgements

- This project uses the ForestNet dataset from the Stanford ML Group, accessible [here](https://stanfordmlgroup.github.io/projects/forestnet/)

