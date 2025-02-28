from torch.utils.data import DataLoader
import pandas as pd
import os

from data.dataset import ForestNetDataset
from torchvision import transforms
from transforms import get_transforms

def create_data_loaders(config):
    """
    Create training, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary containing dataloaders
    """
    # Read the CSV files
    dataset_path = config["data"]["dataset_path"]
    test_path = os.path.join(dataset_path, "test.csv")
    train_path = os.path.join(dataset_path, "train.csv")
    validation_path = os.path.join(dataset_path, "val.csv")
    
    # Load DataFrames and process them
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)
    
    if config["data"].get("sample_data", False):
        seed = config["training"]["seed"]
        sample_size = config["data"].get("sample_size", 10)
        test_df = test_df.sample(n=sample_size, random_state=seed)
        train_df = train_df.sample(n=sample_size, random_state=seed)
        val_df = val_df.sample(n=sample_size, random_state=seed)
    
    # Create label mapping
    labels = sorted(train_df["merged_label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    # Apply class balancing if enabled
    if config["data"].get("balance_classes", True):
        train_df = balance_classes(train_df)
    
    ## TODO, think about if this is the right place for this, also seems weird to me that a necessary
    ## preprocessing is labelled as a transform. should discuss this
    transform = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),

        # TODO: Look into calculating these values for our dataset. It probably has a lot more green than other
        # datasets.
        # These normalization values are typical for natural images.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ForestNetDataset(
        train_df, dataset_path, transform=transform,
        spatial_augmentation=config["transforms"]["spatial_augmentation"],
        pixel_augmentation=config["transforms"]["pixel_augmentation"],
        resize=config["transforms"]["resize"],
        is_training=False, ## TODO, change to true after figuring out why it breaks
        label_map=label_to_index,
        use_masks=config["data"]["use_masking"]
    )
    
    val_dataset = ForestNetDataset(
        val_df, dataset_path, transform=transform,
        is_training=False, label_map=label_to_index,
        use_masks=config["data"]["use_masking"]
    )
    
    test_dataset = ForestNetDataset(
        test_df, dataset_path, transform=transform,
        is_training=False, label_map=label_to_index,
        use_masks=config["data"]["use_masking"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"]["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False, 
        num_workers=config["data"]["num_workers"]
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False, 
        num_workers=config["data"]["num_workers"]
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "label_to_index": label_to_index,
        "index_to_label": {idx: label for label, idx in label_to_index.items()}
    }

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Balance classes by oversampling."""
    max_count = df["merged_label"].value_counts().max()
    balanced_df = df.groupby("merged_label", group_keys=False).apply(
        lambda x: x.sample(max_count, replace=True, random_state=42)
    ).reset_index(drop=True)
    return balanced_df