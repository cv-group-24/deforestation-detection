import os
import numpy as np
from PIL import Image, ImageDraw

from torch.utils.data import Dataset
import torch
from data.transforms import get_transforms, valid_crop

import pickle
import matplotlib.pyplot as plt

class ForestNetDataset(Dataset):
    def __init__(self,
                 df,
                 dataset_path,
                 transform=None,
                 label_map=None,
                 spatial_augmentation="none",
                 pixel_augmentation="none",
                 resize="none",
                 is_training=False,
                 use_landsat=False,
                 use_masks=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the image paths and labels.
            dataset_path (str): The base directory for the images.
            transform (callable, optional): A function/transform to apply to the images.
            label_map (dict, optional): Mapping from label names to integers.
        """
        self.df = df
        self.dataset_path = dataset_path
        self.transform = transform
        self.label_map = label_map

        self.spatial_augmentation = spatial_augmentation
        self.pixel_augmentation = pixel_augmentation
        self.resize = resize
        self.is_training = is_training
        self.use_landsat = use_landsat

        self.use_masks = use_masks

        # Get the combined transformations
        self.augmentations = get_transforms(
            resize=self.resize,
            spatial_augmentation=self.spatial_augmentation,
            pixel_augmentation=self.pixel_augmentation,
            is_training=self.is_training,
            use_landsat=self.use_landsat
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_rel_path = row["example_path"] + "/images/visible/composite.png"
            image_path = os.path.join(self.dataset_path, image_rel_path)
            image = Image.open(image_path).convert("RGB")
            
            # Get sample path for mask generation
            sample_path = os.path.join(self.dataset_path, row["example_path"])
            label_value = self.label_map[row["merged_label"]] if self.label_map else 1
            
            # Generate mask - this will be our ground truth
            mask = get_mask(image_path, sample_path, label_value)
            
            # Convert image to numpy array for augmentation
            image_np = np.array(image)
            mask_np = mask
            
            # Apply transformations if needed
            if self.transform:
                image = Image.fromarray(image_np)
                image = self.transform(image)
            else:
                # Convert to tensor manually if no transform is provided
                image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
                
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(mask_np).long()  # Use long for class labels
            
            return image, mask_tensor
        except Exception as e:
            print(f"Error loading image at index {idx} from path {image_path}: {e}")
            raise e

class ConcatDataset(Dataset):
    def __init__(self, original_dataset, augmented_dataset):
        """
        Args:
            original_dataset (ForestNetDataset): Original dataset
            augmented_dataset (ForestNetDataset): Augmented dataset
        """
        self.original_dataset = original_dataset
        self.augmented_dataset = augmented_dataset

    def __len__(self):
        # The combined length is the sum of the lengths of both datasets
        return len(self.original_dataset) + len(self.augmented_dataset)

    def __getitem__(self, idx):
        # Determine if we are accessing the original or augmented image
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]  # Original image
        else:
            return self.augmented_dataset[idx - len(self.original_dataset)]  # Augmented image



def get_mask(im_path, sample_path, label, debug=False):
    """Generate a semantic segmentation mask from polygons"""
    
    # Get image dimensions
    pil_image = Image.open(im_path).convert('RGB')
    height, width = pil_image.size[1], pil_image.size[0]
    
    # Create blank mask (background = 0)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    try:
        # Load polygon data
        with open(os.path.join(sample_path, 'forest_loss_region.pkl'), 'rb') as f:
            polygon = pickle.load(f)
            
        # Draw polygons based on geometry type
        if polygon.geom_type == 'Polygon':
            coords = np.array(polygon.exterior.coords)
            draw.polygon([tuple(coord) for coord in coords], outline=label, fill=label)
        elif hasattr(polygon, 'geoms'):
            for poly in polygon.geoms:
                coords = np.array(poly.exterior.coords)
                draw.polygon([tuple(coord) for coord in coords], outline=label, fill=label)
                
    except Exception as e:
        if debug:
            print(f"Error processing polygon for {sample_path}: {e}")
        # Return all-background mask on error
        return np.zeros((height, width), dtype=np.uint8)
    
    # Convert to numpy array and ensure proper label values
    return np.array(mask).astype(np.uint8)