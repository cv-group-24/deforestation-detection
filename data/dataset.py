import os
import numpy as np
from PIL import Image, ImageDraw

from torch.utils.data import Dataset
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
            # Debug: print the image_path to see if it looks correct
            image = Image.open(image_path).convert("RGB")

            # Apply masking if enabled
            if self.use_masks:
                sample_path = os.path.join(self.dataset_path, row["example_path"])
                label_value = self.label_map[row["merged_label"]] if self.label_map else 1
                
                # Import get_mask function if not already imported
                mask_bool = get_mask(image_path, sample_path, label_value)
                
                # Convert PIL image to numpy array
                image_np = np.array(image)
                
                # Apply mask - set pixels outside the mask to black
                masked_image_np = image_np.copy()
                masked_image_np[~mask_bool] = 0
                
                # Convert back to PIL image
                image = Image.fromarray(masked_image_np)

            # Convert PIL image to numpy array before augmentation
            image_np = np.array(image)
            image_aug = np.array(image)

            # Apply augmentation transformations if in training mode
            if self.is_training:
                for augmentation in self.augmentations:

                    # plt.imshow(image_np)
                    # plt.show()

                    if image_aug.ndim == 3:  # Shape (H, W, C)
                        image_aug = np.expand_dims(image_aug, axis=0)  # Make it (1, H, W, C)

                    augmented = augmentation(images=image_aug)
                    if isinstance(augmented, dict):
                        image_aug = augmented['images'][0]
                    elif isinstance(augmented, list):
                        image_aug = augmented[0]
                    elif isinstance(augmented, np.ndarray):
                        image_aug = augmented

                    if image_aug.ndim == 4:
                        image_aug = image_aug.squeeze(axis=0)

                if self.use_masks and valid_crop(image_aug, image_np):
                    image_np = image_aug

            # Convert back to PIL image after augmentation if needed
            if self.transform:
                image = Image.fromarray(image_np)
                image = self.transform(image)

            label = row["merged_label"]
            if self.label_map is not None:
                label = self.label_map[label]

            return image, label
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
    """
    Generate a mask with debugging information
    """

    pil_image = Image.open(im_path).convert('RGB')
    rgb_image = np.array(pil_image)
    height, width = rgb_image.shape[:2]
    
    if debug:
        print(f"Image dimensions: {width}x{height}")
    
    polygon = None
    try:
        with open(os.path.join(sample_path, 'forest_loss_region.pkl'), 'rb') as f:
            polygon = pickle.load(f)
            if debug:
                print(f"Polygon type: {polygon.geom_type}")
                if polygon.geom_type == 'Polygon':
                    print(f"Polygon bounds: {polygon.bounds}")
                    print(f"First few coords: {list(polygon.exterior.coords)[:3]}")
                elif hasattr(polygon, 'geoms'):
                    print(f"Number of polygons: {len(polygon.geoms)}")
                    if len(polygon.geoms) > 0:
                        print(f"First polygon bounds: {polygon.geoms[0].bounds}")
    except Exception as e:
        print(f"Error loading polygon for {sample_path}: {e}")
        return np.ones((height, width), dtype=bool)
    
    # Get centroid
    lon = polygon.centroid.xy[0][0]
    lat = polygon.centroid.xy[1][0]
    
    if debug:
        print(f"Centroid: lon={lon}, lat={lat}")
    
    # Calculate meters per degree
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    m_per_deg_lon = 111132.954 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    
    # Resolution in meters per pixel
    res = 4.77  # PLANETSCOPE resolution
    
    # Calculate degree span of image
    deg_lat = (height * res) / m_per_deg_lat
    deg_lon = (width * res) / m_per_deg_lon
    
    if debug:
        print(f"Degree spans: lon={deg_lon}, lat={deg_lat}")
    
    # Create a blank mask
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # Variable to track if any polygon was drawn
    any_polygon_drawn = False
    
    try:
        # Process based on geometry type
        if polygon.geom_type == 'Polygon':
            coords = np.array(polygon.exterior.coords)
            draw.polygon([tuple(coord) for coord in coords],
                    outline=label, fill=label)
            any_polygon_drawn = True

        elif hasattr(polygon, 'geoms'):
            # Handle MultiPolygon
            for poly in polygon.geoms:
                coords = np.array(poly.exterior.coords)
                
                draw.polygon([tuple(coord) for coord in coords],
                    outline=label, fill=label)
                any_polygon_drawn = True

    except Exception as e:
        print(f"Error creating polygon for {sample_path}: {e}")
        if debug:
            print(f"Exception details: {e}")
        return np.ones((height, width), dtype=bool)
    
    except Exception as e:
        print(f"Error creating polygon for {sample_path}: {e}")
        if debug:
            print(f"Exception details: {e}")
        return np.ones((height, width), dtype=bool)
    
    # Convert mask to numpy array
    mask_np = np.array(mask)
    
    # Set ignored values to 0
    mask_np[mask_np == 255] = 0
    
    # Make sure mask is boolean
    mask_bool = mask_np.astype(bool)
    
    if debug:
        # Print stats about the mask
        mask_sum = np.sum(mask_bool)
        mask_percentage = (mask_sum / (height * width)) * 100
        print(f"Mask covers {mask_sum} pixels ({mask_percentage:.2f}% of image)")
        
        # If no polygon was drawn or mask is all zeros, return a full mask for debugging
        if not any_polygon_drawn or mask_sum == 0:
            print("WARNING: No valid polygon was drawn!")
            
        # Create debug visualization
        plt.figure(figsize=(16, 4))  # Made wider to accommodate 4 subplots
        
        plt.subplot(141)
        plt.imshow(rgb_image)
        plt.title("Original Image")
        
        plt.subplot(142)
        plt.imshow(mask_bool, cmap='gray')
        plt.title("Generated Mask")
        
        plt.subplot(143)
        masked_img = rgb_image.copy()
        masked_img[~mask_bool] = 0
        plt.imshow(masked_img)
        plt.title("Masked Image")

        # New subplot showing outline
        plt.subplot(144)
        # Show original image first
        plt.imshow(rgb_image)
        # Create outline by subtracting eroded mask from original mask
        from scipy.ndimage import binary_erosion
        outline = mask_bool.astype(np.uint8) - binary_erosion(mask_bool).astype(np.uint8)
        # instantiate a new image that is the rgb_image but is white where outline = 1
        image_to_show = rgb_image.copy()
        image_to_show[outline == 1] = [255, 255, 255]
        
        plt.imshow(image_to_show)
        plt.title("Outline on Image")
        
        plt.tight_layout()
        plt.show()
    
    return mask_bool