import os
import numpy as np
from PIL import Image, ImageDraw

from torch.utils.data import Dataset
from data.transforms import get_transforms

import pickle
import matplotlib.pyplot as plt
from geopy import distance
import json
import torch as torch

from utils.constants import *

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
                 use_masks=False,
                 feature_scale=False):
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

        self.feature_scale = feature_scale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_rel_path = row["example_path"] + "/images/visible/composite.png"
            image_path = os.path.join(self.dataset_path, image_rel_path)
            # Debug: print the image_path to see if it looks correct
            image = Image.open(image_path).convert("RGB")

            # get the latitude and longtidue of the image to calculase osm data
            multi_modal_tensor = self.get_multi_modal_features(row, feature_scale=self.feature_scale)

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

            # Apply augmentation transformations if in training mode
            if self.is_training:
                for augmentation in self.augmentations:

                    # plt.imshow(image_np)
                    # plt.show()

                    if image_np.ndim == 3:  # Shape (H, W, C)
                        image_np = np.expand_dims(image_np, axis=0)  # Make it (1, H, W, C)

                    augmented = augmentation(images=image_np)
                    if isinstance(augmented, dict):
                        image_np = augmented['images'][0]
                    elif isinstance(augmented, list):
                        image_np = augmented[0]
                    elif isinstance(augmented, np.ndarray):
                        image_np = augmented

                    if image_np.ndim == 4:
                        image_np = image_np.squeeze(axis=0)

            # Convert back to PIL image after augmentation if needed
            if self.transform:
                image = Image.fromarray(image_np)
                image = self.transform(image)

            label = row["merged_label"]
            if self.label_map is not None:
                label = self.label_map[label]

            return image, multi_modal_tensor, label  # Include OSM tensor in return
        except Exception as e:
            print(f"Error loading image at index {idx} from path {image_path}: {e}")
            raise e
    
    # Using geodesic distance provided by geopy
    # See: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
    def get_osm_features(self, sample_path, lat, lon, feature_scale=False):

        aux_path = os.path.join(sample_path, "auxiliary")
        osm_path = os.path.join(aux_path, 'osm.json')

        # Load the combined JSON file
        with open(osm_path, 'r') as f:
            osm_data = json.load(f)

        # Extract the street and city data
        street = osm_data.get("closest_street", {})
        city = osm_data.get("closest_city", {})

        street_dist = distance.distance((lat, lon),
                                        (street.get('lat'), street.get('lon'))).km                                  
        city_dist = distance.distance((lat, lon),
                                        (city.get('lat'), city.get('lon'))).km

        if feature_scale:
            street_min, street_max = OSM_SCALING['street']
            street_dist = self._feature_scale(street_dist, street_min, street_max, False)
            city_min, city_max = OSM_SCALING['city']
            city_dist = self._feature_scale(city_dist, city_min, city_max, False)
                                        
        features = {'street_dist': street_dist,
                    'city_dist': city_dist}        
        return features 

    def get_ncep_features(self, sample_path, feature_scale=False):
        aux_path = os.path.join(sample_path, "auxiliary")
        ncep_file = os.path.join(aux_path, "ncep.npy")
        ncep_data = np.load(ncep_file)

        scaled_ncep_data = np.zeros_like(ncep_data) if feature_scale else ncep_data
        ## Manually looked at the ordering of NCEP Scaling constants in 'download_ncep.py' 
        ## and ordered the NCEP scaling accordingly. This way we can scale the NCEP data
        ## with hopefully the right scaling for the min, max and mean value for each feature
        if feature_scale:
            for i, feature in enumerate(NCEP_SCALING):
                min_val = NCEP_SCALING[feature][0]
                max_val = NCEP_SCALING[feature][1]
                for j in range(3):
                    scaled_ncep_data[3 * i + j] = self._feature_scale(ncep_data[3 * i + j], min_val, max_val)

        ## returns an array of 84 values, 3 values per ncep feature for 28 features
        return scaled_ncep_data
    
    def get_srtm_img(self, sample_path, feature_scale=False):
        aux_path = os.path.join(sample_path, "auxiliary")
        srtm_file = os.path.join(aux_path, "srtm.npy")
        srtm = np.load(srtm_file)

        ## upon manual inspection, I've determined that band 0 is altitude, band 1 is aspect and band 2 is slope
        band_to_id = {0: "altitude", 1: "aspect", 2: "slope"}

        srtm_scaled = np.zeros_like(srtm) if feature_scale else srtm 

        if feature_scale:
            for i in range(srtm.shape[0]):
                band_name = band_to_id[i]
                ## scale based on max and min values for each band
                srtm_scaled[i] = self._feature_scale(srtm[i], SRTM_SCALING[band_name][0], SRTM_SCALING[band_name][1])
        
        ## now we scale it back to have image values
        srtm_final = self._feature_scale(srtm_scaled, 0, 255) if feature_scale else srtm_scaled
        
        ## returns a 332 x 332 image for each of the 3 SRTM bands, so 3 x 332 x 332
        return srtm_final
    
    def get_gfc_img(self, sample_path, feature_scale=False):
        aux_path = os.path.join(sample_path, "auxiliary")
        gfc_file = os.path.join(aux_path, "gfc.npy")
        ## GFC contains the GFC gain band and the GFC cover band, the original code uses only the gain band
        gfc = np.load(gfc_file)

        gfc_gain = gfc[0]  # GFC gain band

        ## TODO: maybe take a look at what the cover band represents and see if it's useful
        gfc_cover = gfc[1]  # GFC cover band

        ## According to original code, the gfc gain band is already rescaled, so no rescaling needed
        ## however, upon manual inspection, I see that the max value tends to be 100, so I will rescale it to [0, 255]
        gfc_gain = self._feature_scale(gfc_gain, 0, 255) if feature_scale else gfc_gain

        ## gfc is a 2 x 1 x 332 x 332 image, so we return the 332 x 332 image
        ## we return only gfc_gain, not sure if we want to use cover as well, but we could try
        return gfc_gain
    
    def get_ir_img(self, sample_path, feature_scale=False):
        images_path = os.path.join(sample_path, "images")
        ir_path = os.path.join(images_path, "infrared", "composite.npy")
        ir_img = np.load(ir_path)

        # In the original code, the only scaling done is to scale the image to [0, 255]
        scaled_ir_img = self._feature_scale(ir_img, 0, 255) if feature_scale else ir_img

        ## return a 332 x 332 x 3 image, not sure why
        return scaled_ir_img

    def _feature_scale(self, x, x_min, x_max, clip=True):
        """
        Scale features from [x_min, x_max] to [0, 1] or [0, 255] range.
        
        Args:
            x: Input array or scalar value
            x_min: Minimum value of the original range
            x_max: Maximum value of the original range
            clip: Whether to clip values outside the [x_min, x_max] range
        
        Returns:
            Scaled array or scalar
        """
        # Handle None values
        if x is None:
            return 0.0
        
        # Convert to numpy array if not already
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Replace any None values with 0
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.object_):
            x = np.where(x == None, 0, x.astype(float))
        
        # Scale to [0, 1]
        if clip:
            x = np.clip(x, x_min, x_max)
        
        scaled = (x - x_min) / (x_max - x_min)
        
        return scaled

    def _get_ndvi(self, rgb_image, ir_image, feature_scale=False):
        """Calculate NDVI from RGB and IR images."""
        red_band_index = RGB_BANDS.index(RED_BAND)
        red_band = rgb_image[:, :, red_band_index].astype('float')
        ir_band_index = IR_BANDS.index(NIR_BAND)
        nir_band = ir_image[:, :, ir_band_index].astype('float')
        ndvi_unscaled = ((nir_band - red_band + 1e-6) /
                         (nir_band + red_band + 1e-6))
        # NDVI ranges between -1 and 1
        ndvi_scaled = self._feature_scale(ndvi_unscaled, -1, 1) if feature_scale else ndvi_unscaled

        # now convert to image values
        ndvi_final = self._feature_scale(ndvi_scaled, 0, 255) if feature_scale else ndvi_scaled

        ## returns a 332 x 332 image
        return ndvi_final
    
    def _get_img_stats(self, img, band_names):
        """Calculate statistics for image bands."""
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]  # Add channel dimension for 2D images
        
        # Ensure img is in (C, H, W) format
        if img.shape[-1] == len(band_names):
            img = np.transpose(img, (2, 0, 1))
        
        stats = []
        for band_idx in range(img.shape[0]):
            band_data = img[band_idx]
            stats.extend([
                band_data.min(),
                band_data.max(),
                band_data.mean(),
                band_data.std()
            ])
        return stats
    
    def get_multi_modal_features(self, row, feature_scale = False):
        sample_path = os.path.join(self.dataset_path, row["example_path"]) 

        ## constant features ##

        # simply dict of street_dist and city_dist
        osm_features = self.get_osm_features(sample_path, row["latitude"], row["longitude"], feature_scale)
        # 1-d list of 84 values, 3 values per ncep feature for 28 features
        ncep_features = self.get_ncep_features(sample_path, feature_scale)

        ## image features ## 
        # 3 x 332 x 332 image representing the srtm values for different bands
        srtm_img = self.get_srtm_img(sample_path, feature_scale)

        # 332 x 332 image representing the gfc gain band, no rescaling needed
        gfc_img = self.get_gfc_img(sample_path, feature_scale)

        # 332 x 332 x 3 image representing the infrared values
        ir_img = self.get_ir_img(sample_path, feature_scale)

        # Calculate NDVI from RGB and IR images
        image_path = os.path.join(self.dataset_path, row["example_path"], "images/visible/composite.png")
        rgb_image = np.array(Image.open(image_path).convert("RGB"))

        # 332 x 332 image representing the NDVI values
        ndvi = self._get_ndvi(rgb_image, ir_img, feature_scale)

        # Calculate statistics for each image type
        rgb_stats = self._get_img_stats(rgb_image, RGB_BANDS)
        srtm_stats = self._get_img_stats(srtm_img, ['band1', 'band2', 'band3'])
        gfc_stats = self._get_img_stats(gfc_img, ['gain'])
        ir_stats = self._get_img_stats(ir_img, IR_BANDS)
        ndvi_stats = self._get_img_stats(ndvi, ['ndvi'])

        # Flatten and combine all features
        features = [
            osm_features["street_dist"],
            osm_features["city_dist"],
            *ncep_features,  # Unpack NCEP features
            *rgb_stats,      # Unpack RGB statistics
            *srtm_stats,     # Unpack SRTM statistics
            *gfc_stats,      # Unpack GFC statistics
            *ir_stats,       # Unpack IR statistics
            *ndvi_stats      # Unpack NDVI statistics
        ]
        
        return torch.tensor(features, dtype=torch.float32)

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