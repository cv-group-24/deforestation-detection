import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from torch.utils.data import Dataset
# from data.transforms import get_transforms

from geopy import distance

import pickle
import matplotlib.pyplot as plt
import json

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

        # # Get the combined transformations
        # self.augmentations = get_transforms(
        #     resize=self.resize,
        #     spatial_augmentation=self.spatial_augmentation,
        #     pixel_augmentation=self.pixel_augmentation,
        #     is_training=self.is_training,
        #     use_landsat=self.use_landsat
        # )
        self.augmentations = None

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

            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx} from path {image_path}: {e}")
            raise e
        

class BaseFeatureHandler:

    def _feature_scale(self, arr, min_val, max_val, rescale=True):
        if arr is None: #AD
            arr = 0.0

        elif type(arr) == float and arr is not None: #AD
            arr= (arr- min_val) / (max_val - min_val)

        elif arr.size == 1 and arr != None: #AD
            arr = (arr - min_val) / (max_val - min_val)

        elif type(arr) != float and arr.size > 1: #AD
            if len(arr.shape) == 1:
                for i in range(arr.shape[0]):
                    if arr[i] == None:
                        arr[i] = 0

            elif len(arr.shape) == 2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if arr[i][j] == None:
                            arr[i][j] = 0

            elif len(arr.shape) == 3:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        for k in range(arr.shape[2]):
                            if arr[i][j][k] == None:
                                arr[i][j][k] = 0

            arr = (arr - min_val) / (max_val - min_val)
            
        if rescale:
            arr = (arr * 255) #AD
            arr= np.uint8(arr) #AD
            
        return arr
    

class ConstantFeatureHandler(BaseFeatureHandler):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        ## Constants
        self.OSM_SCALING = {
            'city': (0.19590, 513.49534),
            'street': (0.00327, 513.49534)
        }

    def _get_constant_features(self, row):
        """
        Get constant features for a given row in the dataset.
        
        Args:
            row (pd.Series): Row in the dataset.
        
        Returns:
            dict: Dictionary of constant features.
        """
        lat = row["lat"]
        lon = row["lon"]

        sample = row["example_path"]
        sample_path = os.path.join(self.dataset_path, sample)
        aux_path = os.path.join(sample_path, "auxiliary")
        
        features = {}
        
        # Get constant features
        features.update(self._get_peat_features(aux_path))
        features.update(self._get_osm_features(aux_path, lat, lon))
        features.update(self._get_ncep_features(aux_path))
        
        return features

    def _get_peat_features(self, aux_path):
        peat_path = os.path.join(aux_path, 'peat.json')
        peat = json.loads(open(peat_path).read())['peat']
        peat_val = 1. if peat else 0.
        return {'peat': peat_val}
    
    # Using geodesic distance provided by geopy
    # See: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
    def _get_osm_features(self, aux_path, lat, lon, feature_scale=False):
        # street_path = os.path.join(aux_path, 'closest_street.json')
        # street = json.loads(open(street_path).read())
        # city_path = os.path.join(aux_path, 'closest_city.json')        
        # city = json.loads(open(city_path).read())
        osm_path = os.path.join(aux_path, 'osm.json')
        osm = json.loads(open(osm_path).read())
        street = osm['closest_street']
        city = osm['closest_city']
        street_dist = distance.distance((lat, lon),
                                        (street.get('lat'), street.get('lon'))).km                                  
        city_dist = distance.distance((lat, lon),
                                      (city.get('lat'), city.get('lon'))).km
                                      
        if feature_scale:
            street_min, street_max = self.OSM_SCALING['street']
            street_dist = self._feature_scale(street_dist, street_min, street_max, False)
            city_min, city_max = self.OSM_SCALING['city']
            city_dist = self._feature_scale(city_dist, city_min, city_max, False)
                                      
        features = {'street_dist': street_dist,
                    'city_dist': city_dist}        
        return features  

    def _get_ncep_features(self, aux_path, feature_scale=False):
        ## TODO figure out tf is going on in their implementation
        ## commented below, but it's weird because they take 'feat_name.npy' which 
        ## doesn't exist, as there is only a ncep.npy file in the directory
        # features = {}
        # ncep_path = os.path.join(aux_path, NCEP_DIR)
        # for feat_name in NCEP_FEATURES:
        #     path = os.path.join(ncep_path, f'{feat_name}.npy')
        #     feat = np.load(path, allow_pickle=True) #AD
            
        #     # NOTE: Pretty hacky right now for retreiving feature scaling keys 
        #     # from feature names, should fix this eventually. 
        #     if feature_scale:
        #         if feat_name in TEMP_FEATURES:
        #             min_scale, max_scale = NCEP_SCALING[feat_name]
        #         else:
        #             min_scale, max_scale = NCEP_SCALING[feat_name[:-4]]
        #         feat = self._feature_scale(feat, min_scale, max_scale, False)
            
        #     features[feat_name] = feat
        features = {
            'ncep': None
        }
        
        return features
    

class ImageFeatureHandler(BaseFeatureHandler):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        ## Constants 
        self.TREE_COVER_SCALING = (0.0, 100.0)
        self.RGB_BANDS = ['red', 'green', 'blue']
        self.IR_BANDS = ['nir', 'swir1', 'swir2']
        self.NDVI_BAND = 'ndvi'
        self.SRTM_BANDS = ['altitude', 'aspect', 'slope']
        self.GFC_GAIN_BAND = 'gain'
        self.RED_BAND = 'red'
        self.NIR_BAND = 'nir'
        self.SRTM_SCALING = {
            'altitude': (-23.0, 4228.0),
            'slope': (0.0, 8421.0),
            'aspect': (-17954.0, 18000.0)
        }

    def _get_image_features(self, row, feature_scale=False):

        sample = row["example_path"]
        sample_path = os.path.join(self.dataset_path, sample)
        aux_path = os.path.join(sample_path, "auxiliary")
        images_path = os.path.join(sample_path, "images")
        image_path = os.path.join(images_path, "visible", "composite.png")

        rgb_img = self._get_image(image_path)
        ir_img = self._get_ir(image_path)
        srtm_img = self._get_srtm_img(aux_path)
        ndvi_img = self._get_ndvi(rgb_img, ir_img)[..., np.newaxis]
        gfc_gain_img = self._get_gfc_img(aux_path)
        #gfc_cover_img = self._get_gfc_img(aux_path, GFC_COVER_BAND) #AD: modified
        
        mask = self._get_mask(sample_path)
              
        if feature_scale:
            # Scale to [0-1] without re-scaling.
            # GFC Gain already [0-1].
            rgb_img = self._feature_scale(rgb_img, 0, 255, False)
            ir_img = self._feature_scale(ir_img, 0, 255, False)
            srtm_img = self._feature_scale(srtm_img, 0, 255, False)
            ndvi_img = self._feature_scale(ndvi_img, 0, 255, False)            
            cover_min, cover_max = self.TREE_COVER_SCALING
            # gfc_cover_img = self._feature_scale(gfc_cover_img, cover_min, cover_max, False) #AD: modified
            
        # gfc_img = np.stack([gfc_gain_img, gfc_cover_img], axis=2) #AD: modified
                
        imgs = {
            'rgb': {'image': rgb_img, 'band_names': self.RGB_BANDS},
            'ir': {'image': ir_img, 'band_names': self.IR_BANDS},
            'ndvi': {'image': ndvi_img, 'band_names': [self.NDVI_BAND]},
            'srtm': {'image': srtm_img, 'band_names': self.SRTM_BANDS}, 
            'gfc': {'image': gfc_gain_img, 'band_names': [self.GFC_GAIN_BAND]} #AD: modified
            }         
        return imgs, mask
    
    def _get_image(self, im_path, feature_scale=False):
        pil_image = Image.open(im_path).convert('RGB')
        pil_image = ImageEnhance.Brightness(pil_image).enhance(1.5)
        image = np.array(pil_image)
        
        if feature_scale:
            image = self._feature_scale()
        
        return image
    
    def _get_ir(self, im_path, feature_scale=False):
        ir_path = im_path.replace('visible', 'infrared')
        if 'small_composite' in im_path or 'full_composite' in im_path:
            ir_path += '.npy'
        else:
            #ir_path = ir_path.replace('png', 'npy') #AD
            ir_name = str(os.path.split(ir_path)[1][0:4]) + '_ir_0.npy' #AD
            ir_path = os.path.join(os.path.dirname(ir_path), ir_name) #AD
        ir = np.load(ir_path).astype(np.uint8)
        return ir
    
    def _get_gfc_img(self, aux_path, band_name):
        gfc_img = np.load(os.path.join(aux_path, f'{band_name}.npy'), allow_pickle=True )[0] #AD
        return gfc_img         
                
    def _get_srtm_img(self, aux_path):
        srtm_img = np.stack([self._get_srtm(aux_path, band) for band in self.SRTM_BANDS], axis = 2) #AD: remove axis = 2
        return srtm_img 
    
    def _get_srtm(self, aux_path, srtm_band):
        srtm_unscaled = np.load(os.path.join(aux_path, f'{srtm_band}.npy'))
        srtm_min, srtm_max = self.SRTM_SCALING[srtm_band]
        srtm_scaled = self._feature_scale(srtm_unscaled, srtm_min, srtm_max)
        return srtm_scaled 

    def _get_ndvi(self, rgb_image, ir_image):
        red_band_index = self.RGB_BANDS.index(self.RED_BAND)
        red_band = rgb_image[:, :, red_band_index].astype('float')
        ir_band_index = self.IR_BANDS.index(self.NIR_BAND)
        nir_band = ir_image[:, :, ir_band_index].astype('float')
        ndvi_unscaled = ((nir_band - red_band + 1e-6) /
                         (nir_band + red_band + 1e-6))
        # NDVI ranges between -1 and 1
        ndvi_scaled = self._feature_scale(
            ndvi_unscaled, -1, 1
        )

        return ndvi_scaled  
        
            
    def _get_mask(self, sample_path, debug=False):
        """
        Generate a mask with debugging information
        """

        pil_image = Image.open(self.image_path).convert('RGB')
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
            print(f"Error loading polygon for {self.dataset_path}: {e}")
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
                        outline=1, fill=1)
                any_polygon_drawn = True

            elif hasattr(polygon, 'geoms'):
                # Handle MultiPolygon
                for poly in polygon.geoms:
                    coords = np.array(poly.exterior.coords)
                    
                    draw.polygon([tuple(coord) for coord in coords],
                        outline=1, fill=1)
                    any_polygon_drawn = True

        except Exception as e:
            print(f"Error creating polygon for {self.dataset_path}: {e}")
            if debug:
                print(f"Exception details: {e}")
            return np.ones((height, width), dtype=bool)
        
        except Exception as e:
            print(f"Error creating polygon for {self.dataset_path}: {e}")
            if debug:
                print(f"Exception details: {e}")
            return np.ones((height, width), dtype=bool)
        
        # Convert mask to numpy array
        mask_np = np.array(mask)
        
        # Set ignored values to 0
        mask_np[mask_np == 255] = 0
        mask_np[mask_np != 0] = 1
        
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
    
if __name__ == "__main__":
    ## instantiate the handlers so that I can test them
    dataset_path = r"C:\Users\chris\Documents\DSAIT\CV - Q3\deforestation-detection\data\raw\ForestNetDataset"
    image_handler = ImageFeatureHandler(dataset_path)
    constant_handler = ConstantFeatureHandler(dataset_path)

    row = {
        "example_path": "examples/4.430849118860583_96.1016343478138",
        "lat": 4.430849118860583,
        "lon": 96.1016343478138
    }

    ## test the _get_constant_features method
    constant_features = constant_handler._get_constant_features(row)
    print(constant_features)

    ## test the _get_image_features method
    image_features, mask = image_handler._get_image_features(row)

