import os
import pickle
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

FORESTNET_DIR = 'C:\\Users\\chris\\Documents\\DSAIT\\CV - Q3\\deforestation-detection\data\\deep\\downloads\\ForestNetDataset'
LABEL_IGNORE_VALUE = 255

INDONESIA_ALL_LABELS = [
    'Oil palm plantation',
    'Timber plantation',
    'Other large-scale plantations',
    'Grassland shrubland',
    'Small-scale agriculture',
    'Small-scale mixed plantation',
    'Small-scale oil palm plantation',
    'Mining',
    'Fish pond',
    'Logging',
    'Secondary forest',
    'Other',
    '?', #AD
    'Rubber plantation', #AD
    'Timber sales', #AD
    'Fruit plantation', #AD
    'Wildfire', #AD
    'Small-scale cassava plantation', #AD
    'Small-scale maize plantation', #AD
    'Small-scale other plantation', #AD
    'Infrastructure', #AD
    'Hunting', #AD
    'Selective logging'] #AD

def get_mask(sample_path, label, debug=False):
    """
    Generate a mask with debugging information
    """
    sample_dir = os.path.join(FORESTNET_DIR, 'examples', sample_path)
    
    im_path = os.path.join(sample_dir, 'images/visible/composite.png')
    pil_image = Image.open(im_path).convert('RGB')
    rgb_image = np.array(pil_image)
    height, width = rgb_image.shape[:2]
    
    if debug:
        print(f"Image dimensions: {width}x{height}")
    
    polygon = None
    try:
        with open(os.path.join(sample_dir, 'forest_loss_region.pkl'), 'rb') as f:
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
    mask = Image.new('L', (width, height), LABEL_IGNORE_VALUE)
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
    mask_np[mask_np == LABEL_IGNORE_VALUE] = 0
    
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

label = INDONESIA_ALL_LABELS.index('Timber plantation')
mask = get_mask('1.2673855496545907_118.13648785567229', label, debug=True)