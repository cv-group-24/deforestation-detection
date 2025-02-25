import os
import pickle
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np

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

def get_mask(sample_path, label):
    sample_dir = os.path.join(FORESTNET_DIR, 'examples', sample_path)

    im_path = os.path.join(sample_dir, 'images/visible/composite.png')
    pil_image = Image.open(im_path).convert('RGB')
    pil_image = ImageEnhance.Brightness(pil_image).enhance(1.5)
    rgb_image = np.array(pil_image)

    polygon = None
    with open(os.path.join(sample_dir,'forest_loss_region.pkl'), 'rb') as f: #AD: MODIFIED
        polygon = pickle.load(f)
    
    mask = Image.new('L', rgb_image.shape[:2], LABEL_IGNORE_VALUE)

    draw = ImageDraw.Draw(mask)

    shape_type = polygon.geom_type
    if label is not None:
        label = int(label)
    #Need to convert lon/lat shape in coordinates matrix (AD)
    lon = polygon.centroid.xy[0][0]
    lat = polygon.centroid.xy[1][0]
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    m_per_deg_lon = 111132.954 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    res = 4.77 #TO CHANGE WITH SENSOR (15 FOR LANDSAT AND 4.77 FOR PLANETSCOPE)
    deg_lat = (332 * res * 0.5) / m_per_deg_lat #AD
    deg_lon = (332 * res * 0.5) / m_per_deg_lon #AD

    div_lon = 2 * deg_lon / 332 #AD
    div_lat = 2 * deg_lat / 332 #AD

    if shape_type == 'Polygon':
        coords = np.array(polygon.exterior.coords)
        for coord in coords:
            coord[0] = int((coord[0] - lon + deg_lon) / div_lon)
            coord[1] = int((coord[1] - lat + deg_lat) / div_lat)

        draw.polygon([tuple(coord) for coord in coords],
                     outline=label, fill=label)
    elif shape_type == 'MultiPolygon':
        for poly in polygon.geoms:  # Changed from 'polygon' to 'polygon.geoms'
            coords = np.array(poly.exterior.coords)
            for coord in coords:
                coord[0] = int((coord[0] - lon + deg_lon) / div_lon)
                coord[1] = int((coord[1] - lat + deg_lat) / div_lat)

            draw.polygon([tuple(coord) for coord in coords],
                         outline=label, fill=label)
    else:
        for poly in polygon.geoms:  # Changed from 'polygon' to 'polygon.geoms' 
            coords = np.array(poly.exterior.coords)
            for coord in coords:
                coord[0] = int((coord[0] - lon + deg_lon) / div_lon)
                coord[1] = int((coord[1] - lat + deg_lat) / div_lat)

            draw.polygon([tuple(coord) for coord in coords],
                         outline=label, fill=label)
            
    mask = np.array(mask).astype(np.int32)

    mask[mask == LABEL_IGNORE_VALUE] = 0
    mask[mask != 0] = 1        
    mask = mask.squeeze().astype(bool)
    return mask

label = INDONESIA_ALL_LABELS.index('Timber plantation')
mask = get_mask('4.430849118860583_96.1016343478138', label)