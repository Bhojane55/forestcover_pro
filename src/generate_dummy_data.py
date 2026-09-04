import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_geotiff(filepath, width, height, dtype, is_mask=False):
    # 10m pixel size mapping
    transform = from_origin(300000.0, 2100000.0, 10.0, 10.0)
    
    with rasterio.open(
        filepath, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=dtype,
        crs='EPSG:32643', # UTM Zone 43N (Covers Mumbai/western India)
        transform=transform,
    ) as dst:
        if is_mask:
            # Reference mask: 0 (non-forest) and 1 (forest)
            data = np.random.randint(0, 2, (height, width)).astype(dtype)
        else:
            # Spectral bands: random floats 0 to 1
            data = np.random.rand(height, width).astype(dtype)
        dst.write(data, 1)

def main():
    # Dynamically find the data directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'data', 'imagery')
    lbl_dir = os.path.join(base_dir, 'data', 'labels')
    
    # 5km x 5km at 10m resolution = 500x500 pixels
    width, height = 500, 500
    bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B08.tif', 'B11.tif', 'NDVI.tif', 'NDMI.tif']
    
    print("Generating 500x500 Sentinel-2 bands...")
    for band in bands:
        create_dummy_geotiff(os.path.join(img_dir, band), width, height, rasterio.float32)
        
    print("Generating reference forest mask...")
    create_dummy_geotiff(os.path.join(lbl_dir, 'forest_mask.tif'), width, height, rasterio.uint8, is_mask=True)
    print("Success! Dummy GeoTIFFs created in data/ directory.")

if __name__ == '__main__':
    main()
