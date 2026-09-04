import os
import torch
import torch.nn.functional as F
import numpy as np
import rasterio

from unet import UNet

def predict_full_image():
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # File Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'data', 'imagery')
    model_path = os.path.join(base_dir, 'models', 'unet_best_model.pth')
    output_path = os.path.join(base_dir, 'outputs', 'maps', 'predicted_forest_mask.tif')

    bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B08.tif', 'B11.tif', 'NDVI.tif', 'NDMI.tif']

    # 1. Load and Normalize the Full 500x500 Image
    print("Loading imagery...")
    band_data = []
    meta = None
    
    for i, band in enumerate(bands):
        band_path = os.path.join(img_dir, band)
        with rasterio.open(band_path) as src:
            # Save the geospatial metadata from the first band so our output aligns in QGIS
            if i == 0:
                meta = src.meta.copy()
                
            arr = src.read(1).astype(np.float32)
            arr = np.nan_to_num(arr)
            
            # Min-Max Normalization (same as training)
            b_min, b_max = arr.min(), arr.max()
            if b_max > b_min:
                arr = (arr - b_min) / (b_max - b_min)
            else:
                arr = np.zeros_like(arr)
                
            band_data.append(arr)

    # Shape: (7, 500, 500)
    image = np.stack(band_data, axis=0)
    
    # Convert to Tensor and add Batch dimension -> Shape: (1, 7, 500, 500)
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    # 2. Pad to nearest multiple of 16 (512x512)
    # padding format for F.pad is (left, right, top, bottom)
    # We need 12 pixels on the right and 12 on the bottom
    padded_tensor = F.pad(image_tensor, (0, 12, 0, 12), mode='constant', value=0)
    
    # 3. Load Model
    model = UNet(in_channels=7, out_channels=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")
    model.eval()

    # 4. Run Inference
    print("Running U-Net prediction...")
    with torch.no_grad():
        logits = model(padded_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float() # Binary threshold
        
    # 5. Crop back to 500x500 and convert to numpy
    # preds shape is (1, 1, 512, 512). We crop the last two dimensions.
    final_mask = preds[0, 0, :500, :500].cpu().numpy().astype(rasterio.uint8)

    # 6. Save as GeoTIFF
    print("Saving prediction to GeoTIFF...")
    meta.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=None
    )
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(final_mask, 1)
        
    print(f"Success! Map saved to: {output_path}")

if __name__ == "__main__":
    predict_full_image()

