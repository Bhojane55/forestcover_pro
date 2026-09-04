import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class SentinelDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, stride=64):
        self.patch_size = patch_size
        self.img_dir = os.path.join(data_dir, 'imagery')
        self.lbl_dir = os.path.join(data_dir, 'labels')
        
        # The exact 7 channels you and your teammate agreed on
        self.bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B08.tif', 'B11.tif', 'NDVI.tif', 'NDMI.tif']
        
        # 500x500 is tiny, so we load the whole thing into RAM at initialization
        self.image, self.mask = self._load_full_data()
        
        # Generate starting coordinates (y, x) for all our 128x128 patches
        self.patches = self._generate_patch_coords(stride)
        
    def _load_full_data(self):
        band_data = []
        for band in self.bands:
            band_path = os.path.join(self.img_dir, band)
            with rasterio.open(band_path) as src:
                arr = src.read(1)
                # Good practice: replace any NaN (NoData) pixels with 0
                arr = np.nan_to_num(arr)
                band_data.append(arr)
                
        # Stack into shape: (Channels=7, Height=500, Width=500)
        image = np.stack(band_data, axis=0)
        
        # Load the reference mask
        mask_path = os.path.join(self.lbl_dir, 'forest_mask.tif')
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask = np.nan_to_num(mask)
            
        return image, mask
        
    def _generate_patch_coords(self, stride):
        _, h, w = self.image.shape
        coords = []
        
        # Slide a window across the 500x500 image to generate overlapping patches
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                coords.append((y, x))
                
        return coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        
        # Slice the tensors using our generated patch coordinates
        img_patch = self.image[:, y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = self.mask[y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img_patch).float()
        
        # Add a channel dimension to the mask so it's (1, 128, 128)
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)
        
        return img_tensor, mask_tensor

# --- Quick Test Code ---
if __name__ == "__main__":
    # If you run this file directly, it will test the dataset loader
    data_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    dataset = SentinelDataset(data_directory, patch_size=128, stride=64)
    print(f"Total overlapping patches generated: {len(dataset)}")
    
    img, mask = dataset[0]
    print(f"Image tensor shape: {img.shape} (Expected: 7, 128, 128)")
    print(f"Mask tensor shape: {mask.shape} (Expected: 1, 128, 128)")
