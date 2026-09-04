import os
import rasterio
import numpy as np
import json

def calculate_forest_area(mask_path, pixel_resolution=10):
    """
    Reads the predicted GeoTIFF and calculates the total forest area.
    pixel_resolution: 10 meters for Sentinel-2
    """
    print(f"Reading mask from: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        
        # Count how many pixels were classified as forest (1)
        forest_pixels = np.sum(mask == 1)
        
        # Calculate area
        # 1 pixel = 10m * 10m = 100 sq meters
        pixel_area_sqm = pixel_resolution ** 2
        total_area_sqm = forest_pixels * pixel_area_sqm
        
        # Convert to Hectares (1 Hectare = 10,000 sq meters)
        total_area_ha = total_area_sqm / 10000.0
        
        return int(forest_pixels), total_area_ha

def estimate_carbon(area_ha, carbon_density_t_ha=72.56):
    """
    Estimates total carbon stored based on forest area.
    
    Parameters:
    - area_ha: Total forest area in hectares
    - carbon_density_t_ha: Biomass/Carbon density parameter. 
      Default is 72.56 tC/ha (Source: Forest Survey of India, ISFR 2021 average)
    """
    total_carbon_tonnes = area_ha * carbon_density_t_ha
    return total_carbon_tonnes

def main():
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mask_path = os.path.join(base_dir, 'outputs', 'maps', 'predicted_forest_mask.tif')
    report_path = os.path.join(base_dir, 'outputs', 'metrics', 'carbon_report.json')

    if not os.path.exists(mask_path):
        print("Error: Predicted mask not found. Please run predict.py first.")
        return

    # 1. Calculate Area
    print("Calculating total forest area...")
    pixel_count, area_ha = calculate_forest_area(mask_path)
    
    # 2. Estimate Carbon
    print("Estimating carbon stock...")
    # You can override the 72.56 default by passing a different value here later
    carbon_density = 72.56 
    total_carbon = estimate_carbon(area_ha, carbon_density_t_ha=carbon_density)

    # 3. Print Results
    print("\n" + "="*40)
    print("🌍 FOREST CARBON ESTIMATION REPORT 🌍")
    print("="*40)
    print(f"Study Area Size      : 5 km x 5 km (2500 Hectares)")
    print(f"Total Forest Pixels  : {pixel_count:,} pixels")
    print(f"Total Forest Area    : {area_ha:,.2f} Hectares")
    print(f"Carbon Density Used  : {carbon_density} tonnes/Hectare (ISFR 2021)")
    print(f"Total Estimated Carbon: {total_carbon:,.2f} Tonnes")
    print("="*40 + "\n")

    # 4. Save to JSON for later use in dashboards/visualizations
    report_data = {
        "methodology_reference": "India State of Forest Report (ISFR) 2021 Average",
        "carbon_density_t_ha": carbon_density,
        "forest_pixels": pixel_count,
        "forest_area_hectares": round(area_ha, 2),
        "total_estimated_carbon_tonnes": round(total_carbon, 2)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()

