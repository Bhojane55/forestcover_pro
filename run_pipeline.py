import subprocess
import sys
import os

def run_step(script_name, description):
    print(f"\n{'='*55}")
    print(f"🚀 STEP: {description}")
    print(f"📂 Running: src/{script_name}")
    print(f"{'='*55}\n")
    
    # sys.executable ensures it uses the exact same Python (python3, venv, etc.) 
    # that the user used to launch this master script.
    result = subprocess.run([sys.executable, f"src/{script_name}"])
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: src/{script_name} crashed. Stopping pipeline.")
        sys.exit(1)

def main():
    print("\n🌲 INITIALIZING FOREST COVER & CARBON ML PIPELINE 🌲\n")
    
    # Ensure the script is run from the root directory
    if not os.path.exists("src/train.py"):
        print("❌ ERROR: Please run this script from the root 'forest-cover-carbon-mapping' folder.")
        sys.exit(1)

    # Run the pipeline in order
    run_step("train.py", "Training U-Net Model")
    run_step("evaluate.py", "Evaluating Model Metrics")
    run_step("predict.py", "Generating Forest Mask GeoTIFF")
    run_step("carbon.py", "Estimating Carbon Storage")
    
    print("\n✅ PIPELINE COMPLETELY SUCCESSFULLY!")
    print("The final map is in 'outputs/maps/' and the report is in 'outputs/metrics/'.\n")

if __name__ == "__main__":
    main()
