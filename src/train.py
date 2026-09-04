import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import our custom modules
from dataset import SentinelDataset
from unet import UNet

# --- 1. Define Loss Function ---
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch's BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        # 1. Calculate BCE Loss
        bce_loss = self.bce(logits, targets)
        
        # 2. Calculate Dice Loss
        probs = torch.sigmoid(logits) # Convert logits to probabilities
        smooth = 1e-6
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (probs_flat.sum() + targets_flat.sum() + smooth))
        
        return bce_loss + dice_loss

# --- 2. Main Training Loop ---
def train_model():
    # Setup Apple Silicon (MPS) or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Accelerating using Apple Silicon (MPS)...")
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    # File Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    model_save_path = os.path.join(base_dir, 'models', 'unet_best_model.pth')

    # Load & Split Dataset (80% Train, 20% Val)
    print("Loading and normalizing dataset...")
    full_dataset = SentinelDataset(data_dir, patch_size=128, stride=64)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total patches: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    # DataLoaders manage batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = UNet(in_channels=7, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5 # Small number for prototype testing
    best_val_loss = float('inf')

    print("\nStarting Training Loop...")
    for epoch in range(num_epochs):
        # --- Train Phase ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save if model improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[*] Best model saved to {model_save_path}\n")

if __name__ == "__main__":
    train_model()
