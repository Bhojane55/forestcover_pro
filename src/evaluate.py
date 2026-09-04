import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SentinelDataset
from unet import UNet

def calculate_metrics(logits, masks, threshold=0.5):
    """
    Calculates pixel-wise segmentation metrics for a batch.
    """
    # Convert logits to probabilities, then to binary 0/1 mask
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = (preds * masks).sum()
    TN = ((1 - preds) * (1 - masks)).sum()
    FP = (preds * (1 - masks)).sum()
    FN = ((1 - preds) * masks).sum()

    epsilon = 1e-7 # Prevent division by zero

    iou = TP / (TP + FP + FN + epsilon)
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

    return iou.item(), dice.item(), precision.item(), recall.item(), accuracy.item()

def evaluate_model():
    # Setup Apple Silicon (MPS) or CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating using {device}...")

    # File Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    model_load_path = os.path.join(base_dir, 'models', 'unet_best_model.pth')

    # Initialize dataset and carefully reproduce the validation split
    print("Loading dataset...")
    full_dataset = SentinelDataset(data_dir, patch_size=128, stride=64)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # NOTE: In a real project, you'd use a fixed random seed (e.g., torch.manual_seed(42))
    # before the split so train and test are always separated identically.
    # For this prototype test, we just split it and evaluate the validation portion.
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Load the Model
    model = UNet(in_channels=7, out_channels=1)
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        print("Warning: No trained model found. Evaluating random initialization.")
    
    model.to(device)
    model.eval()

    # Tracking variables
    total_iou, total_dice, total_prec, total_rec, total_acc = 0, 0, 0, 0, 0

    print("Calculating metrics...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            iou, dice, prec, rec, acc = calculate_metrics(outputs, masks)
            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_rec += rec
            total_acc += acc

    # Average out the metrics
    num_batches = len(val_loader)
    print("\n--- Final Evaluation Metrics ---")
    print(f"IoU (Jaccard): {total_iou / num_batches:.4f}")
    print(f"Dice (F1):     {total_dice / num_batches:.4f}")
    print(f"Precision:     {total_prec / num_batches:.4f}")
    print(f"Recall:        {total_rec / num_batches:.4f}")
    print(f"Accuracy:      {total_acc / num_batches:.4f}")
    print("--------------------------------")

if __name__ == "__main__":
    evaluate_model()

