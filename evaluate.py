import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import torch
from glob import glob
from monai.networks.nets import BasicUNet
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
)

def evaluate_unet():
    # Setup data (assume test data is in data/imagesTs and data/labelsTs or we use validation set)
    data_dir = "data"
    images = sorted(glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTs", "*.nii.gz")))
    
    # Fallback to Train split if Ts is not found
    if not images:
        print("Test data not found in imagesTs. Falling back to imagesTr validation split...")
        images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))[-4:]  # Last 4 as dummy test
        labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[-4:]
        
    if not images:
        print("No datasets found to evaluate.")
        return

    test_data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),
    ])
    
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available() else "mps")
    
    # Load Model
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)
    
    weights_path = "weights/best_metric_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}. Please train the model first.")
        return
        
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    total_time = 0.0
    num_scans = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_data in test_loader:
            inputs, batch_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Predict
            start_time = time.time()
            outputs = model(inputs)
            # Threshold to get binary mask
            outputs = (torch.sigmoid(outputs) > 0.5).float()
            
            # Wait for GPU sync if necessary for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
                
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            num_scans += len(inputs)
            
            # Calculate metrics
            dice_metric(y_pred=outputs, y=batch_labels)

    # Aggregate result
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    avg_inference_time = total_time / num_scans if num_scans > 0 else 0
    
    print("-" * 40)
    print(f"Total Scans Evaluated:    {num_scans}")
    print(f"Average Dice Score:       {metric:.4f}")
    if metric > 0.85:
        print("Target Dice Score (>0.85) achieved!")
    else:
        print("Target Dice Score (>0.85) NOT achieved.")
        
    print(f"Average Inference Time:   {avg_inference_time:.4f} seconds/scan")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_unet()
