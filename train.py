import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from glob import glob
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandAffined,
)

def train_unet():
    # Setup data
    data_dir = "data"
    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    
    if not images or not labels:
        print("No data found! Please ensure data/imagesTr and data/labelsTr are populated.")
        return

    data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    # 80/20 train/val split
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1)
        )
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),
    ])

    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available() else "mps")
    
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training Loop
    max_epochs = 50
    best_metric = -1
    best_metric_epoch = -1
    
    os.makedirs("weights", exist_ok=True)
    
    print(f"Starting training on {device}...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, batch_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                
                # Apply sigmoid and threshold for metric calculation
                val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "weights/best_metric_model.pth")
                print(f"Saved new best metric model! Dice: {best_metric:.4f}")

    print(f"Training complete. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    train_unet()
