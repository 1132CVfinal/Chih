import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_reader import SegmentationDataset
from densenet import DenseNet2D
import os

# Hyperparameters
batch_size = 8
num_epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & DataLoader
train_dataset = SegmentationDataset('train.txt')
val_dataset = SegmentationDataset('val.txt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = DenseNet2D(dropout=True, prob=0.2).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
best_val_loss = float('inf')
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model_segmentation.pth")
        print("✅ Saved new best model.")

print("Training completed.")
