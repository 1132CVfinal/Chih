import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset_reader import SegmentationDataset  # 使用你的 SegmentationDataset

# ----------- Simple Siamese Network ----------- #
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 256),  # 假設 input image size 是 256x256
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# ----------- Contrastive Loss ----------- #
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# ----------- Pair Generator Dataset ----------- #
import random
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = [os.path.basename(p).split('_')[0] for p in base_dataset.image_paths]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img1, _ = self.base_dataset[idx]
        label1 = self.labels[idx]

        should_get_same_class = random.randint(0, 1)
        while True:
            idx2 = random.randint(0, len(self.base_dataset) - 1)
            label2 = self.labels[idx2]
            if (label1 == label2) == bool(should_get_same_class):
                break

        img2, _ = self.base_dataset[idx2]
        label = torch.tensor([int(label1 != label2)], dtype=torch.float32)
        return img1, img2, label

# ----------- Training Loop ----------- #
def train():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    base_dataset = SegmentationDataset('train.txt', transform=None)
    train_dataset = SiameseDataset(base_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0.0
        model.train()
        for img1, img2, label in train_loader:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'siamese_model.pth')
    print("Model saved to siamese_model.pth")

if __name__ == '__main__':
    train()
