import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import os
from dataset_reader import SegmentationDataset, PolarDataset  # 請放同一目錄下

# ----------- Siamese Dataset Wrapper ----------- #
class SiameseDataset(torch.utils.data.Dataset):
    """
    接收一個 PolarDataset（base_dataset），
    每次 __getitem__ 隨機傳回一對 (imgA, imgB) 及 label（同人=1，不同人=0）。
    假設路徑格式： train_dataset/DataSetName/personID/eye/L or R/filename.jpg
    → personID 視為同一人的依據，即 path.split('/')[2]
    """
    def __init__(self, base_dataset: PolarDataset):
        self.base_dataset = base_dataset
        # 建立 person_id → 該 person 底下的 index 列表
        from collections import defaultdict
        label_dict = defaultdict(list)
        for idx, rel_path in enumerate(self.base_dataset.image_paths):
            # e.g. "train_dataset/CASIA-Iris-Lamp/102/L/S2102L08.jpg"
            parts = rel_path.split('/')
            person_id = parts[2]  # index=2 才是 personID
            label_dict[person_id].append(idx)
        self.labels  = label_dict
        self.indices = list(range(len(self.base_dataset)))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. anchor
        img1_tensor, rel_path1 = self.base_dataset[idx]
        parts = rel_path1.split('/')
        person1 = parts[2]

        # 2. 隨機決定 positive(同人) or negative(異人)
        if random.random() < 0.5:
            # positive：同一 person1 底下隨機一張（不等於 idx）
            idx2 = idx
            while idx2 == idx:
                idx2 = random.choice(self.labels[person1])
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            # negative：隨機選一個與 person1 不同的 person2
            person2 = random.choice(list(self.labels.keys()))
            while person2 == person1:
                person2 = random.choice(list(self.labels.keys()))
            idx2 = random.choice(self.labels[person2])
            label = torch.tensor(0.0, dtype=torch.float32)

        img2_tensor, rel_path2 = self.base_dataset[idx2]
        # 回傳 (img1, img2) 與 label
        return (img1_tensor, img2_tensor), label


# ----------- 定義 Siamese Network 架構 ----------- #
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 你可自行調整 CNN 架構，以下示範一個簡單版本
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     # → [32,128,128]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),     # → [64,64,64]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),     # → [128,32,32]

            nn.Flatten(),        # → [128*32*32]
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 最後 embedding vector 長度 128
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, img1, img2):
        f1 = self.forward_once(img1)  # [batch,128]
        f2 = self.forward_once(img2)
        return f1, f2


# ----------- 對比損失 Contrastive Loss ----------- #
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, f1, f2, label):
        # label=1 → 同人，要把距離拉小； label=0 → 異人，要把距離拉大
        dist = nn.functional.pairwise_distance(f1, f2, keepdim=True)  # [batch,1]
        loss_pos = label * (dist ** 2)
        loss_neg = (1 - label) * torch.clamp(self.margin - dist, min=0.0) ** 2
        return torch.mean(loss_pos + loss_neg)


# ----------- Training Loop (main) ----------- #
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ---- 1. 準備訓練集 & 驗證集 PolarDataset ----
    train_list = "train_fixed.txt"
    val_list   = "val_fixed.txt"  # 你自行準備的 validation 列表

    # PolarDataset 會自動做 RITNet segmentation + iris normalization → 回傳 [1,256,256]
    polar_train = PolarDataset(
        list_file=train_list,
        root_dir=".",  # 代表 train_fixed.txt 裡面寫的相對路徑，都以 "." 為基底
        ritnet_model_path="../RITnet/best_model.pkl"
    )
    polar_val = PolarDataset(
        list_file=val_list,
        root_dir=".",
        ritnet_model_path="../RITnet/best_model.pkl"
    )

    # 再把它們包成 SiameseDataset
    siamese_train = SiameseDataset(polar_train)
    siamese_val   = SiameseDataset(polar_val)

    train_loader = DataLoader(siamese_train, batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(siamese_val,   batch_size=16, shuffle=False, num_workers=2)

    # ---- 2. 定義網路 & 損失函數 & 優化器 ----
    model     = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    for epoch in range(num_epochs):
        # ===== Train =====
        model.train()
        total_loss = 0.0
        for (img1, img2), label in train_loader:
            img1 = img1.to(device)  # [batch,1,256,256]
            img2 = img2.to(device)
            label = label.to(device).unsqueeze(1)  # [batch,1]

            f1, f2 = model(img1, img2)
            loss = criterion(f1, f2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1:02d}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        scheduler.step()

        # 每 5 個 epoch 存一次模型（可選）
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"siamese_epoch{epoch+1}.pth")

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for (img1, img2), label in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device).unsqueeze(1)

                f1, f2 = model(img1, img2)
                loss = criterion(f1, f2, label)
                val_loss += loss.item()

                # 計算 distance，再用 threshold 決定是否判為同人
                dist = nn.functional.pairwise_distance(f1, f2)
                preds = (dist < 0.5).float()  # threshold 可自行調整
                correct += (preds == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch+1:02d}/{num_epochs}] Val   Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---- 訓練結束後存 final model ----
    torch.save(model.state_dict(), "siamese_final.pth")
    print("Training completed. Model saved to siamese_final.pth")

if __name__ == "__main__":
    main()
