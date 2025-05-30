import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
from torchvision import transforms
import os

from dataset_reader import SegmentationDataset  # 只用到 image，不用 mask

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
            nn.Linear(64 * 64 * 64, 256),  # 輸入為 256×256，經兩次 MaxPool → 64×64
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
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


# ----------- SiameseDataset：從 SegmentationDataset 產生正負 pair ----------- #
class SiameseDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: SegmentationDataset，__getitem__ 回傳 (Tensor_image, Tensor_mask)
        我們只用到 image → base_dataset[idx][0]
        """
        self.base_dataset = base_dataset
        # 解析 person_id；假設路徑格式為 'CASIA-Iris-Thousand/447/L/S5447L00.jpg'
        self.labels = [path.split('/')[1] for path in base_dataset.image_paths]

        # 把相同 person_id 的索引分組
        from collections import defaultdict
        label_dict = defaultdict(list)
        for i, person_id in enumerate(self.labels):
            label_dict[person_id].append(i)
        self.label_dict = label_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 取第一張影像 Tensor
        img1, _ = self.base_dataset[idx]  # img1 形狀 [1,256,256]
        person1 = self.labels[idx]

        # 隨機決定正樣本（同人）還是負樣本（異人）
        if random.random() < 0.5:
            # positive pair
            idx2 = idx
            while idx2 == idx:
                idx2 = random.choice(self.label_dict[person1])
            label = 1.0
        else:
            # negative pair
            person2 = random.choice(list(self.label_dict.keys()))
            while person2 == person1:
                person2 = random.choice(list(self.label_dict.keys()))
            idx2 = random.choice(self.label_dict[person2])
            label = 0.0

        img2, _ = self.base_dataset[idx2]  # img2 形狀 [1,256,256]
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ----------- SiamesePairsDataset：讀 val_pairs.txt (p1, p2, label) ----------- #
class SiamesePairsDataset(Dataset):
    def __init__(self, pair_txt, transform=None, root_dir=''):
        """
        pair_txt: 每行 "相對路徑1 相對路徑2 label"
                  EX: "CASIA-Iris-Thousand/447/L/S5447L00.jpg CASIA-Iris-Thousand/447/L/S5447L01.jpg 1"
        transform: torchvision.transforms，只處理 PIL Image
        root_dir: 如果 pair_txt 中路徑不是絕對路徑，就在前面加上 root_dir
        """
        self.pairs = []
        self.transform = transform
        self.root_dir = root_dir

        with open(pair_txt, 'r') as f:
            for line in f:
                p1, p2, label = line.strip().split()
                if not os.path.isabs(p1):
                    p1 = os.path.join(self.root_dir, p1)
                if not os.path.isabs(p2):
                    p2 = os.path.join(self.root_dir, p2)
                self.pairs.append((p1, p2, int(label)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert('L')
        img2 = Image.open(p2).convert('L')
        if self.transform:
            img1 = self.transform(img1)  # 變成 [1,256,256]
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ----------- Training & Validation Loop ----------- #
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 定義 transform（專為 PIL 用）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果要 normalize：（灰階）
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. 建立訓練資料 (SegmentationDataset) 及其對應的 SiameseDataset
    train_base = SegmentationDataset(
        list_file='train_fixed.txt',
        root_dir='train_dataset',
        mask_dir='masks',
        transform=None  # SegmentationDataset 已經 resize & 給出 Tensor
    )
    train_dataset = SiameseDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # 3. 建立驗證資料 (SiamesePairsDataset) 需要先產出 val_pairs.txt
    val_dataset = SiamesePairsDataset(
        pair_txt='val_pairs.txt',
        transform=transform,
        root_dir='train_dataset'  # 讀取時會自動拼成 "train_dataset/CASIA-..."
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 4. 初始化模型、Loss、優化器
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 30
    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        total_loss = 0.0
        for img1, img2, label in train_loader:
            # train_base 已回傳 [1,256,256] 的 Tensor，因此不用再轉 PIL
            img1 = img1.to(device)  # [B,1,256,256]
            img2 = img2.to(device)
            label = label.to(device)  # [B]

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # ---------- Validation ----------
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1 = img1.to(device)  # 這裡 img1, img2 已經是 [B,1,256,256]
                img2 = img2.to(device)
                label = label.to(device)

                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                total_val_loss += loss.item()

                # 用歐式距離加閾值判斷正/負
                dist = torch.nn.functional.pairwise_distance(out1, out2)
                preds = (dist < 0.5).float()  # 閾值可在驗證結果後再微調
                correct += (preds == label).sum().item()
                total += label.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 5. 儲存模型
    torch.save(model.state_dict(), 'siamese_model_2.pth')
    print("Model saved to siamese_model_2.pth")


if __name__ == '__main__':
    main()
