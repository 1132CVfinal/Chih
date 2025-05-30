import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import os
from PIL import Image  # SiamesePairsDataset 需要使用 PIL 來開 PNG/JPG

from dataset_reader import SegmentationDataset  # 你前面已經寫好的

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
            nn.Linear(64 * 64 * 64, 256),  # 對應輸入 256×256 經過兩次 MaxPool → 64×64
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


# ----------- SiameseDataset：使用 SegmentationDataset 建立 pair ----------- #
class SiameseDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: 一個 SegmentationDataset，回傳 (image, mask) 或 (image, _) 
                      這裡只需要 image，因此取 0。
        """
        self.base_dataset = base_dataset
        # 解析 person_id；假設路徑格式為 'CASIA-Iris-Thousand/447/L/S5447L00.jpg'
        self.labels = [path.split('/')[1] for path in base_dataset.image_paths]

        # 將相同 person_id 的索引集合起來
        from collections import defaultdict
        label_dict = defaultdict(list)
        for i, person_id in enumerate(self.labels):
            label_dict[person_id].append(i)
        self.label_dict = label_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 取出第一張影像
        img1, _ = self.base_dataset[idx]
        person1 = self.labels[idx]

        # 隨機決定要同人 (positive) 還是不同人 (negative)
        if random.random() < 0.5:
            # positive：同一人的另一張
            idx2 = idx
            while idx2 == idx:
                idx2 = random.choice(self.label_dict[person1])
            label = 1.0
        else:
            # negative：不同人
            person2 = random.choice(list(self.label_dict.keys()))
            while person2 == person1:
                person2 = random.choice(list(self.label_dict.keys()))
            idx2 = random.choice(self.label_dict[person2])
            label = 0.0

        img2, _ = self.base_dataset[idx2]
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ----------- SiamesePairsDataset：讀取已預先生成的 pair (val) ----------- #
class SiamesePairsDataset(Dataset):
    def __init__(self, pair_txt, transform=None, root_dir=''):
        """
        pair_txt: 每行的格式是 "path/to/img1 path/to/img2 label"
        transform: torchvision.transforms, 對 PIL Image 做處理
        root_dir: 如果 pair_txt 中的路徑沒有指定完整路徑，可以加上此根目錄
        """
        self.pairs = []
        self.transform = transform
        self.root_dir = root_dir

        with open(pair_txt, 'r') as f:
            for line in f:
                p1, p2, label = line.strip().split()
                # 如果路徑不是絕對路徑，就加上 root_dir
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
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ----------- Training Loop ----------- #
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ———————————— 1. 定義 transform ————————————
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 如果需要 normalization，可再加上下面這行（灰階）
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ———————————— 2. 建立 Dataset & DataLoader ————————————
    # 訓練時：用 SegmentationDataset 載原圖 (mask 可忽略)，並生成 SiameseDataset
    train_base = SegmentationDataset('train_fixed.txt', root_dir='train_dataset', mask_dir='masks', transform=None)
    train_dataset = SiameseDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # 驗證時：讀取已事先生成的 val_pairs.txt
    val_dataset = SiamesePairsDataset('val_pairs.txt', transform=transform, root_dir='train_dataset')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # ———————————— 3. 初始化模型、損失函數、優化器 ————————————
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ———————————— 4. 開始訓練＆驗證 ————————————
    num_epochs = 30
    for epoch in range(num_epochs):
        # ---- 4.1 Train ----
        model.train()
        total_loss = 0.0
        for img1, img2, label in train_loader:
            # SegmentationDataset 回傳的是 Tensor 但没有經過 resize/ToTensor，
            # 要先把 image 轉為 FloatTensor[1×256×256]，以下示範：
            img1 = img1.to(device)
            img2 = img2.to(device)
            # 下面這行若是 NumPy array，要先呼叫 transforms：
            img1 = transform(Image.fromarray((img1.squeeze(0).cpu().numpy()*255).astype('uint8')))
            img2 = transform(Image.fromarray((img2.squeeze(0).cpu().numpy()*255).astype('uint8')))
            img1 = img1.unsqueeze(0).to(device)  # final shape: [1,1,256,256]
            img2 = img2.unsqueeze(0).to(device)
            label = label.to(device)

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # ---- 4.2 Validation ----
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                total_val_loss += loss.item()

                # 用距離 + 閾值 估算正/負
                dist = torch.nn.functional.pairwise_distance(out1, out2)
                preds = (dist < 0.5).float()  # 閾值可依驗證結果再調整
                correct += (preds == label).sum().item()
                total += label.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ———————————— 5. 儲存模型 ————————————
    torch.save(model.state_dict(), 'siamese_model_2.pth')
    print("Model saved to siamese_model_2.pth")


if __name__ == '__main__':
    main()
