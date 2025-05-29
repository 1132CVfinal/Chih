import torch
from torch.utils.data import Dataset
import cv2
import os

class IrisDataset(Dataset):
    def __init__(self, list_file, root_dir='.', transform=None):
        """
        list_file: 包含影像路徑的文字檔 ( 每行一張圖片路徑，相對於 root_dir)
        root_dir: 影像路徑的根目錄
        transform: 影像轉換（可選）
        """
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        self.image_paths = [line.strip() for line in lines]
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at {img_path}")
        if self.transform:
            image = self.transform(image)
        return image, img_path

class SegmentationDataset(Dataset):
    def __init__(self, list_file, root_dir='train_dataset', mask_dir='masks', transform=None):
        with open(list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        mask_path = os.path.join(self.mask_dir, rel_path.replace('.jpg', '_mask.png'))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = image / 255.0  # normalize to 0-1
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # [1, H, W]

        return image, mask
