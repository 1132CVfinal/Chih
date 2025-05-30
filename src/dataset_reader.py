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
        
        # 注意：這裡的 self.image_paths 應該是已去除 prefix “train_dataset/” 的
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
            # transform 應該能接受 NumPy array（灰階）或 PIL Image
            image = self.transform(image)
        return image, img_path

class SegmentationDataset(Dataset):
    def __init__(self, list_file, root_dir='train_dataset', mask_dir='masks', transform=None):
        """
        list_file: 每行是已去掉 'train_dataset/' 的相對路徑，例如：
            CASIA-Iris-Thousand/447/L/S5447L00.jpg
            CASIA-Iris-Lamp/102/L/S2102L08.jpg
        root_dir: 放原圖的資料夾（預設 'train_dataset'）
        mask_dir: 放 masks 的資料夾（預設 'masks'）
        transform: 可選，如果你想再對 image tensor 做其他操作
        """
        with open(list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        # 組成完整路徑：train_dataset/CASIA-Iris-Thousand/447/L/S5447L00.jpg
        img_path = os.path.join(self.root_dir, rel_path)

        # 決定要讀哪種副檔名的 mask
        if rel_path.endswith('.jpg'):
            mask_path = os.path.join(self.mask_dir, rel_path.replace('.jpg', '_mask.png'))
        elif rel_path.endswith('.png'):
            mask_path = os.path.join(self.mask_dir, rel_path.replace('.png', '_mask.png'))
        else:
            raise ValueError(f"Unsupported file extension in path: {rel_path}")

        # 讀影像與 mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Resize 到 256x256
        desired_size = (256, 256)
        image = cv2.resize(image, desired_size)
        mask = cv2.resize(mask, desired_size)

        # normalize 到 [0,1]
        image = image / 255.0
        mask = mask / 255.0

        # 轉成 Tensor, shape [1, 256, 256]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 如果想對 image 做額外 transform，可以在這裡呼叫
        if self.transform:
            image = self.transform(image)

        # 只回傳 image, mask；如果你不需要 mask，也可以只用 image
        return image, mask