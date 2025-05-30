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
        list_file: 已去掉 “train_dataset/” prefix 的影像相對路徑清單
        root_dir: 放原始影像的資料夾（一般設 'train_dataset'）
        mask_dir: 放已存在 mask 的資料夾（一般設 'masks'）
        transform: 對原始影像 (PIL or NumPy) 的變換
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
        img_path = os.path.join(self.root_dir, rel_path)

        # 根據副檔名決定 mask 的命名規則
        if rel_path.endswith('.jpg'):
            mask_path = os.path.join(self.mask_dir, rel_path.replace('.jpg', '_mask.png'))
        elif rel_path.endswith('.png'):
            mask_path = os.path.join(self.mask_dir, rel_path.replace('.png', '_mask.png'))
        else:
            raise ValueError(f"Unsupported file extension in path: {rel_path}")

        # 讀取影像與 mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # resize 至 (256, 256)
        desired_size = (256, 256)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        image = cv2.resize(image, desired_size)
        mask = cv2.resize(mask, desired_size)

        # Normalize 至 [0,1]
        image = image / 255.0
        mask = mask / 255.0

        # 轉成 Tensor，[1, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # （可選）對影像做進一步 transform，例如 data augmentation
        if self.transform:
            # 注意：transform 若使用 torchvision.transforms，通常要先把 NumPy array 轉成 PIL Image
            # 例如：
            #   pil_img = Image.fromarray((image.squeeze(0).numpy() * 255).astype(np.uint8))
            #   image = self.transform(pil_img)
            pass

        return image, mask
