import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class IrisDataset(Dataset):
    def __init__(self, list_file, root_dir='train_dataset', transform=None):
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

import torch
from torch.utils.data import Dataset
import cv2
import os

class SegmentationDataset(Dataset):
    def __init__(self, list_file, root_dir='train_dataset', mask_dir='masks', transform=None):
        """
        list_file: 每行是已去掉 'train_dataset/' 的相對路徑，例如：
            CASIA-Iris-Thousand/447/L/S5447L00.jpg
            CASIA-Iris-Lamp/102/L/S2102L08.jpg
        root_dir: 放原圖的資料夾（預設 'train_dataset'），相對於專案根目錄
        mask_dir: 放 masks 的資料夾（預設 'masks'），相對於專案根目錄
        transform: 可選，如果你想對 image Tensor 做額外轉換
        """
        with open(list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        # 完整圖片路徑：src/train_dataset/CASIA-Iris-Thousand/447/L/S5447L00.jpg
        img_path = os.path.join(self.root_dir, rel_path)

        # 遮罩檔名：同樣資料夾結構，但檔名後綴 "_mask.png"
        if rel_path.endswith('.jpg'):
            mask_rel = rel_path.replace('.jpg', '_mask.png')
        elif rel_path.endswith('.png'):
            mask_rel = rel_path.replace('.png', '_mask.png')
        else:
            raise ValueError(f"Unsupported file extension in path: {rel_path}")
        mask_path = os.path.join(self.mask_dir, mask_rel)

        # 讀 image 與 mask（灰階）
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found:  {mask_path}")

        # 將都 resize 成 256×256
        desired_size = (256, 256)
        image = cv2.resize(image, desired_size)
        mask  = cv2.resize(mask, desired_size)

        # normalize 到 [0,1]
        image = image / 255.0
        mask  = mask / 255.0

        # 轉成 Tensor，shape = [1, 256, 256]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask  = torch.tensor(mask,  dtype=torch.float32).unsqueeze(0)

        # 如果要對 image 做額外 transform，就在這裡呼叫
        if self.transform:
            image = self.transform(image)

        # 回傳 image 與 mask；有需要也可以只回 image
        return image, mask


import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from ritnet_inference import RITNetInference
from improved_iris_recognition import ImprovedIrisRecognition

class PolarDataset(Dataset):
    """
    將 list.txt（每行一張灰階 iris 圖相對路徑）
    經過 RITNet 做 segmentation → ImprovedIrisRecognition 做 normalize (polar)
    最終輸出一張 [1,256,256] 的 Tensor（已經做完 polar 展開並 resize）
    """
    def __init__(self, list_file, root_dir='.', ritnet_model_path='../RITnet/best_model.pkl'):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.image_paths = [line.strip() for line in lines if line.strip()]
        self.root_dir = root_dir

        # 用 RITNet 做 segmentation
        self.ritnet = RITNetInference(
            model_path=ritnet_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 用 ImprovedIrisRecognition 處理邊界與 normalize
        self.improved = ImprovedIrisRecognition()

        self.desired_size = (256, 256)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        abs_path = os.path.join(self.root_dir, rel_path)
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {abs_path}")

        # segmentation → 取得 iris_mask, pupil_mask
        iris_mask, pupil_mask, _, _ = self.ritnet.segment_iris(img)

        # boundary + normalization
        print("find_iris_boundaries")
        boundaries = self.improved.find_iris_boundaries(iris_mask, pupil_mask)
        iris_bbox, pupil_bbox = boundaries
        normalized = None
        if iris_bbox is not None and pupil_bbox is not None:
            print("normalize_iris")
            normalized = self.improved.normalize_iris(img, iris_bbox, pupil_bbox)

        if normalized is None:
            print("fallback")
            fallback = cv2.resize(img, (64, 32), interpolation=cv2.INTER_LINEAR)
            normalized = fallback.astype(np.uint8)

        normalized_resized = cv2.resize(normalized, self.desired_size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.tensor(normalized_resized / 255.0, dtype=torch.float32).unsqueeze(0)

        return tensor, rel_path
