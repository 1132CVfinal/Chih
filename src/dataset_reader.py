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
    

from ritnet_inference import RITNetInference

class PolarDataset(Dataset):
    """
    讀一個 list.txt (每行一張灰階 iris 圖的相對路徑)，
    利用 RITNetInference segment+normalize → 回傳一個 [1, 256, 256] 的 Tensor polar 圖
    """
    def __init__(self, list_file, root_dir='.', ritnet_model_path='../RITnet/best_model.pkl'):
        """
        list_file: 每行一張圖片相對路徑 (例如 "train_dataset/CASIA-Iris-Lamp/102/L/S2102L08.jpg")
        root_dir: 這些相對路徑對應的根目錄 (通常是專案根目錄)
        ritnet_model_path: RITNet 的權重檔 (.pkl)
        """
        # 1. 把 list.txt 裡的每行路徑讀進來
        with open(list_file, 'r') as f:
            lines = f.readlines()
        # 去掉空白、換行
        self.image_paths = [line.strip() for line in lines if line.strip()]
        self.root_dir = root_dir

        # 2. 載入 RITNetInference (做 segmentation + normalize)
        self.ritnet = RITNetInference(model_path=ritnet_model_path,
                                      device='cuda' if torch.cuda.is_available() else 'cpu')

        # 3. 你可以預先決定最終要把 normalize 出來的 polar 圖 resize 到多大
        #    這裡示範：一律 resize 到 256×256
        self.desired_size = (256, 256)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 讀原始灰階影像
        rel_path = self.image_paths[idx]  # e.g. "train_dataset/CASIA-Iris-Lamp/102/L/S2102L08.jpg"
        abs_path = os.path.join(self.root_dir, rel_path)
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {abs_path}")

        # 2. 呼叫 RITNetInference.segment_iris 得到 iris_mask, pupil_mask
        iris_mask, pupil_mask = self.ritnet.segment_iris(img)
        #    如果 segmentation 失敗 (mask=None)，RITNetInference 內部會 fallback 用 HoughCircles

        # 3. 找邊界、normalize → 得到 2D numpy polar 圖 (radial_res×angular_res = 32×64)
        boundaries = self.ritnet.find_iris_boundaries(iris_mask, pupil_mask)
        normalized = self.ritnet.normalize_iris(img, *boundaries)
        #    如果 boundary detection 失敗，就得到 None → 在這種情況我們就直接把整張灰階圖填滿
        if normalized is None:
            # fallback：把原圖 resize 到 polar 大小 (32×64)，再 resize 回 256×256
            fallback = cv2.resize(img, (64, 32))
            normalized = fallback.astype('uint8')

        # 4. scaled down/up：先把 normalized (32×64) 放大到 256×256 (確保 CNN 有足夠的空間)
        normalized_resized = cv2.resize(normalized, self.desired_size, interpolation=cv2.INTER_LINEAR)

        # 5. 轉成 Tensor 並做 normalize (0~1)
        tensor = torch.tensor(normalized_resized / 255.0, dtype=torch.float32).unsqueeze(0)
        # 最後形狀是 [1, 256, 256]

        # 6. 回傳 (tensor, rel_path) 或只回 tensor
        return tensor, rel_path