import numpy as np
import cv2
import torch
from torchvision import transforms
import argparse
from pathlib import Path

# 假設已經有了RITNet模型
from ritnet_model import RITNet  # 需要導入適當的RITNet實現

class IrisRecognition:
    def __init__(self):
        # 加載RITNet模型
        self.segmentation_model = self._load_segmentation_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),  # RITNet期望的輸入大小
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def _load_segmentation_model(self):
        # 加載預訓練的RITNet模型
        model = RITNet()
        model.load_state_dict(torch.load('ritnet_model.pth'))
        model.eval()
        return model
    
    def segment_iris(self, image):
        # 使用RITNet進行分割
        tensor_image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            segmentation = self.segmentation_model(tensor_image)
        
        # 將分割結果轉換為掩碼
        mask = torch.argmax(segmentation, dim=1).squeeze().numpy()
        # 假設類別1是虹膜
        iris_mask = (mask == 1).astype(np.uint8)
        # 假設類別0是瞳孔
        pupil_mask = (mask == 0).astype(np.uint8)
        
        return iris_mask, pupil_mask
    
    def normalize_iris(self, image, iris_mask, pupil_mask):
        # 找到瞳孔和虹膜的輪廓
        pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iris_contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not pupil_contours or not iris_contours:
            return None  # 無法找到輪廓
        
        # 找到最大的輪廓（應該是瞳孔和虹膜）
        pupil_contour = max(pupil_contours, key=cv2.contourArea)
        iris_contour = max(iris_contours, key=cv2.contourArea)
        
        # 擬合橢圓
        try:
            (pupil_x, pupil_y), (pupil_major, pupil_minor), pupil_angle = cv2.fitEllipse(pupil_contour)
            (iris_x, iris_y), (iris_major, iris_minor), iris_angle = cv2.fitEllipse(iris_contour)
        except:
            return None  # 無法擬合橢圓
        
        # 計算瞳孔和虹膜的半徑
        pupil_radius = (pupil_major + pupil_minor) / 4
        iris_radius = (iris_major + iris_minor) / 4
        
        # 使用Daugman的橡皮片模型進行標準化
        normalized_iris = self._rubber_sheet_model(image, (pupil_x, pupil_y), pupil_radius, 
                                                 (iris_x, iris_y), iris_radius)
        
        return normalized_iris
    
    def _rubber_sheet_model(self, image, pupil_center, pupil_radius, iris_center, iris_radius, 
                           angular_res=64, radial_res=64):
        normalized = np.zeros((radial_res, angular_res), dtype=np.uint8)
        
        # 極坐標轉換
        for i in range(angular_res):
            for j in range(radial_res):
                theta = 2 * np.pi * i / angular_res
                r = pupil_radius + (j / radial_res) * (iris_radius - pupil_radius)
                
                x = int(pupil_center[0] + r * np.cos(theta))
                y = int(pupil_center[1] + r * np.sin(theta))
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized[j, i] = image[y, x]
        
        return normalized
    
    def extract_features(self, normalized_iris):
        # 使用Gabor濾波器提取特徵
        # 這裡我們使用一個簡化版本的Gabor特徵提取
        ksize = 11
        sigma = 3
        theta = 0
        lambda_ = 10
        gamma = 0.5
        psi = 0
        
        filters = []
        for theta in np.arange(0, np.pi, np.pi/8):  # 8個不同的方向
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
            filters.append(kern)
        
        features = []
        for kern in filters:
            filtered = cv2.filter2D(normalized_iris, cv2.CV_8UC1, kern)
            # 二值化
            _, binary = cv2.threshold(filtered, 0, 1, cv2.THRESH_BINARY)
            features.extend(binary.flatten())
        
        return np.array(features)
    
    def compare_iris(self, img1, img2):
        # 分割
        iris_mask1, pupil_mask1 = self.segment_iris(img1)
        iris_mask2, pupil_mask2 = self.segment_iris(img2)
        
        # 標準化
        normalized_iris1 = self.normalize_iris(img1, iris_mask1, pupil_mask1)
        normalized_iris2 = self.normalize_iris(img2, iris_mask2, pupil_mask2)
        
        if normalized_iris1 is None or normalized_iris2 is None:
            return 0.5  # 無法比較時返回中間值
        
        # 特徵提取
        features1 = self.extract_features(normalized_iris1)
        features2 = self.extract_features(normalized_iris2)
        
        # 計算漢明距離
        distance = np.sum(features1 != features2) / len(features1)
        
        return distance  # 已經在0-1範圍內，0表示相同，1表示不同