#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
import torch.nn.functional as F
from densenet import DenseNet2D
import math
import numpy as np

class RITNetInference:
    def __init__(self, model_path='best_model.pkl', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = DenseNet2D(in_channels=1, out_channels=4, channel_size=32, dropout=True, prob=0.2)
        
        # Load pretrained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def segment_iris(self, image):
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Get prediction results
        predictions = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize back to original image size
        h, w = image.shape[:2]
        predictions_resized = cv2.resize(predictions.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        background_mask = (predictions_resized == 0).astype(np.uint8)
        sclera_mask = (predictions_resized == 1).astype(np.uint8)
        iris_mask = (predictions_resized == 2).astype(np.uint8)
        pupil_mask = (predictions_resized == 3).astype(np.uint8)
        
        return iris_mask, pupil_mask, sclera_mask, background_mask
    
    def visualize_segmentation(self, image, save_path=None):
        iris_mask, pupil_mask, sclera_mask, background_mask = self.segment_iris(image)
        
        h, w = image.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        vis[background_mask == 1] = [0, 0, 0]      # Black: background
        vis[sclera_mask == 1] = [255, 0, 0]        # Red: sclera
        vis[iris_mask == 1] = [0, 255, 0]          # Green: iris
        vis[pupil_mask == 1] = [0, 0, 255]         # Blue: pupil
        
        if save_path:
            cv2.imwrite(save_path, vis)
        
        return vis

    # ADDED
    def crop_iris(self, image):
        """
        對傳入的 gray-scale image (np.ndarray) 先做 segmentation，
        找 iris_mask 上非零的 bounding‐box，再把原始 image 裁切，回傳裁切後只剩 iris 的影像。
        如果沒偵測到 iris，就直接回傳整張原圖。
        """
        iris_mask, pupil_mask, sclera_mask, background_mask = self.segment_iris(image)
        # 只關心 iris_mask
        ys, xs = np.where(iris_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            # 如果根本沒有偵測到 iris，就回傳原始 gray image
            return image
        # bounding box：xmin, xmax, ymin, ymax
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        # 注意：numpy slicing 是 [ymin : ymax+1, xmin : xmax+1]
        cropped = image[ymin : ymax+1, xmin : xmax+1]
        return cropped

    def find_iris_boundaries(self, iris_mask, pupil_mask):
        """
        從 iris_mask 與 pupil_mask 中找出邊界。
        回傳 (iris_bbox, pupil_bbox)，每個 bbox 是 (xmin, ymin, xmax, ymax)。
        如果某個 mask 沒有找到任何像素，對應的 bbox 就是 None。
        """
        def get_bbox(mask):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return None
            return (xs.min(), ys.min(), xs.max(), ys.max())
        
        iris_bbox = get_bbox(iris_mask)
        pupil_bbox = get_bbox(pupil_mask)

        return iris_bbox, pupil_bbox
        

    def normalize_iris(self, image, iris_bbox, pupil_bbox, output_height=32, output_width=64):
        """
        簡單實作：根據 iris 與 pupil 的 bounding box 做 polar 展開。
        回傳一張 shape = (32, 64) 的灰階圖像（np.ndarray）。
        """

        def get_center_radius(bbox):
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            r = max((x_max - x_min), (y_max - y_min)) // 2
            return cx, cy, r

        # 若 bbox 其中一個是 None，則回傳 None
        if iris_bbox is None or pupil_bbox is None:
            return None

        iris_cx, iris_cy, iris_r = get_center_radius(iris_bbox)
        pupil_cx, pupil_cy, pupil_r = get_center_radius(pupil_bbox)

        # output image
        polar_img = np.zeros((output_height, output_width), dtype=np.uint8)

        for theta_idx in range(output_width):
            theta = 2 * math.pi * theta_idx / output_width
            for r_idx in range(output_height):
                r_frac = r_idx / output_height
                # interpolate between pupil and iris circle
                x = int((1 - r_frac) * pupil_cx + r_frac * iris_cx + 
                        ((1 - r_frac) * pupil_r + r_frac * iris_r) * math.cos(theta))
                y = int((1 - r_frac) * pupil_cy + r_frac * iris_cy + 
                        ((1 - r_frac) * pupil_r + r_frac * iris_r) * math.sin(theta))

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    polar_img[r_idx, theta_idx] = image[y, x]

        return polar_img




if __name__ == "__main__":
    ritnet = RITNetInference('best_model.pkl')
    
    # Read test image
    test_image = cv2.imread('test_eye_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Perform segmentation
    iris_mask, pupil_mask, sclera_mask, background_mask = ritnet.segment_iris(test_image)
    
    # Visualize results
    vis_result = ritnet.visualize_segmentation(test_image, 'segmentation_result.jpg')
    
    print(f"Segmentation completed!")
    print(f"Iris pixel count: {np.sum(iris_mask)}")
    print(f"Pupil pixel count: {np.sum(pupil_mask)}")


