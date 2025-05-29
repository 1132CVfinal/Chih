#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
import torch.nn.functional as F
from densenet import DenseNet2D

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