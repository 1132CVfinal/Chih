#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
from pathlib import Path
import math
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Import RITNet
from ritnet_inference import RITNetInference

class ImprovedIrisRecognition:
    def __init__(self, ritnet_model_path='best_model.pkl', debug=False):
        self.debug = debug
        
        try:
            self.segmentation_model = RITNetInference(ritnet_model_path)
            if self.debug:
                print(" RITNet model loaded successfully")
        except Exception as e:
            print(f" RITNet model loading failed: {e}")
            self.segmentation_model = None
        
        self.gabor_params = []
        for theta in np.arange(0, np.pi, np.pi/6): 
            for freq in [0.05, 0.15, 0.25]:  
                self.gabor_params.append({'theta': theta, 'freq': freq})
    
    def simple_iris_detection(self, image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) >= 2:
                circles = sorted(circles, key=lambda x: x[2])
                pupil = circles[0]
                iris = circles[1] if len(circles) > 1 else circles[0]
                
                pupil_center = (pupil[0], pupil[1])
                pupil_radius = pupil[2]
                iris_center = (iris[0], iris[1])
                iris_radius = iris[2]
                
                return pupil_center, pupil_radius, iris_center, iris_radius
        
        return None, None, None, None
    
    def segment_iris(self, image):
        if self.segmentation_model is not None:
            try:
                iris_mask, pupil_mask, _, _ = self.segmentation_model.segment_iris(image)
                
                iris_pixels = np.sum(iris_mask)
                pupil_pixels = np.sum(pupil_mask)
                
                if iris_pixels > 100 and pupil_pixels > 10:  
                    return iris_mask, pupil_mask
                else:
                    if self.debug:
                        print("⚠️ RITNet segmentation quality poor, using backup method")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ RITNet segmentation failed: {e}, using backup method")
        
        pupil_center, pupil_radius, iris_center, iris_radius = self.simple_iris_detection(image)
        if pupil_center is not None:
            h, w = image.shape
            iris_mask = np.zeros((h, w), dtype=np.uint8)
            pupil_mask = np.zeros((h, w), dtype=np.uint8)
            
            cv2.circle(iris_mask, iris_center, int(iris_radius), 1, -1)
            cv2.circle(pupil_mask, pupil_center, int(pupil_radius), 1, -1)
            
            return iris_mask, pupil_mask
        
        return None, None
    
    def find_iris_boundaries(self, iris_mask, pupil_mask):
        if iris_mask is None or pupil_mask is None:
            return None, None, None, None
        
        pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iris_contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not pupil_contours or not iris_contours:
            return None, None, None, None
        
        pupil_contour = max(pupil_contours, key=cv2.contourArea)
        iris_contour = max(iris_contours, key=cv2.contourArea)
        
        try:
            pupil_moments = cv2.moments(pupil_contour)
            iris_moments = cv2.moments(iris_contour)
            
            if pupil_moments['m00'] != 0 and iris_moments['m00'] != 0:
                pupil_center = (int(pupil_moments['m10']/pupil_moments['m00']), 
                              int(pupil_moments['m01']/pupil_moments['m00']))
                iris_center = (int(iris_moments['m10']/iris_moments['m00']), 
                             int(iris_moments['m01']/iris_moments['m00']))
                
                pupil_area = cv2.contourArea(pupil_contour)
                iris_area = cv2.contourArea(iris_contour)
                
                pupil_radius = np.sqrt(pupil_area / np.pi)
                iris_radius = np.sqrt(iris_area / np.pi)
                
                return pupil_center, pupil_radius, iris_center, iris_radius
            
        except Exception as e:
            if self.debug:
                print(f"Boundary detection error: {e}")
        
        return None, None, None, None
    
    def normalize_iris(self, image, pupil_center, pupil_radius, iris_center, iris_radius, 
                      angular_res=64, radial_res=32):
        if pupil_center is None or iris_center is None:
            return None
        
        if pupil_radius <= 0 or iris_radius <= pupil_radius:
            return None
            
        normalized = np.zeros((radial_res, angular_res), dtype=np.uint8)
        
        center_x, center_y = pupil_center
        
        for i in range(angular_res):
            for j in range(radial_res):
                theta = 2 * np.pi * i / angular_res
                # Non-linear radial mapping, denser sampling in inner iris
                r_normalized = j / radial_res
                r = pupil_radius + (r_normalized ** 0.5) * (iris_radius - pupil_radius)
                
                x = int(center_x + r * np.cos(theta))
                y = int(center_y + r * np.sin(theta))
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized[j, i] = image[y, x]
        
        return normalized
    
    def extract_multiple_features(self, normalized_iris):
        if normalized_iris is None:
            return None
        
        features = []
        
        gabor_features = self.extract_gabor_features(normalized_iris)
        if gabor_features is not None:
            features.extend(gabor_features)
        
        lbp_features = self.extract_lbp_features(normalized_iris)
        if lbp_features is not None:
            features.extend(lbp_features)
        
        stat_features = self.extract_statistical_features(normalized_iris)
        if stat_features is not None:
            features.extend(stat_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_gabor_features(self, normalized_iris):
        features = []
        
        for params in self.gabor_params:
            ksize = 15
            sigma = 3
            theta = params['theta']
            lambda_ = 1.0 / params['freq']
            gamma = 0.5
            psi = 0
            
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(normalized_iris, cv2.CV_32F, kernel)
            
            threshold = np.mean(filtered)
            binary = (filtered > threshold).astype(np.uint8)
            
            regions = []
            h, w = binary.shape
            for i in range(0, h, 4):
                for j in range(0, w, 8):
                    region = binary[i:i+4, j:j+8]
                    density = np.mean(region)
                    regions.append(density)
            
            features.extend(regions)
        
        return features
    
    def extract_lbp_features(self, normalized_iris, radius=2, n_points=8):
        def lbp(image, radius, n_points):
            h, w = image.shape
            lbp_image = np.zeros((h, w), dtype=np.uint8)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    binary_string = ""
                    
                    for p in range(n_points):
                        angle = 2 * np.pi * p / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if 0 <= x < h and 0 <= y < w:
                            binary_string += '1' if image[x, y] >= center else '0'
                        else:
                            binary_string += '0'
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            return lbp_image
        
        lbp_img = lbp(normalized_iris, radius, n_points)
        
        hist, _ = np.histogram(lbp_img, bins=2**n_points, range=(0, 2**n_points))
        
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-8)
        
        return hist.tolist()
    
    def extract_statistical_features(self, normalized_iris):
        features = []
        
        features.append(np.mean(normalized_iris))
        features.append(np.std(normalized_iris))
        features.append(np.median(normalized_iris))
        
        h, w = normalized_iris.shape
        for i in range(0, h, h//4):
            for j in range(0, w, w//8):
                region = normalized_iris[i:i+h//4, j:j+w//8]
                if region.size > 0:
                    features.append(np.mean(region))
                    features.append(np.std(region))
        
        return features
    
    def calculate_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.5
        
        if len(features1) != len(features2):
            return 0.5
        
        features1 = np.array(features1)
        features2 = np.array(features2)
        
        if np.std(features1) < 1e-8 or np.std(features2) < 1e-8:
            return 0.5
        
        try:
            euclidean_dist = euclidean(features1, features2)
            max_possible_dist = np.sqrt(2 * len(features1))  # Assume features in [0,1] range
            euclidean_score = euclidean_dist / max_possible_dist
            
            cosine_sim = cosine_similarity([features1], [features2])[0, 0]
            cosine_score = (1 - cosine_sim) / 2  # Convert to distance, range [0,1]
            
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            correlation_score = (1 - correlation) / 2
            
            final_score = 0.4 * euclidean_score + 0.4 * cosine_score + 0.2 * correlation_score

            final_score = np.clip(final_score, 0, 1)
            
            if self.debug and np.random.random() < 0.1:  # Randomly print some debug info
                print(f"  Euclidean: {euclidean_score:.3f}, Cosine: {cosine_score:.3f}, Correlation: {correlation_score:.3f}, Final: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            if self.debug:
                print(f"Similarity calculation error: {e}")
            return 0.5
    
    def compare_iris(self, img1, img2):
        """
        Compare two iris images
        """
        try:
            if self.debug:
                print(f"\nStarting iris image comparison...")
            
            iris_mask1, pupil_mask1 = self.segment_iris(img1)
            iris_mask2, pupil_mask2 = self.segment_iris(img2)
            
            if iris_mask1 is None or iris_mask2 is None:
                if self.debug:
                    print("   Segmentation failed")
                return 0.5

            boundaries1 = self.find_iris_boundaries(iris_mask1, pupil_mask1)
            boundaries2 = self.find_iris_boundaries(iris_mask2, pupil_mask2)
            
            if boundaries1[0] is None or boundaries2[0] is None:
                if self.debug:
                    print("   Boundary detection failed")
                return 0.5
            
            normalized_iris1 = self.normalize_iris(img1, *boundaries1)
            normalized_iris2 = self.normalize_iris(img2, *boundaries2)
            
            if normalized_iris1 is None or normalized_iris2 is None:
                if self.debug:
                    print("   Normalization failed")
                return 0.5
            
            features1 = self.extract_multiple_features(normalized_iris1)
            features2 = self.extract_multiple_features(normalized_iris2)
            
            if features1 is None or features2 is None:
                if self.debug:
                    print("   Feature extraction failed")
                return 0.5

            score = self.calculate_similarity(features1, features2)
            
            if self.debug:
                print(f"   Comparison completed, score: {score:.4f}")
            
            return score
            
        except Exception as e:
            if self.debug:
                print(f"   Comparison process error: {e}")
            return 0.5

if __name__ == "__main__":
    iris_system = ImprovedIrisRecognition('best_model.pkl', debug=True)
    
    # Test
    img1 = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('test2.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is not None and img2 is not None:
        score = iris_system.compare_iris(img1, img2)
        print(f"Final similarity score: {score:.4f}")