# validate_recognition.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset_reader import PolarDataset
from train_recognition import SiameseNetwork   # 注意：這裡載入的是 SiameseNetwork 而非 SiameseNet

def build_tensor_dict(polar_dataset, encoder_model, device):
    """
    將整個 polar_dataset 的所有影像餵入 encoder，記住每張影像對應的 embedding（128 維向量）。
    回傳一個 dict: {rel_path: embedding_numpy_array}
    """
    encoder_model.eval()
    tensor_to_feat = {}
    loader = DataLoader(polar_dataset, batch_size=16, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            imgs, rel_paths = batch       # imgs: [B,1,256,256]; rel_paths: list[str]
            imgs = imgs.to(device)
            feats = encoder_model.encoder(imgs)  # 取出整個 SiameseNetwork 的 encoder 層
            feats = feats.cpu().numpy()          # shape = (B, 128)
            for i, rp in enumerate(rel_paths):
                # copy 一份到 dict
                tensor_to_feat[rp] = feats[i].copy()
    return tensor_to_feat

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 1. 載入整個 SiameseNetwork（從 train_recognition.py）並把其參數 load 進來
    full_model = SiameseNetwork().to(device)
    full_model.load_state_dict(torch.load("siamese_final.pth", map_location=device))
    full_model.eval()

    # 2. 建立 validation 用的 PolarDataset（只需要 val_fixed.txt 即可）
    #    注意：ritnet_model_path 以 src/ 為工作目錄，要指到專案根目錄裡的 ../RITnet/best_model.pkl
    val_list = "val_fixed.txt"
    polar_val = PolarDataset(
        list_file=val_list,
        root_dir="train_dataset", 
        ritnet_model_path="../RITnet/best_model.pkl"
    )

    # 3. 先把所有 val 圖都跑一次 encoder，記錄其 embedding
    print("Computing embeddings for validation set...")
    feat_dict = build_tensor_dict(polar_val, full_model, device)
    print(f"Total validation embeddings: {len(feat_dict)}")

    # 4. 讀 val_pairs.txt（每行格式: rel_path1<TAB或空格>rel_path2<TAB或空格>label）
    pair_list_path = "val_pairs.txt"
    pairs = []
    with open(pair_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1, p2, lab = parts
            lab = int(lab)
            # 只保留都能在 feat_dict 找到的
            if (p1 in feat_dict) and (p2 in feat_dict):
                pairs.append((p1, p2, lab))

    if len(pairs) == 0:
        print("No valid pairs found in val_pairs.txt.")
        return

    # 5. 計算每對 (p1,p2) 的距離 dist 與 label
    dists = []
    labels = []
    for p1, p2, lab in pairs:
        f1 = feat_dict[p1]  # numpy array (128,)
        f2 = feat_dict[p2]
        # Euclidean distance
        dist = np.linalg.norm(f1 - f2)
        dists.append(dist)
        labels.append(lab)

    dists  = np.array(dists)
    labels = np.array(labels)

    # 6. 計算 AUC、找最佳 threshold、計算 accuracy
    from sklearn.metrics import roc_curve, roc_auc_score
    # 用負距離表示相似度：score 越大表示越同人
    auc = roc_auc_score(labels, -dists)
    fpr, tpr, thresholds = roc_curve(labels, -dists)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]

    preds = (dists < best_thresh).astype(int)
    accuracy = np.mean(preds == labels)

    print(f"Validation AUC:      {auc:.4f}")
    print(f"Best threshold:      {best_thresh:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f} (dist < {best_thresh:.4f})")

    # 7. 計算 d-prime
    genuine  = dists[labels == 1]
    imposter = dists[labels == 0]
    g_mean, i_mean = genuine.mean(), imposter.mean()
    g_var, i_var   = genuine.var(),  imposter.var()
    d_prime = abs(g_mean - i_mean) / np.sqrt(0.5 * (g_var + i_var))
    print(f"d' = {d_prime:.4f}")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
if __name__ == "__main__":
    main()
