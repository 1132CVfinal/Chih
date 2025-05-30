# validate_recognition.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset_reader import PolarDataset
from train_recognition import SiameseNet  # assuming 定義在那裡

def build_tensor_dict(polar_dataset, model, device):
    """
    先把所有 validation polar 圖都傳到 encoder，記住 embedding，
    之後 pair_list 可以直接取 embedding 來算距離。
    """
    model.eval()
    tensor_to_feat = {}
    loader = DataLoader(polar_dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            imgs, rel_paths = batch  # imgs: [B,1,256,256], rel_paths: list[str]
            imgs = imgs.to(device)
            feats = model.encoder(imgs)  # [B,128]
            feats = feats.cpu().numpy()
            for i, rp in enumerate(rel_paths):
                tensor_to_feat[rp] = feats[i].copy()  # 存一份到 dict
    return tensor_to_feat

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 載訓練好的 SiameseNet (只載 encoder 部分)
    encoder = SiameseNet().encoder.to(device)
    encoder.load_state_dict(torch.load("siamese_final.pth"))
    encoder.eval()

    # 2. 建立 polar dataset，但只需要 val_fixed.txt (所有驗證集要 normalize 的影像)
    #    假設 val_fixed.txt 每行是 "train_dataset/CASIA-Iris-Thousand/047/R/S5047R04.jpg" 之類
    val_fixed_list = "val_fixed.txt"
    polar_val = PolarDataset(list_file=val_fixed_list,
                             root_dir=".",
                             ritnet_model_path="../RITnet/best_model.pkl")

    # 3. 先把所有 validation 圖的 embedding 算好，存在 dict {rel_path: feat_vector}
    feat_dict = build_tensor_dict(polar_val, encoder, device)

    # 4. 讀 val_pairs.txt (每行: "rel_path1 rel_path2 label")
    pair_list_path = "val_pairs.txt"
    pairs = []
    with open(pair_list_path, 'r') as f:
        for line in f:
            p1, p2, lab = line.strip().split()
            lab = int(lab)
            if p1 not in feat_dict or p2 not in feat_dict:
                continue
            pairs.append((p1, p2, lab))

    # 5. 計算所有 pair 的距離與預測
    dists = []
    labels = []
    for p1, p2, lab in pairs:
        f1 = feat_dict[p1]  # numpy array (128,)
        f2 = feat_dict[p2]
        # Euclidean distance
        dist = np.linalg.norm(f1 - f2)
        dists.append(dist)
        labels.append(lab)

    dists = np.array(dists)
    labels = np.array(labels)

    # 6. 找最佳 threshold (based on ROC)
    from sklearn.metrics import roc_curve, roc_auc_score
    auc = roc_auc_score(labels, -dists)  # 負距離表達相似度
    fpr, tpr, thresholds = roc_curve(labels, -dists)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]

    preds = (dists < best_thresh).astype(int)
    accuracy = np.mean(preds == labels)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"Validation Accuracy (dist<{best_thresh}): {accuracy:.4f}")

    # 7. 也可以計算 d-prime (同 d′公式)
    genuine = dists[labels == 1]
    imposter = dists[labels == 0]
    g_mean, i_mean = genuine.mean(), imposter.mean()
    g_var, i_var = genuine.var(), imposter.var()
    d_prime = abs(g_mean - i_mean) / np.sqrt(0.5 * (g_var + i_var))
    print(f"d' = {d_prime:.4f}")

if __name__ == "__main__":
    main()
