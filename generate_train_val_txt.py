import os
import random

dataset_root = 'train_dataset'
train_txt = 'train.txt'
val_txt = 'val.txt'
val_ratio = 0.1  # 10% 做驗證

# 收集所有圖片路徑
all_images = []
for root, _, files in os.walk(dataset_root):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

# 打亂順序並分割
random.seed(42)
random.shuffle(all_images)
val_count = int(len(all_images) * val_ratio)
val_images = all_images[:val_count]
train_images = all_images[val_count:]

# 寫入 train.txt 和 val.txt
with open(train_txt, 'w') as f:
    f.write('\n'.join(train_images))

with open(val_txt, 'w') as f:
    f.write('\n'.join(val_images))

print(f"總共 {len(all_images)} 張圖片，其中 {len(train_images)} 張訓練、{len(val_images)} 張驗證")
