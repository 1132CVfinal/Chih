import os
import cv2
from ritnet_inference import RITNetInference

def make_mask_path(img_path, input_root='train_dataset', mask_root='masks'):
    # 以原圖路徑為基準，替換根目錄，並改副檔名為_png
    rel_path = os.path.relpath(img_path, input_root)
    mask_path = os.path.join(mask_root, os.path.splitext(rel_path)[0] + '_mask.png')
    return mask_path

def ensure_dir(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

def main(image_list_txt, input_root='train_dataset', mask_root='masks', model_path='best_model.pkl'):
    ritnet = RITNetInference(model_path)

    with open(image_list_txt, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]

    for img_path in img_paths:
        full_img_path = img_path.replace('\\', os.sep)  # 兼容不同系統路徑
        print(f"Processing {full_img_path} ...")
        img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Cannot load image {full_img_path}, skipping.")
            continue

        iris_mask, pupil_mask, sclera_mask, background_mask = ritnet.segment_iris(img)

        # 這邊你可以選擇要存哪個 mask，這裡以 iris_mask 為例
        mask_to_save = iris_mask * 255  # 轉成 0-255 圖片格式

        mask_save_path = make_mask_path(full_img_path, input_root, mask_root)
        ensure_dir(mask_save_path)
        cv2.imwrite(mask_save_path, mask_to_save)
        print(f"Saved mask to {mask_save_path}")

if __name__ == "__main__":
    main('all.txt')
