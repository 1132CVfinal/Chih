import argparse
import numpy as np
import cv2
from iris_recognition import IrisRecognition  # 導入我們的算法類

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')

    args = parser.parse_args()
    
    # 初始化我們的虹膜辨識系統
    iris_system = IrisRecognition()

    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        for line in in_file:
            lineparts = line.split(',')
            img1_path = lineparts[0].strip()
            img2_path = lineparts[1].strip()

            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            # 使用我們的算法計算相似度分數
            score = iris_system.compare_iris(img1, img2)

            output_line = f"{img1_path}, {img2_path}, {score}"
            print(output_line)
            out_file.write(output_line.rstrip('\n') + '\n')