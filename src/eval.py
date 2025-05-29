import argparse
import numpy as np
from pathlib import Path


def calculate_d_prime(genuine_scores, imposter_scores):
    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    g_mean = np.mean(genuine_scores)
    g_var = np.var(genuine_scores)
    i_mean = np.mean(imposter_scores)
    i_var = np.var(imposter_scores)
    d_prime = np.absolute(g_mean - i_mean) / np.sqrt(0.5 * (g_var + i_var))

    return d_prime

def get_uid(file_path):
    p = Path(file_path)
    img_name = p.parts[-1]
    eye = p.parts[-2]
    uid = p.parts[-3]
    dataset = p.parts[-4]
    # print(f"dataset: {dataset}, uid: {uid}, eye: {eye}, img_name: {img_name}")

    return uid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iris Recognition Evaluation')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the results')

    args = parser.parse_args()

    genuine_scores = []
    imposter_scores = []

    """
    Each line in args.input represents a result of comparison:
        img1_path, img2_path, score
    For example:
        dataset/Ganzin-J7EF-Gaze/001/L/view_3_1.png, dataset/Ganzin-J7EF-Gaze/001/L/view_3_2.png, 0.013
        dataset/Ganzin-J7EF-Gaze/001/L/view_3_1.png, dataset/Ganzin-J7EF-Gaze/002/L/view_3_1.png, 0.996
    """
    with open(args.input, 'r') as file:
        for line in file:
            lineparts = line.split(',')
            score = float(lineparts[2].strip())

            if score < 0 or score > 1:
                print("[Error] score should be normalized to 0~1 before evaluation")
                print(line)
                exit(1)

            id1 = get_uid(lineparts[0].strip())
            id2 = get_uid(lineparts[1].strip())
            # print(f"id1: {id1}, id2: {id2}")

            if id1 == id2:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

    d_prime = calculate_d_prime(genuine_scores, imposter_scores)
    print(f"d' score = {d_prime:.4f}")
