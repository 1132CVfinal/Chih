import random
from collections import defaultdict

def build_val_pairs(txt_path, output_path, num_pairs=1000):
    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f.readlines()]

    label_to_paths = defaultdict(list)
    for p in paths:
        # 假設 p = "CASIA-Iris-Thousand/447/L/S5447L00.jpg"
        person_id = p.split('/')[1]
        label_to_paths[person_id].append(p)

    pairs = []
    half = num_pairs // 2

    # 正樣本
    for _ in range(half):
        person = random.choice(list(label_to_paths.keys()))
        if len(label_to_paths[person]) < 2:
            continue
        p1, p2 = random.sample(label_to_paths[person], 2)
        pairs.append((p1, p2, 1))

    # 負樣本
    people = list(label_to_paths.keys())
    for _ in range(half):
        p1_person, p2_person = random.sample(people, 2)
        p1 = random.choice(label_to_paths[p1_person])
        p2 = random.choice(label_to_paths[p2_person])
        pairs.append((p1, p2, 0))

    # 儲存到 output_path
    with open(output_path, 'w') as f:
        for p1, p2, lbl in pairs:
            f.write(f"{p1} {p2} {lbl}\n")

    print(f"Saved {len(pairs)} pairs to {output_path}")

# example usage:
# build_val_pairs('val_fixed.txt', 'val_pairs.txt', num_pairs=1000)
