import os

base_dir = "Ganzin-J7EF-Gaze"
persons = [f"{i:03d}" for i in range(1, 11)]
ks = range(1, 11)

# 預設保留的完整檔名集合
keep_files = set()

for person in persons:
    for k in ks:
        keep_files.add(os.path.join(base_dir, person, "L", f"view_3_{k}.png"))
        keep_files.add(os.path.join(base_dir, person, "R", f"view_2_{k}.png"))

for person in os.listdir(base_dir):
    person_path = os.path.join(base_dir, person)
    if not os.path.isdir(person_path):
        continue

    for side in ["L", "R"]:
        side_path = os.path.join(person_path, side)
        if not os.path.isdir(side_path):
            continue

        for filename in os.listdir(side_path):
            file_path = os.path.join(side_path, filename)
            # 只針對png檔案處理
            if filename.endswith(".png"):
                # 如果不在保留清單裡就刪除
                if file_path not in keep_files:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Kept: {file_path}")
