import os

base_dir = "Ganzin-J7EF-Gaze"

persons = [f"{i:03d}" for i in range(1, 11)]  # '001'~'010'
ks = range(1, 11)  # 1~10

for person in persons:
    # 刪除 L 資料夾的 view_3_k.png
    for k in ks:
        file_path_L = os.path.join(base_dir, person, "L", f"view_3_{k}.png")
        if os.path.exists(file_path_L):
            os.remove(file_path_L)
            print(f"Deleted {file_path_L}")
        else:
            print(f"Not found: {file_path_L}")
    # 刪除 R 資料夾的 view_2_k.png
    for k in ks:
        file_path_R = os.path.join(base_dir, person, "R", f"view_2_{k}.png")
        if os.path.exists(file_path_R):
            os.remove(file_path_R)
            print(f"Deleted {file_path_R}")
        else:
            print(f"Not found: {file_path_R}")
